#!/usr/bin/env python3
"""
端到端对抗效果评估 — 一次研究，两份报告，RAGAS 对比打分。

用法:
    python eval_adversarial.py -q "你的研究问题"
    python eval_adversarial.py -q "你的研究问题" --skip-ragas

效果:
    研究执行一次，在 SYNTHESIZING 后自动保存对抗前报告，
    ADVERSARIAL 后自动保存对抗后报告，最后用 RAGAS 对比。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml

from deep_research.agents import AgentPool
from deep_research.core import Orchestrator, ResearchContext
from deep_research.core.schema import ResearchReport, WorkflowState
from deep_research.memory import KnowledgeBase, SessionMemory
from deep_research.memory.embedder import MemoryEmbedder
from deep_research.models import ModelRouter
from deep_research.planner import Planner
from deep_research.tools import (
    ArxivReaderTool,
    BrowserTool,
    CalculatorTool,
    CodeSandboxTool,
    FileReaderTool,
    NotepadTool,
    ToolRegistry,
    WebSearchTool,
)

log = logging.getLogger(__name__)


# 兼容 ragas 0.4 缺失 vertexai 模块
def _patch_vertexai() -> None:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        dummy = types.ModuleType("langchain_community.chat_models.vertexai")
        dummy.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = dummy
        import langchain_community.chat_models
        langchain_community.chat_models.vertexai = dummy


_patch_vertexai()


# =============================================================================
# 搜索结果拦截器 —— 捕获对抗阶段 Blue Agent SEARCH 的新来源
# =============================================================================
class CapturingWebSearchTool:
    """包装 WebSearchTool，拦截对抗阶段 Blue Agent 的所有搜索结果。

    Blue Agent 在 SEARCH 修复时会调用 web_search 获取新来源，
    这些结果对 RAGAS Faithfulness 评估至关重要——否则对抗后报告
    中新增/修正的 claims 无法在原始 contexts 中找到支撑。
    """

    name = "web_search"

    def __init__(self, delegate, capture_list: list[str]) -> None:
        self._delegate = delegate
        self._capture = capture_list

    @staticmethod
    def get_schema() -> dict:
        return WebSearchTool.get_schema()

    async def execute(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        results = await self._delegate.execute(query, num_results=num_results)
        for r in results:
            if isinstance(r, dict) and "error" not in r:
                snippet = r.get("snippet", "") or r.get("content", "")
                if len(snippet) >= 20:
                    self._capture.append(snippet[:2000])
        return results


# =============================================================================
# 进度输出（带 flush，防止缓冲导致看起来"卡住"）
# =============================================================================
def _p(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# 数据容器
# =============================================================================
@dataclass
class EvalData:
    query: str = ""
    report_before: ResearchReport | None = None   # SYNTHESIZING 后
    report_after: ResearchReport | None = None     # ADVERSARIAL 后
    contexts: list[str] = field(default_factory=list)
    contexts_after: list[str] = field(default_factory=list)  # 对抗阶段 Blue SEARCH 新增
    sources: list[dict] = field(default_factory=list)


# =============================================================================
# Orchestrator 构建（注入保存 hook）
# =============================================================================
def _create_tool_registry(capture_list: list[str] | None = None) -> ToolRegistry:
    """创建工具注册表。若提供 capture_list，则 WebSearchTool 会被包装以拦截搜索结果。"""
    r = ToolRegistry()
    ws = WebSearchTool()
    if capture_list is not None:
        ws = CapturingWebSearchTool(ws, capture_list)
    r.register(ws)
    r.register(BrowserTool())
    r.register(ArxivReaderTool())
    r.register(FileReaderTool())
    r.register(CodeSandboxTool())
    r.register(CalculatorTool())
    r.register(NotepadTool())
    return r


def _extract_contexts(results: list) -> list[str]:
    """从 agent trajectory 自动提取检索上下文。"""
    contexts: list[str] = []
    for r in results:
        for step in (r.trajectory or []):
            if step.get("role") != "tool":
                continue
            res = step.get("result")
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        s = item.get("snippet", "") or item.get("summary", "")
                        if len(s) >= 20:
                            contexts.append(s[:2000])
            elif isinstance(res, dict):
                s = res.get("content", "")
                if isinstance(s, str) and len(s) >= 20:
                    contexts.append(s[:2000])
                for p in res.get("papers", []):
                    if isinstance(p, dict) and len(p.get("summary", "")) >= 20:
                        contexts.append(p["summary"][:2000])
    seen = set()
    uniq = []
    for c in contexts:
        k = c[:100]
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def build_orchestrator(
    config: dict,
    session_memory: SessionMemory,
    knowledge_base: KnowledgeBase,
    ed: EvalData,
) -> Orchestrator:
    model_cfg = config.get("model", {})
    backend = model_cfg.get("backend", "deepseek")

    client_kwargs: dict = {}
    for k in ("base_model", "base_url", "api_key", "temperature", "top_p", "max_tokens"):
        if k in model_cfg:
            client_kwargs[k] = model_cfg[k]

    llm_client = ModelRouter.create_backend(backend, **client_kwargs)
    log.info("LLM: %s", backend)

    planner = Planner(llm_client)
    registry = _create_tool_registry(capture_list=ed.contexts_after)

    sampling_cfg = model_cfg.get("backend_sampling", {})
    backend_defaults = sampling_cfg.get(backend, {})
    module_overrides = sampling_cfg.get("modules", {})
    backend_map = model_cfg.get("backend_mapping", {})

    _T2M = {
        "search": "solver", "analyze": "solver", "verify": "solver",
        "synthesize": "summarizer", "red_agent": "red_agent",
        "blue_agent": "blue_agent", "issue_arbiter": "issue_arbiter",
    }

    def policy_factory(task_type: str = "search"):
        mod = _T2M.get(task_type, "solver")
        m = dict(backend_defaults)
        m.update(client_kwargs)
        if mod in module_overrides:
            m.update(module_overrides[mod])
        mb = backend_map.get(mod, backend)
        return ModelRouter.create_backend(mb, **m)

    def tools_factory():
        return list(registry._tools.values())

    orch_cfg = config.get("orchestrator", {})
    pool = AgentPool(
        policy_factory=policy_factory,
        tools_factory=tools_factory,
        max_idle=orch_cfg.get("max_concurrent", 5),
        config=config,
        session_memory=session_memory,
        knowledge_base=knowledge_base,
    )

    arbiter = None
    if config.get("adversarial", {}).get("issue_arbiter", {}).get("enabled", True):
        try:
            arbiter = policy_factory("issue_arbiter")
        except Exception:
            pass

    orch = Orchestrator(
        planner=planner, agent_pool=pool,
        session_memory=session_memory, knowledge_base=knowledge_base,
        config=config, issue_arbiter_client=arbiter,
    )

    # ---- Hook: SYNTHESIZING 完成后保存对抗前报告 ----
    orig_synth = orch._do_synthesizing
    async def synth_hook():
        r = await orig_synth()
        if orch._report is not None:
            ed.report_before = orch._report
            ed.contexts = _extract_contexts(orch._results)
            ed.sources = getattr(orch._report, "sources", [])
            ed.query = orch._query
            _p(f"  [hook] 对抗前报告已保存 (conf={getattr(orch._report, 'confidence', 0):.2f}, "
               f"ctx={len(ed.contexts)}条)")
        return r

    # ---- Hook: ADVERSARIAL 完成后保存对抗后报告 ----
    orig_adv = orch._do_adversarial
    async def adv_hook():
        r = await orig_adv()
        if orch._report is not None:
            ed.report_after = orch._report
            _p(f"  [hook] 对抗后报告已保存 (conf={getattr(orch._report, 'confidence', 0):.2f}, "
               f"rounds={getattr(orch._report, 'adversarial_rounds', 0)}, "
               f"新增搜索={len(ed.contexts_after)}条)")
        return r

    orch._state_handlers[WorkflowState.SYNTHESIZING] = synth_hook
    orch._state_handlers[WorkflowState.ADVERSARIAL] = adv_hook

    return orch


# =============================================================================
# RAGAS 比较
# =============================================================================
@dataclass
class RagasResult:
    faith_before: float
    faith_after: float
    relevancy_before: float
    relevancy_after: float
    # 对抗五维独立评分（AspectCritique 0/1 → 聚合为 0-1）
    aspect_before: dict[str, float] = field(default_factory=dict)
    aspect_after: dict[str, float] = field(default_factory=dict)
    improved: bool = False


# 对抗五维 → AspectCritique 定义（独立于 Red Agent 的二次评判）
ADVERSARIAL_ASPECTS = {
    "factual": "Does the report contain accurate facts (dates, numbers, names, statistics) without factual errors or internal contradictions in factual claims?",
    "hallucination": "Does the report avoid fabricating unsupported claims, fake details, or presenting speculation and inference as established fact?",
    "logic": "Is the report logically coherent with consistent reasoning, no self-contradictions, no causal fallacies, and valid argument chains?",
    "source": "Does the report cite credible, authoritative, and timely sources rather than relying on low-quality or unverifiable references?",
    "coverage": "Does the report comprehensively address all subtopics implied by the query, presenting balanced viewpoints without major omissions?",
}


async def run_ragas(ed: EvalData, config: dict) -> RagasResult | None:
    """用 ragas 0.4 多指标评估：AnswerRelevancy + AspectCritique(五维) + Faithfulness。

    指标选择逻辑：
      - AnswerRelevancy：不依赖 contexts，衡量报告是否切题回答 query
      - AspectCritique × 5：LLM 独立评判报告的五维质量（对抗模块的二次校验）
      - Faithfulness：保留作为参考，但不再作为 primary KPI
    """
    try:
        from openai import OpenAI
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas.evaluation import aevaluate
        from ragas.llms import llm_factory
        from ragas.metrics import AspectCritique, AnswerRelevancy
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_core.embeddings import Embeddings
    except ImportError as e:
        _p(f"  RAGAS 指标导入失败: {e}。安装: pip install ragas>=0.2 langchain-openai langchain-core")
        return None

    backend = config.get("model", {}).get("backend", "deepseek")
    try:
        llm_client = ModelRouter.create_backend(backend, temperature=0.1, max_tokens=16384)
    except Exception as e:
        _p(f"  评判 LLM 不可用: {e}")
        return None

    import httpx
    openai_client = OpenAI(
        base_url=llm_client.base_url,
        api_key=llm_client.api_key,
        timeout=300,
        max_retries=2,
        http_client=httpx.Client(
            transport=httpx.HTTPTransport(verify=False),
            timeout=300,
        ),
    )

    model_name = getattr(llm_client, "model_name", "deepseek-chat")
    ragas_llm = llm_factory(model_name, client=openai_client)
    ragas_llm.model_args["max_tokens"] = 16384

    # --- Embeddings（AnswerRelevancy 需要）---
    # MemoryEmbedder 没有标准的 embed_documents/embed_query 接口，
    # 用 sentence-transformers 直连（与 MemoryEmbedder 同源模型）
    try:
        from sentence_transformers import SentenceTransformer

        class _STEmbeddings(Embeddings):
            """LangChain 兼容的 sentence-transformers 嵌入适配器。"""
            def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
                self._model = SentenceTransformer(model_name)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._model.encode(texts, normalize_embeddings=True).tolist()

            def embed_query(self, text: str) -> list[float]:
                return self._model.encode(text, normalize_embeddings=True).tolist()

        ragas_embeddings = LangchainEmbeddingsWrapper(_STEmbeddings())
        _p("  嵌入模型已加载 (BAAI/bge-small-zh-v1.5)")
    except Exception:
        _p("  ⚠ 嵌入模型不可用，跳过 AnswerRelevancy")
        ragas_embeddings = None

    # --- 构建指标列表 ---
    metrics: list = []

    # 1. AnswerRelevancy（核心指标，不依赖 contexts）
    if ragas_embeddings is not None:
        metrics.append(AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings))

    # 2. AspectCritique × 5（独立评判对抗五维）
    aspect_metrics: dict[str, AspectCritique] = {}
    for dim_key, definition in ADVERSARIAL_ASPECTS.items():
        ac = AspectCritique(name=dim_key, definition=definition, llm=ragas_llm)
        aspect_metrics[dim_key] = ac
        metrics.append(ac)

    # 3. Faithfulness（保留参考）
    metrics.append(Faithfulness(llm=ragas_llm))

    # --- 准备数据 ---
    ctx = ed.contexts or ["(no contexts)"]
    ctx_after_raw = ctx + (ed.contexts_after or [])
    seen = set()
    ctx_after: list[str] = []
    for c in ctx_after_raw:
        k = c[:100]
        if k not in seen:
            seen.add(k)
            ctx_after.append(c)
    if ed.contexts_after:
        _p(f"  📎 对抗阶段捕获 {len(ed.contexts_after)} 条新搜索结果，"
           f"合并后 contexts: {len(ctx)} → {len(ctx_after)} 条")

    MAX_RESPONSE_CHARS = 24000
    before_text = ed.report_before.content
    after_text = ed.report_after.content
    if len(after_text) > MAX_RESPONSE_CHARS:
        _p(f"  ⚠ 对抗后报告过长 ({len(after_text)}字符)，截断至 {MAX_RESPONSE_CHARS} 字符")
        after_text = after_text[:MAX_RESPONSE_CHARS] + "\n\n[... 后续内容已截断 ...]"
    if len(before_text) > MAX_RESPONSE_CHARS:
        before_text = before_text[:MAX_RESPONSE_CHARS] + "\n\n[... 后续内容已截断 ...]"

    before_ds = EvaluationDataset(samples=[
        SingleTurnSample(user_input=ed.query, response=before_text, retrieved_contexts=ctx)
    ])
    after_ds = EvaluationDataset(samples=[
        SingleTurnSample(user_input=ed.query, response=after_text, retrieved_contexts=ctx_after)
    ])

    # --- 分别评估 ---
    import math
    fb, fa = float("nan"), float("nan")
    rb_val, ra_val = float("nan"), float("nan")
    aspect_before: dict[str, float] = {}
    aspect_after: dict[str, float] = {}

    def _extract_scores(df) -> dict[str, float]:
        """从 ragas 结果 DataFrame 提取所有指标分数。"""
        scores: dict[str, float] = {}
        for col in df.columns:
            try:
                scores[col] = float(df[col].iloc[0])
            except (ValueError, TypeError):
                pass
        return scores

    try:
        rb = await aevaluate(dataset=before_ds, metrics=metrics)
        dfb = rb.to_pandas()
        sb = _extract_scores(dfb)
        fb = sb.get("faithfulness", float("nan"))
        rb_val = sb.get("answer_relevancy", float("nan"))
        aspect_before = {k: sb.get(k, float("nan")) for k in ADVERSARIAL_ASPECTS}
    except Exception as e:
        _p(f"  ⚠ 对抗前 RAGAS 评估失败: {e}")

    try:
        ra = await aevaluate(dataset=after_ds, metrics=metrics)
        dfa = ra.to_pandas()
        sa = _extract_scores(dfa)
        fa = sa.get("faithfulness", float("nan"))
        ra_val = sa.get("answer_relevancy", float("nan"))
        aspect_after = {k: sa.get(k, float("nan")) for k in ADVERSARIAL_ASPECTS}
    except Exception as e:
        _p(f"  ⚠ 对抗后 RAGAS 评估失败: {e}")

    # 综合判断：AnswerRelevancy + 五维均分 任意一项改善即算改善
    aspect_before_avg = (
        sum(v for v in aspect_before.values() if not math.isnan(v)) / max(len(aspect_before), 1)
        if aspect_before else float("nan")
    )
    aspect_after_avg = (
        sum(v for v in aspect_after.values() if not math.isnan(v)) / max(len(aspect_after), 1)
        if aspect_after else float("nan")
    )

    ar_improved = (not math.isnan(rb_val) and not math.isnan(ra_val) and ra_val > rb_val)
    aspect_improved = (not math.isnan(aspect_before_avg) and not math.isnan(aspect_after_avg)
                       and aspect_after_avg > aspect_before_avg)

    return RagasResult(
        faith_before=fb, faith_after=fa,
        relevancy_before=rb_val, relevancy_after=ra_val,
        aspect_before=aspect_before, aspect_after=aspect_after,
        improved=ar_improved or aspect_improved,
    )


# =============================================================================
# 报告保存
# =============================================================================
def save_report(ed: EvalData, ragas: RagasResult | None, out_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sq = "".join(c if c.isalnum() or c in "_-" else "_" for c in ed.query[:30])
    path = out_dir / f"eval_adversarial_{sq}_{ts}.json"

    rpt = {
        "query": ed.query,
        "timestamp": datetime.now().isoformat(),
        "n_contexts": len(ed.contexts),
        "n_contexts_after": len(ed.contexts) + len(ed.contexts_after),
        "n_contexts_captured": len(ed.contexts_after),
        "report_before": {
            "confidence": getattr(ed.report_before, "confidence", None),
            "length": len(ed.report_before.content) if ed.report_before else 0,
        },
        "report_after": {
            "confidence": getattr(ed.report_after, "confidence", None),
            "length": len(ed.report_after.content) if ed.report_after else 0,
            "adversarial_rounds": getattr(ed.report_after, "adversarial_rounds", None),
            "final_score": getattr(ed.report_after, "final_score", None),
            "dimension_scores": {
                d.value: s for d, s in (
                    getattr(ed.report_after, "dimension_scores", {}) or {}
                ).items()
            } if ed.report_after else {},
        },
    }
    if ragas:
        rpt["ragas"] = {
            "faithfulness": {"before": ragas.faith_before, "after": ragas.faith_after,
                             "delta": round(ragas.faith_after - ragas.faith_before, 4)},
            "answer_relevancy": {"before": ragas.relevancy_before, "after": ragas.relevancy_after,
                                 "delta": round(ragas.relevancy_after - ragas.relevancy_before, 4)},
            "aspect_critique": {
                "before": ragas.aspect_before,
                "after": ragas.aspect_after,
            },
            "improved": ragas.improved,
        }

    path.write_text(json.dumps(rpt, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# =============================================================================
# main
# =============================================================================
async def main() -> int:
    parser = argparse.ArgumentParser(description="对抗模块端到端效果评估")
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-c", "--config", default="config/default.yaml")
    parser.add_argument("-o", "--output", default="outputs")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()

    # logging 配置
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for n in ("openai", "httpx", "httpcore", "primp", "urllib3", "aiosqlite"):
        logging.getLogger(n).setLevel(logging.WARNING)

    _p()
    _p("=" * 60)
    _p(" 对抗模块效果评估")
    _p("=" * 60)
    _p(f" 问题: {args.query}")
    _p(f" 配置: {args.config}")
    _p(f" RAGAS: {'跳过' if args.skip_ragas else '启用'}")
    _p()

    # [1] 配置
    _p("[1/4] 加载配置 ...")
    cp = Path(args.config)
    config = yaml.safe_load(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    if "adversarial" not in config:
        config["adversarial"] = {}
    config["adversarial"]["enabled"] = True
    config["adversarial"]["entry_confidence_threshold"] = 0.95  # 强制对抗一定执行
    _p("      OK (对抗强制开启)")

    # [2] 知识库 + embedder
    _p("[2/4] 初始化知识库和嵌入模型 ...")
    mc = config.get("memory", {})
    sm = SessionMemory(db_path=mc.get("session_db_path", "data/session_memory.db"))
    kb = KnowledgeBase(
        db_path=mc.get("knowledge_db_path", "data/knowledge_base.db"),
        embedder=MemoryEmbedder(),
        config=mc,
    )
    await sm.initialize()
    await kb.initialize()
    _p("      OK")

    ed = EvalData(query=args.query)

    try:
        # [3] 构建 orchestrator
        _p("[3/4] 构建编排器 ...")
        orch = build_orchestrator(config, sm, kb, ed)
        oc = config.get("orchestrator", {})
        timeout = oc.get("global_timeout_seconds", 300)
        _p(f"      OK (timeout={timeout}s)")

        # [4] 执行研究
        _p(f"[4/4] 执行深度研究 ...")
        _p(f"      流程: Planner -> Search -> Synthesize -> Adversarial")
        _p()

        ctx = ResearchContext(
            topic=args.query,
            max_iterations=oc.get("max_replan_rounds", 3),
            enable_adversarial=True,
        )

        t0 = time.time()
        state = await orch.run(ctx, timeout_seconds=timeout)
        elapsed = time.time() - t0

        _p(f"\n研究完成 ({elapsed:.0f}s, 状态={state.value})")

        # ---- 输出对比 ----
        if ed.report_before is None:
            _p("FAIL: 未生成报告")
            return 1

        _p()
        _p("-" * 40)
        _p(f" 对抗前 | conf={getattr(ed.report_before, 'confidence', 0):.2f} | "
           f"len={len(ed.report_before.content)} | ctx={len(ed.contexts)}条")

        if ed.report_after is None:
            _p(f" 对抗后 | (未执行)")
        else:
            _p(f" 对抗后 | conf={getattr(ed.report_after, 'confidence', 0):.2f} | "
               f"len={len(ed.report_after.content)} | "
               f"rounds={getattr(ed.report_after, 'adversarial_rounds', 0)} | "
               f"score={getattr(ed.report_after, 'final_score', 0):.1f}")
            _p(f"         五维: {getattr(ed.report_after, 'dimension_scores', {})}")
        _p("-" * 40)

        # ---- RAGAS ----
        ragas: RagasResult | None = None
        if not args.skip_ragas and ed.report_after is not None:
            _p()
            _p("RAGAS 评估中 ...")
            ragas = await run_ragas(ed, config)

            if ragas:
                import math
                _p()
                _p("  ═══════ RAGAS 多指标评估 ═══════")
                _p(f"  AnswerRelevancy:  {ragas.relevancy_before:.4f} -> {ragas.relevancy_after:.4f}  "
                   f"({'++' if ragas.relevancy_after > ragas.relevancy_before else '--'})")
                _p("  AspectCritique (独立评判对抗五维):")
                for dim in ADVERSARIAL_ASPECTS:
                    ab = ragas.aspect_before.get(dim, float("nan"))
                    aa = ragas.aspect_after.get(dim, float("nan"))
                    if math.isnan(ab) and math.isnan(aa):
                        continue
                    arrow = "++" if (not math.isnan(ab) and not math.isnan(aa) and aa > ab) else \
                            "--" if (not math.isnan(ab) and not math.isnan(aa) and aa < ab) else "=="
                    _p(f"    {dim:<14}: {ab:.4f} -> {aa:.4f}  ({arrow})")
                _p(f"  Faithfulness:     {ragas.faith_before:.4f} -> {ragas.faith_after:.4f}  (参考)")
                _p()
                _p(f"  结论: {'✅ 对抗模块有效改善报告质量' if ragas.improved else '⚠️ 本次未显著改善'}")

        # ---- 保存 ----
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = save_report(ed, ragas, out_dir)
        _p(f"\n报告: {saved}")

        return 0

    except Exception:
        logging.exception("异常")
        return 1
    finally:
        await sm.close()
        await kb.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
