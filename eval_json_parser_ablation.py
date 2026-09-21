#!/usr/bin/env python3
"""
eval_json_parser_ablation.py
================================================================================
端到端 JSON 多层容错解析消融 + 红蓝对抗效果评估。

运行一次完整研究流程（JSON 容错 + 对抗全部开启），在 4 个 JSON 解析节点
拦截原始 LLM 输出，同时记录原生 json.loads 基线与项目多层容错解析的结果。
同时保存 Summarizer 后的对抗前报告与红蓝对抗后的最终报告，用规则评测和
RAGAS（Faithfulness / AnswerRelevancy / AspectCritique）按章节分别评估。

用法:
    python eval_json_parser_ablation.py -q "你的研究问题"
    python eval_json_parser_ablation.py -q "你的研究问题" --skip-ragas
    python eval_json_parser_ablation.py --bench-id tech_001
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
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
from deep_research.agents.blue_agent import BlueTeamAgent
from deep_research.agents.red_agent import RedTeamAgent
from deep_research.core import Orchestrator, ResearchContext
from deep_research.core.issue_merger import IssueMerger
from deep_research.core.schema import ResearchReport, WorkflowState
from deep_research.evaluation.benchmarks import ResearchBench
from deep_research.memory import KnowledgeBase, SessionMemory
from deep_research.memory.embedder import MemoryEmbedder
from deep_research.models import ModelRouter
from deep_research.planner.planner import Planner
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
from json_repair import loads as json_repair_loads

log = logging.getLogger(__name__)


# =============================================================================
# 兼容 ragas 0.4 缺失 vertexai 模块（已有项目中的 workaround）
# =============================================================================
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
# JSON 解析拦截器：在单轮运行中同时记录基线与多层容错解析结果
# =============================================================================
@dataclass
class JsonParseRecord:
    stage: str
    raw_output: str
    baseline_success: bool = False
    baseline_error: str = ""
    multi_layer_success: bool = False
    multi_layer_error: str = ""
    depth: int = -1
    depth_name: str = "failed"
    raw_len: int = 0


class JsonParseInterceptor:
    """在 Planner / Red / Blue / IssueMerger 四个解析入口插入记录 hook。

    不修改业务代码，通过 monkey-patch 实现。运行结束后恢复原始方法。
    """

    def __init__(self) -> None:
        self.records: list[JsonParseRecord] = []
        self._originals: dict[str, Any] = {}
        self._lock = False

    # ------------------------------------------------------------------
    # 安装 / 卸载
    # ------------------------------------------------------------------
    def install(self) -> None:
        self._originals["Planner._parse_plan"] = Planner._parse_plan
        Planner._parse_plan = self._wrap_method(
            Planner._parse_plan, stage="planner", raw_name="json_str"
        )

        self._originals["RedTeamAgent._parse_dimension_json"] = RedTeamAgent._parse_dimension_json
        RedTeamAgent._parse_dimension_json = self._wrap_method(
            RedTeamAgent._parse_dimension_json, stage="red_agent", raw_name="content"
        )

        self._originals["BlueTeamAgent._parse_fix_json"] = BlueTeamAgent._parse_fix_json
        BlueTeamAgent._parse_fix_json = self._wrap_method(
            BlueTeamAgent._parse_fix_json, stage="blue_agent_fix", raw_name="text"
        )

        self._originals["BlueTeamAgent._parse_verify_json"] = BlueTeamAgent._parse_verify_json
        BlueTeamAgent._parse_verify_json = self._wrap_method(
            BlueTeamAgent._parse_verify_json, stage="blue_agent_verify", raw_name="text"
        )

        # IssueMerger._parse_arbitration_json 是 classmethod，需要特殊处理
        self._originals["IssueMerger._parse_arbitration_json"] = IssueMerger._parse_arbitration_json
        original_func = IssueMerger._parse_arbitration_json.__func__
        IssueMerger._parse_arbitration_json = classmethod(
            self._wrap_method(original_func, stage="issue_merger", raw_name="content")
        )

    def uninstall(self) -> None:
        for name, orig in self._originals.items():
            if name == "IssueMerger._parse_arbitration_json":
                IssueMerger._parse_arbitration_json = orig
            elif name == "Planner._parse_plan":
                Planner._parse_plan = orig
            elif name == "RedTeamAgent._parse_dimension_json":
                RedTeamAgent._parse_dimension_json = orig
            elif name == "BlueTeamAgent._parse_fix_json":
                BlueTeamAgent._parse_fix_json = orig
            elif name == "BlueTeamAgent._parse_verify_json":
                BlueTeamAgent._parse_verify_json = orig
        self._originals.clear()

    def __enter__(self) -> JsonParseInterceptor:
        self.install()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.uninstall()

    # ------------------------------------------------------------------
    # 包装方法
    # ------------------------------------------------------------------
    def _wrap_method(self, original, stage: str, raw_name: str):
        def wrapper(*args, **kwargs):
            if self._lock:
                return original(*args, **kwargs)
            self._lock = True
            try:
                raw = self._get_raw(args, kwargs, raw_name)
                record = self._evaluate_baseline(raw, stage)
                self.records.append(record)

                try:
                    result = original(*args, **kwargs)
                    record.multi_layer_success = self._is_success(result, stage)
                    record.multi_layer_error = ""
                    return result
                except Exception as e:
                    record.multi_layer_success = False
                    record.multi_layer_error = type(e).__name__
                    raise
            finally:
                self._lock = False

        return wrapper

    @staticmethod
    def _get_raw(args, kwargs, raw_name: str) -> str:
        if raw_name in kwargs:
            return kwargs[raw_name] or ""
        # 方法签名中 raw 参数位置：self 之后，所以是 args[1]
        if len(args) > 1:
            return args[1] or ""
        return ""

    # ------------------------------------------------------------------
    # 基线 vs 多层解析能力评估
    # ------------------------------------------------------------------
    def _evaluate_baseline(self, raw: str, stage: str) -> JsonParseRecord:
        raw = raw or ""
        record = JsonParseRecord(stage=stage, raw_output=raw[:8000], raw_len=len(raw))

        # 基线：直接 json.loads
        try:
            json.loads(raw)
            record.baseline_success = True
            record.depth = 0
            record.depth_name = "direct"
            record.multi_layer_success = True  # 基线能过，多层也一定能过
            return record
        except Exception as e:
            record.baseline_success = False
            record.baseline_error = type(e).__name__

        # 多层深度探测：逐层清洗，看哪一层能解析成功
        depth, depth_name, cleaned = self._detect_depth(raw)
        record.depth = depth
        record.depth_name = depth_name
        record.multi_layer_success = depth >= 0
        return record

    @staticmethod
    def _detect_depth(raw: str) -> tuple[int, str, str]:
        """模拟项目多层解析，逐层递增清洗力度，返回首次成功的深度。"""

        def strip_markdown_fences(text: str) -> str:
            lines = text.strip().splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()

        def extract_code_block(text: str) -> str:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                return m.group(1).strip()
            m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            if m:
                return m.group(1).strip()
            return text

        def find_balanced_braces(text: str) -> str:
            start = text.find("{")
            if start == -1:
                return text
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
            return text

        def remove_trailing_commas(text: str) -> str:
            return re.sub(r",(\s*[}\]])", r"\1", text)

        def remove_comments(text: str) -> str:
            return "\n".join(
                line[: line.index("//")] if "//" in line else line
                for line in text.splitlines()
            )

        def normalize_full_width(text: str) -> str:
            for src, dst in (
                ("｛", "{"), ("｝", "}"), ("［", "["), ("］", "]"),
                ("：", ":"), ("，", ","), ("“", '"'), ("”", '"'),
                ("‘", "'"), ("’", "'"),
            ):
                text = text.replace(src, dst)
            return text

        layers = [
            ("markdown_fences", strip_markdown_fences),
            ("code_block_extract", extract_code_block),
            ("balanced_braces", find_balanced_braces),
            ("trailing_comma", remove_trailing_commas),
            ("comments", remove_comments),
            ("full_width", normalize_full_width),
        ]

        current = raw
        for depth, (name, cleaner) in enumerate(layers, start=1):
            try:
                current = cleaner(current)
                json.loads(current)
                return depth, name, current
            except Exception:
                continue

        # 最后一层：json_repair 兜底
        try:
            repaired = json_repair_loads(raw)
            if isinstance(repaired, (dict, list)):
                return len(layers) + 1, "json_repair", raw
        except Exception:
            pass

        return -1, "failed", raw

    @staticmethod
    def _is_success(result: Any, stage: str) -> bool:
        """根据各 stage 返回结果判断是否真正解析成功。"""
        if result is None:
            return False
        if stage == "planner":
            try:
                dag, subtasks = result
                return bool(subtasks)
            except Exception:
                return False
        if stage == "red_agent":
            from deep_research.core.schema import DimensionAttack
            if isinstance(result, DimensionAttack):
                summary = getattr(result, "analysis_summary", "") or ""
                return "JSON" not in summary and "解析" not in summary and "失败" not in summary
            return False
        if stage in ("blue_agent_fix", "blue_agent_verify"):
            if isinstance(result, dict):
                return bool(result)
            return False
        if stage == "issue_merger":
            return isinstance(result, list) and len(result) > 0
        return False


# =============================================================================
# 搜索结果拦截器（复用 eval_adversarial.py 中的实现）
# =============================================================================
class CapturingWebSearchTool:
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
# 数据容器
# =============================================================================
@dataclass
class EvalData:
    query: str = ""
    report_before: ResearchReport | None = None
    report_after: ResearchReport | None = None
    contexts: list[str] = field(default_factory=list)
    contexts_after: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    json_records: list[JsonParseRecord] = field(default_factory=list)


# =============================================================================
# 辅助函数
# =============================================================================
def _p(msg: str = "") -> None:
    print(msg, flush=True)


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


def _create_tool_registry(capture_list: list[str] | None = None) -> ToolRegistry:
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


# =============================================================================
# Orchestrator 构建（注入保存 hook + 搜索结果捕获）
# =============================================================================
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
    orig_synth = orch._state_handlers[WorkflowState.SYNTHESIZING]
    async def synth_hook():
        r = await orig_synth()
        if orch._report is not None:
            ed.report_before = copy.deepcopy(orch._report)
            ed.contexts = _extract_contexts(orch._results)
            ed.sources = getattr(orch._report, "sources", [])
            ed.query = orch._query
            _p(f"  [hook] 对抗前报告已保存 (conf={getattr(orch._report, 'confidence', 0):.2f}, "
               f"ctx={len(ed.contexts)}条)")
        return r

    # ---- Hook: ADVERSARIAL 完成后保存对抗后报告 ----
    orig_adv = orch._state_handlers[WorkflowState.ADVERSARIAL]
    async def adv_hook():
        r = await orig_adv()
        if orch._report is not None:
            ed.report_after = copy.deepcopy(orch._report)
            _p(f"  [hook] 对抗后报告已保存 (conf={getattr(orch._report, 'confidence', 0):.2f}, "
               f"rounds={getattr(orch._report, 'adversarial_rounds', 0)}, "
               f"新增搜索={len(ed.contexts_after)}条)")
        return r

    orch._state_handlers[WorkflowState.SYNTHESIZING] = synth_hook
    orch._state_handlers[WorkflowState.ADVERSARIAL] = adv_hook

    return orch


# =============================================================================
# 报告按章节拆分
# =============================================================================
def split_report_by_sections(report_text: str, min_chars: int = 200) -> list[str]:
    """按二级标题（##）拆分报告，过滤掉太短的章节。"""
    if not report_text:
        return []
    # 统一标题格式，按 ## 切分
    parts = re.split(r"\n##\s+", report_text)
    sections = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i == 0 and not part.startswith("#"):
            # 第一个片段可能是标题前导内容，太短则跳过
            if len(part) < min_chars:
                continue
        sections.append(part)
    return [s for s in sections if len(s) >= min_chars]


# =============================================================================
# RAGAS 评估：按章节分别评估，再取平均
# =============================================================================
@dataclass
class RagasScores:
    faithfulness_before: float = float("nan")
    faithfulness_after: float = float("nan")
    relevancy_before: float = float("nan")
    relevancy_after: float = float("nan")
    aspect_before: dict[str, float] = field(default_factory=dict)
    aspect_after: dict[str, float] = field(default_factory=dict)
    improved: bool = False


ADVERSARIAL_ASPECTS = {
    "factual": "Does the report contain accurate facts (dates, numbers, names, statistics) without factual errors or internal contradictions in factual claims?",
    "hallucination": "Does the report avoid fabricating unsupported claims, fake details, or presenting speculation and inference as established fact?",
    "logic": "Is the report logically coherent with consistent reasoning, no self-contradictions, no causal fallacies, and valid argument chains?",
    "source": "Does the report cite credible, authoritative, and timely sources rather than relying on low-quality or unverifiable references?",
    "coverage": "Does the report comprehensively address all subtopics implied by the query, presenting balanced viewpoints without major omissions?",
}


async def run_ragas_by_sections(ed: EvalData, config: dict) -> RagasScores | None:
    """用 RAGAS 按章节评估对抗前/后报告。"""
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

    # Embeddings
    try:
        from sentence_transformers import SentenceTransformer

        class _STEmbeddings(Embeddings):
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

    # 指标列表
    metrics: list = []
    if ragas_embeddings is not None:
        metrics.append(AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings))
    metrics.append(Faithfulness(llm=ragas_llm))
    for dim_key, definition in ADVERSARIAL_ASPECTS.items():
        metrics.append(AspectCritique(name=dim_key, definition=definition, llm=ragas_llm))

    # 准备上下文
    ctx_before = ed.contexts or ["(no contexts)"]
    ctx_after_raw = ctx_before + (ed.contexts_after or [])
    seen = set()
    ctx_after: list[str] = []
    for c in ctx_after_raw:
        k = c[:100]
        if k not in seen:
            seen.add(k)
            ctx_after.append(c)
    if ed.contexts_after:
        _p(f"  📎 对抗阶段捕获 {len(ed.contexts_after)} 条新搜索结果，"
           f"合并后 contexts: {len(ctx_before)} → {len(ctx_after)} 条")

    # 按章节拆分
    before_sections = split_report_by_sections(ed.report_before.content if ed.report_before else "")
    after_sections = split_report_by_sections(ed.report_after.content if ed.report_after else "")

    if not before_sections or not after_sections:
        _p("  ⚠ 报告为空或无法拆分章节")
        return None

    _p(f"  按章节评估：对抗前 {len(before_sections)} 节，对抗后 {len(after_sections)} 节")

    async def _eval_sections(sections: list[str], contexts: list[str]) -> dict[str, float]:
        """对章节列表跑 RAGAS，按指标取平均。"""
        samples = []
        for sec in sections:
            # 每个章节都使用同一个 query，AnswerRelevancy 衡量章节是否切题
            samples.append(SingleTurnSample(
                user_input=ed.query,
                response=sec[:4000],  # 单节限制长度
                retrieved_contexts=contexts[:10],  # 限制上下文数量
            ))
        ds = EvaluationDataset(samples=samples)
        try:
            result = await aevaluate(dataset=ds, metrics=metrics)
            df = result.to_pandas()
        except Exception as e:
            _p(f"  ⚠ RAGAS 评估失败: {e}")
            return {}

        scores: dict[str, list[float]] = {}
        for col in df.columns:
            if col in ("user_input", "response", "retrieved_contexts"):
                continue
            vals = []
            for v in df[col]:
                try:
                    fv = float(v)
                    if not (math.isnan(fv) or math.isinf(fv)):
                        vals.append(fv)
                except (ValueError, TypeError):
                    pass
            if vals:
                scores[col] = vals
        return {k: sum(v) / len(v) for k, v in scores.items()}

    import math
    before_scores = await _eval_sections(before_sections, ctx_before)
    after_scores = await _eval_sections(after_sections, ctx_after)

    def _get(scores: dict[str, float], key: str) -> float:
        return scores.get(key, float("nan"))

    aspect_before = {k: _get(before_scores, k) for k in ADVERSARIAL_ASPECTS}
    aspect_after = {k: _get(after_scores, k) for k in ADVERSARIAL_ASPECTS}

    aspect_before_avg = sum(v for v in aspect_before.values() if not math.isnan(v)) / max(1, sum(1 for v in aspect_before.values() if not math.isnan(v)))
    aspect_after_avg = sum(v for v in aspect_after.values() if not math.isnan(v)) / max(1, sum(1 for v in aspect_after.values() if not math.isnan(v)))

    rb = _get(before_scores, "answer_relevancy")
    ra = _get(after_scores, "answer_relevancy")
    fb = _get(before_scores, "faithfulness")
    fa = _get(after_scores, "faithfulness")

    improved = False
    if not math.isnan(rb) and not math.isnan(ra) and ra > rb:
        improved = True
    if not math.isnan(aspect_before_avg) and not math.isnan(aspect_after_avg) and aspect_after_avg > aspect_before_avg:
        improved = True

    return RagasScores(
        faithfulness_before=fb,
        faithfulness_after=fa,
        relevancy_before=rb,
        relevancy_after=ra,
        aspect_before=aspect_before,
        aspect_after=aspect_after,
        improved=improved,
    )


# =============================================================================
# 规则评测
# =============================================================================
def run_rule_eval(report: ResearchReport | None, question_id: str, bench: ResearchBench) -> dict[str, Any]:
    if report is None:
        return {"error": "no report"}
    num_sources = len(getattr(report, "sources", []))
    return bench.evaluate_report(report.content, question_id, num_sources=num_sources)


# =============================================================================
# 解析 JSON 记录汇总
# =============================================================================
def summarize_json_records(records: list[JsonParseRecord]) -> dict[str, Any]:
    if not records:
        return {}

    by_stage: dict[str, list[JsonParseRecord]] = {}
    for r in records:
        by_stage.setdefault(r.stage, []).append(r)

    summary = {"total_records": len(records), "by_stage": {}}
    for stage, rs in by_stage.items():
        baseline_ok = sum(1 for r in rs if r.baseline_success)
        multi_ok = sum(1 for r in rs if r.multi_layer_success)
        # 只统计"被多层解析挽救"的样本的深度分布
        saved_depth_counts: dict[str, int] = {}
        for r in rs:
            if r.multi_layer_success and not r.baseline_success and r.depth_name not in ("direct", "failed"):
                saved_depth_counts[r.depth_name] = saved_depth_counts.get(r.depth_name, 0) + 1
        summary["by_stage"][stage] = {
            "count": len(rs),
            "baseline_success_rate": round(baseline_ok / len(rs), 4),
            "multi_layer_success_rate": round(multi_ok / len(rs), 4),
            "saved_by_multi_layer": multi_ok - baseline_ok,
            "saved_depth_distribution": saved_depth_counts,
        }
    return summary


# =============================================================================
# 保存结果
# =============================================================================
def save_results(
    ed: EvalData,
    ragas: RagasScores | None,
    rule_before: dict[str, Any],
    rule_after: dict[str, Any],
    json_summary: dict[str, Any],
    out_dir: Path,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sq = "".join(c if c.isalnum() or c in "_-" else "_" for c in ed.query[:30])
    path = out_dir / f"eval_json_ablation_{sq}_{ts}.json"

    report_before_text = ed.report_before.content if ed.report_before else ""
    report_after_text = ed.report_after.content if ed.report_after else ""

    payload = {
        "query": ed.query,
        "timestamp": datetime.now().isoformat(),
        "report_before": {
            "confidence": getattr(ed.report_before, "confidence", None),
            "length": len(report_before_text),
            "num_sources": len(getattr(ed.report_before, "sources", [])),
        },
        "report_after": {
            "confidence": getattr(ed.report_after, "confidence", None),
            "length": len(report_after_text),
            "num_sources": len(getattr(ed.report_after, "sources", [])),
            "adversarial_rounds": getattr(ed.report_after, "adversarial_rounds", 0),
            "final_score": getattr(ed.report_after, "final_score", None),
            "dimension_scores": {
                d.value: s for d, s in (getattr(ed.report_after, "dimension_scores", {}) or {}).items()
            },
            "adversarial_history": getattr(ed.report_after, "adversarial_history", []),
        },
        "json_parse_records": [
            {
                "stage": r.stage,
                "baseline_success": r.baseline_success,
                "baseline_error": r.baseline_error,
                "multi_layer_success": r.multi_layer_success,
                "multi_layer_error": r.multi_layer_error,
                "depth": r.depth,
                "depth_name": r.depth_name,
                "raw_len": r.raw_len,
                "raw_output": r.raw_output[:1000],  # 保存前1000字符用于复查
            }
            for r in ed.json_records
        ],
        "json_parse_summary": json_summary,
        "rule_eval": {
            "before": rule_before,
            "after": rule_after,
        },
    }

    if ragas:
        import math
        payload["ragas"] = {
            "faithfulness": {
                "before": ragas.faithfulness_before,
                "after": ragas.faithfulness_after,
                "delta": round(ragas.faithfulness_after - ragas.faithfulness_before, 4),
            },
            "answer_relevancy": {
                "before": ragas.relevancy_before,
                "after": ragas.relevancy_after,
                "delta": round(ragas.relevancy_after - ragas.relevancy_before, 4),
            },
            "aspect_critique": {
                "before": ragas.aspect_before,
                "after": ragas.aspect_after,
            },
            "improved": ragas.improved,
        }

    # 同时保存 Markdown 报告便于人工查看
    md_path = out_dir / f"eval_json_ablation_{sq}_{ts}.md"
    md_lines = [
        f"# 实验报告: {ed.query}",
        "",
        "## 对抗前报告",
        report_before_text,
        "",
        "---",
        "",
        "## 对抗后报告",
        report_after_text,
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# =============================================================================
# Per-question runner
# =============================================================================
async def run_single_question(
    query: str,
    question_id: str,
    bench: ResearchBench,
    config: dict,
    sm: SessionMemory,
    kb: KnowledgeBase,
    out_dir: Path,
    skip_ragas: bool,
) -> dict[str, Any]:
    """执行单个题目，返回结果摘要。"""

    _p()
    _p("=" * 60)
    _p(f" 题目: {question_id}")
    _p(f" 问题: {query[:80]}{'...' if len(query) > 80 else ''}")
    _p(f" RAGAS: {'跳过' if skip_ragas else '启用'}")
    _p()

    ed = EvalData(query=query)

    try:
        with JsonParseInterceptor() as interceptor:
            _p("[构建编排器 ...]")
            orch = build_orchestrator(config, sm, kb, ed)
            oc = config.get("orchestrator", {})
            timeout = oc.get("global_timeout_seconds", 300)
            _p(f"      OK (timeout={timeout}s)")

            _p("[执行研究流程 ...]")
            if config.get("adversarial", {}).get("enabled", True):
                _p("      Planner -> Search -> Synthesize -> Adversarial")
            else:
                _p("      Planner -> Search -> Synthesize -> Done")
            _p()

            ctx = ResearchContext(
                topic=query,
                max_iterations=oc.get("max_replan_rounds", 3),
                enable_adversarial=config.get("adversarial", {}).get("enabled", True),
            )

            t0 = time.time()
            state = await orch.run(ctx, timeout_seconds=timeout)
            elapsed = time.time() - t0

            ed.json_records = interceptor.records

        _p(f"\n研究完成 ({elapsed:.0f}s, 状态={state.value})")

        json_summary = summarize_json_records(ed.json_records)
        _p()
        _p("-" * 40)
        _p("JSON 解析消融结果:")
        _p(f"  总解析次数: {json_summary.get('total_records', 0)}")
        for stage, s in json_summary.get("by_stage", {}).items():
            _p(f"  [{stage}] n={s['count']}  基线成功率={s['baseline_success_rate']:.2%}  "
               f"多层成功率={s['multi_layer_success_rate']:.2%}  挽救次数={s['saved_by_multi_layer']}")
            if s['saved_depth_distribution']:
                _p(f"           挽救深度分布: {s['saved_depth_distribution']}")
        _p("-" * 40)

        if ed.report_before is None:
            _p("FAIL: 未生成对抗前报告")
            return {
                "question_id": question_id,
                "status": "failed",
                "error": "no report_before",
                "json_parse_summary": json_summary,
            }

        _p()
        _p("-" * 40)
        _p(f" 对抗前 | conf={getattr(ed.report_before, 'confidence', 0):.2f} | "
           f"len={len(ed.report_before.content)} | ctx={len(ed.contexts)}条")
        if ed.report_after is None:
            _p(" 对抗后 | (未执行)")
        else:
            _p(f" 对抗后 | conf={getattr(ed.report_after, 'confidence', 0):.2f} | "
               f"len={len(ed.report_after.content)} | "
               f"rounds={getattr(ed.report_after, 'adversarial_rounds', 0)} | "
               f"score={getattr(ed.report_after, 'final_score', 0):.1f}")
            _p(f"         五维: {getattr(ed.report_after, 'dimension_scores', {})}")
        _p("-" * 40)

        # 无对抗模式下，将 report_after 回退为 report_before，便于统一输出
        if ed.report_after is None and not config.get("adversarial", {}).get("enabled", True):
            ed.report_after = copy.deepcopy(ed.report_before)
            _p("  (无对抗模式，对抗后报告 = 初始报告)")

        _p()
        _p("规则评测中 ...")
        rule_before = run_rule_eval(ed.report_before, question_id, bench)
        rule_after = run_rule_eval(ed.report_after, question_id, bench)
        _p(f"  综合分: {rule_before.get('composite_score', 0):.3f} -> {rule_after.get('composite_score', 0):.3f}")
        _p(f"  factual: {rule_before.get('metrics', {}).get('factual_accuracy', 0):.3f} -> {rule_after.get('metrics', {}).get('factual_accuracy', 0):.3f}")
        _p(f"  logic:   {rule_before.get('metrics', {}).get('logical_consistency', 0):.3f} -> {rule_after.get('metrics', {}).get('logical_consistency', 0):.3f}")
        _p(f"  source:  {rule_before.get('metrics', {}).get('source_adequacy', 0):.3f} -> {rule_after.get('metrics', {}).get('source_adequacy', 0):.3f}")
        _p(f"  comp:    {rule_before.get('metrics', {}).get('comprehensiveness', 0):.3f} -> {rule_after.get('metrics', {}).get('comprehensiveness', 0):.3f}")

        ragas: RagasScores | None = None
        if not skip_ragas and ed.report_after is not None:
            _p()
            _p("RAGAS 按章节评估中 ...")
            ragas = await run_ragas_by_sections(ed, config)
            if ragas:
                import math
                _p()
                _p("  ═══════ RAGAS 评估 ═══════")
                _p(f"  Faithfulness:      {ragas.faithfulness_before:.4f} -> {ragas.faithfulness_after:.4f}")
                _p(f"  AnswerRelevancy:   {ragas.relevancy_before:.4f} -> {ragas.relevancy_after:.4f}")
                _p("  AspectCritique (按维度):")
                for dim in ADVERSARIAL_ASPECTS:
                    ab = ragas.aspect_before.get(dim, float("nan"))
                    aa = ragas.aspect_after.get(dim, float("nan"))
                    if math.isnan(ab) and math.isnan(aa):
                        continue
                    arrow = "--"
                    if not math.isnan(ab) and not math.isnan(aa):
                        arrow = "++" if aa > ab else "==" if aa == ab else "--"
                    _p(f"    {dim:<14}: {ab:.4f} -> {aa:.4f}  ({arrow})")
                _p()
                _p(f"  结论: {'✅ 对抗改善' if ragas.improved else '⚠️ 本次未显著改善'}")

        saved_path = save_results(ed, ragas, rule_before, rule_after, json_summary, out_dir)
        _p(f"\n结果已保存: {saved_path}")

        return {
            "question_id": question_id,
            "status": state.value,
            "elapsed_seconds": elapsed,
            "json_parse_summary": json_summary,
            "rule_before": rule_before,
            "rule_after": rule_after,
            "report_before": {
                "confidence": getattr(ed.report_before, "confidence", None),
                "length": len(ed.report_before.content),
                "num_sources": len(getattr(ed.report_before, "sources", [])),
            },
            "report_after": {
                "confidence": getattr(ed.report_after, "confidence", None) if ed.report_after else None,
                "length": len(ed.report_after.content) if ed.report_after else 0,
                "num_sources": len(getattr(ed.report_after, "sources", [])) if ed.report_after else 0,
                "adversarial_rounds": getattr(ed.report_after, "adversarial_rounds", 0) if ed.report_after else 0,
                "final_score": getattr(ed.report_after, "final_score", None) if ed.report_after else None,
                "dimension_scores": {
                    d.value: s for d, s in (getattr(ed.report_after, "dimension_scores", {}) or {}).items()
                } if ed.report_after else {},
            },
            "output_file": str(saved_path),
        }

    except Exception as e:
        logging.exception("题目 %s 异常", question_id)
        return {
            "question_id": question_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "json_parse_summary": summarize_json_records(ed.json_records) if ed.json_records else {},
        }


# =============================================================================
# main
# =============================================================================
async def main() -> int:
    parser = argparse.ArgumentParser(description="JSON 解析消融 + 对抗效果端到端评估")
    parser.add_argument("-q", "--query", help="研究问题（与 --bench-id / --run-all 二选一）")
    parser.add_argument("--bench-id", help="ResearchBench 题目 ID，如 tech_001")
    parser.add_argument("--run-all", action="store_true", help="跑 ResearchBench 全部题目")
    parser.add_argument("-c", "--config", default="config/default.yaml")
    parser.add_argument("-o", "--output", default="outputs")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-ragas", action="store_true", help="跳过 RAGAS 评估")
    parser.add_argument("--no-adversarial", action="store_true", help="关闭红蓝对抗循环，只生成初始报告")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for n in ("openai", "httpx", "httpcore", "primp", "urllib3", "aiosqlite"):
        logging.getLogger(n).setLevel(logging.WARNING)

    bench = ResearchBench()
    questions: list[tuple[str, str]] = []
    if args.run_all:
        questions = [(q["id"], q["query"]) for q in bench.questions]
    elif args.bench_id:
        q_obj = next((q for q in bench.questions if q["id"] == args.bench_id), None)
        if q_obj is None:
            _p(f"错误: ResearchBench 中未找到 {args.bench_id}")
            return 1
        questions = [(q_obj["id"], q_obj["query"])]
    elif args.query:
        questions = [("custom", args.query)]
    else:
        _p("错误: 请提供 -q/--query、--bench-id 或 --run-all")
        return 1

    _p()
    _p("=" * 60)
    _p(" JSON 解析消融 + 对抗效果端到端评估")
    _p("=" * 60)
    _p(f" 题目数: {len(questions)}")
    _p(f" 对抗: {'开启' if not args.no_adversarial else '关闭'}")
    _p(f" RAGAS: {'跳过' if args.skip_ragas else '启用'}")
    _p()

    # 加载配置
    cp = Path(args.config)
    config = yaml.safe_load(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    if "adversarial" not in config:
        config["adversarial"] = {}
    config["adversarial"]["enabled"] = not args.no_adversarial
    if not args.no_adversarial:
        config["adversarial"]["entry_confidence_threshold"] = 0.95
    _p(f"[1/3] 配置已加载，对抗: {'开启' if not args.no_adversarial else '关闭'}")

    # 初始化记忆库
    mc = config.get("memory", {})
    sm = SessionMemory(db_path=mc.get("session_db_path", "data/session_memory.db"))
    kb = KnowledgeBase(
        db_path=mc.get("knowledge_db_path", "data/knowledge_base.db"),
        embedder=MemoryEmbedder(),
        config=mc,
    )
    await sm.initialize()
    await kb.initialize()
    _p("[2/3] 记忆库已初始化")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total_start = time.time()

    try:
        for i, (qid, query) in enumerate(questions, 1):
            _p()
            _p("#" * 60)
            _p(f"# [{i}/{len(questions)}] {qid}")
            _p("#" * 60)
            result = await run_single_question(
                query=query,
                question_id=qid,
                bench=bench,
                config=config,
                sm=sm,
                kb=kb,
                out_dir=out_dir,
                skip_ragas=args.skip_ragas,
            )
            results.append(result)

    finally:
        await sm.close()
        await kb.close()

    # 汇总
    total_elapsed = time.time() - total_start
    _p()
    _p("=" * 60)
    _p(" 全部完成")
    _p("=" * 60)
    _p(f" 总耗时: {total_elapsed:.0f}s")
    _p(f" 成功: {sum(1 for r in results if r.get('status') == 'done')} / {len(results)}")

    # 聚合 JSON 解析统计
    all_records: list[JsonParseRecord] = []
    for r in results:
        jps = r.get("json_parse_summary", {})
        for stage, s in jps.get("by_stage", {}).items():
            # 这里只聚合计数，不保留原始 raw_output
            pass

    # 聚合规则评测
    def avg_delta(key: str) -> tuple[float, float, int]:
        before_vals = []
        after_vals = []
        for r in results:
            rb = r.get("rule_before", {}).get("metrics", {})
            ra = r.get("rule_after", {}).get("metrics", {})
            if key in rb and key in ra:
                before_vals.append(rb[key])
                after_vals.append(ra[key])
        if not before_vals:
            return 0.0, 0.0, 0
        return sum(before_vals) / len(before_vals), sum(after_vals) / len(after_vals), len(before_vals)

    _p()
    _p("规则评测平均变化:")
    for key in ("factual_accuracy", "logical_consistency", "source_adequacy", "comprehensiveness"):
        b, a, n = avg_delta(key)
        _p(f"  {key:<20}: {b:.3f} -> {a:.3f} (n={n})")
    comp_before = [r.get("rule_before", {}).get("composite_score", 0) for r in results]
    comp_after = [r.get("rule_after", {}).get("composite_score", 0) for r in results]
    if comp_before and comp_after:
        _p(f"  {'composite_score':<20}: {sum(comp_before)/len(comp_before):.3f} -> {sum(comp_after)/len(comp_after):.3f} (n={len(comp_before)})")

    # 保存汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "success_count": sum(1 for r in results if r.get("status") == "done"),
        "failed_count": sum(1 for r in results if r.get("status") != "done"),
        "total_elapsed_seconds": total_elapsed,
        "skip_ragas": args.skip_ragas,
        "results": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"eval_json_ablation_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"\n汇总结果: {summary_path}")

    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
