#!/usr/bin/env python3
# ruff: noqa: E402
"""
eval_ragas.py
================================================================================
RAGAS 离线评测模块（按章节评估对抗前/后报告）。

RAGAS 评测已从端到端脚本 eval_json_parser_ablation.py 抽离：端到端流程只做
规则评测 + Red 评分（外加耗时/token/成功阶段统计），本模块供后续离线补跑
RAGAS（Faithfulness / AnswerRelevancy / AspectCritique）使用。

本模块为纯模块（无 main/CLI）。使用方式示例：

    from eval_ragas import RagasEvalData, run_ragas_by_sections

    ed = RagasEvalData(query=..., report_before=..., report_after=...,
                       context_items=..., search_items=..., adv_search_start=...)
    scores = await run_ragas_by_sections(ed, config)
================================================================================
"""

from __future__ import annotations

import math
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from deep_research.core.schema import ResearchReport
from deep_research.models import ModelRouter
from deep_research.tools import WebSearchTool


# =============================================================================
# 兼容 ragas 0.4 依赖的 langchain_community.chat_models.vertexai 缺失
# （langchain-community 0.4 已移除该子模块；缺失时用 dummy 顶替，
#  否则 ragas.llms 导入即失败，RAGAS 永远静默不可用）
# =============================================================================
def _patch_vertexai() -> None:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    import importlib

    for name in ("langchain_community", "langchain_community.chat_models"):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            mod = types.ModuleType(name)
            mod.__path__ = []  # 伪包，允许挂载子模块
            sys.modules[name] = mod
            parent, _, attr = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], attr, mod)
    dummy = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy
    sys.modules["langchain_community.chat_models"].vertexai = dummy


# =============================================================================
# 数据容器
# =============================================================================
@dataclass
class RagasEvalData:
    """run_ragas_by_sections 的输入数据（字段名与原端到端脚本的 EvalData 对齐）。"""
    query: str = ""
    report_before: ResearchReport | None = None
    report_after: ResearchReport | None = None
    # 研究阶段从 trajectory 提取的检索材料（带 URL，用于与报告引用来源匹配）
    context_items: list[dict[str, str]] = field(default_factory=list)
    # CapturingWebSearchTool 全程捕获的 web_search 结果（带 URL）
    search_items: list[dict[str, str]] = field(default_factory=list)
    # 进入对抗阶段时 search_items 的长度，此后的条目即对抗阶段新增搜索
    adv_search_start: int = 0


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


# =============================================================================
# 辅助函数
# =============================================================================
def _p(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Windows GBK 控制台无法编码 emoji 等字符时降级为可打印形式
        print(msg.encode("gbk", "replace").decode("gbk"), flush=True)


# =============================================================================
# 搜索结果拦截器（捕获 web_search 结果，作为 RAGAS retrieved_contexts）
# =============================================================================
class CapturingWebSearchTool:
    name = "web_search"

    def __init__(self, delegate, capture_list: list[dict]) -> None:
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
                    self._capture.append({
                        "url": str(r.get("url", "")).strip(),
                        "text": snippet[:2000],
                    })
        return results


# =============================================================================
# 检索上下文提取与排序
# =============================================================================
def _extract_context_items(results: list) -> list[dict[str, str]]:
    """从 agent trajectory 提取检索上下文，保留 URL 以便与报告引用来源匹配。

    同一 URL 去重并保留最长文本（browser 全文优先于 search snippet）；
    无 URL 的文本按前缀去重。
    """
    by_url: dict[str, str] = {}
    anon: list[str] = []
    seen_anon: set[str] = set()

    def _add(url: str, text: str) -> None:
        if not isinstance(text, str) or len(text) < 20:
            return
        text = text[:2000]
        url = (url or "").strip()
        if url:
            if len(text) > len(by_url.get(url, "")):
                by_url[url] = text
        else:
            key = text[:100]
            if key not in seen_anon:
                seen_anon.add(key)
                anon.append(text)

    for r in results:
        for step in (r.trajectory or []):
            if step.get("role") != "tool":
                continue
            res = step.get("result")
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        _add(item.get("url", ""), item.get("snippet", "") or item.get("summary", ""))
            elif isinstance(res, dict):
                _add(res.get("url", ""), res.get("content", ""))
                for p in res.get("papers", []):
                    if isinstance(p, dict):
                        _add(p.get("pdf_url", "") or p.get("url", ""), p.get("summary", ""))
    return [{"url": u, "text": t} for u, t in by_url.items()] + [
        {"url": "", "text": t} for t in anon
    ]


_CITATION_REF = re.compile(r"\[(\d+)\]")


def _rank_contexts(
    items: list[dict[str, str]],
    report: ResearchReport | None,
) -> list[str]:
    """按报告引用频次排序上下文：被引用次数越多的来源排越前，未被引用的垫底。

    排序信号来自报告正文中的 [N] 引用标记（sources 的 citation_id → URL 映射）。
    输入列表先按 URL 去重（保留最长文本），无 URL 的文本按前缀去重后置底。
    """
    merged: dict[str, str] = {}
    anon: list[str] = []
    seen_anon: set[str] = set()
    for it in items:
        url, text = it.get("url", ""), it.get("text", "")
        if url:
            if len(text) > len(merged.get(url, "")):
                merged[url] = text
        else:
            key = text[:100]
            if key not in seen_anon:
                seen_anon.add(key)
                anon.append(text)

    counts: dict[str, int] = {}
    if report is not None:
        content = getattr(report, "content", "") or ""
        ref_counts: dict[int, int] = {}
        for m in _CITATION_REF.findall(content):
            cid = int(m)
            ref_counts[cid] = ref_counts.get(cid, 0) + 1
        for src in getattr(report, "sources", []) or []:
            cid = src.get("citation_id")
            url = str(src.get("url", "")).strip()
            if cid and url and ref_counts.get(int(cid)):
                counts[url] = counts.get(url, 0) + ref_counts[int(cid)]

    ranked = sorted(
        merged.items(),
        key=lambda kv: (0 if counts.get(kv[0]) else 1, -counts.get(kv[0], 0), -len(kv[1])),
    )
    return [t for _, t in ranked] + anon


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
async def run_ragas_by_sections(ed: RagasEvalData, config: dict) -> RagasScores | None:
    """用 RAGAS 按章节评估对抗前/后报告。"""
    _patch_vertexai()
    try:
        from openai import OpenAI
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas.evaluation import aevaluate
        from ragas.llms import llm_factory
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._aspect_critic import AspectCritic
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
        metrics.append(AspectCritic(name=dim_key, definition=definition, llm=ragas_llm))

    # 准备上下文：按报告引用频次排序，被引用越多的来源越优先进入核对池
    ctx_before = _rank_contexts(ed.context_items, ed.report_before) or ["(no contexts)"]
    adv_items = ed.search_items[ed.adv_search_start:]
    ctx_after = _rank_contexts(ed.context_items + adv_items, ed.report_after) or ctx_before
    if adv_items:
        _p(f"  📎 对抗阶段捕获 {len(adv_items)} 条新搜索结果，"
           f"合并去重后 contexts: {len(ctx_before)} → {len(ctx_after)} 条")

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
                retrieved_contexts=contexts[:20],  # 已按引用频次排序，取头部 20 条
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
