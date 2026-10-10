#!/usr/bin/env python3
# ruff: noqa: E402
"""
eval_json_parser_ablation.py
================================================================================
端到端 JSON 多层容错解析消融 + 红蓝对抗效果评估（规则评测 + Red 评分）。

运行一次完整研究流程（JSON 容错 + 对抗全部开启），在 4 个 JSON 解析节点
拦截原始 LLM 输出，同时记录原生 json.loads 基线与项目多层容错解析的结果。
同时保存 Summarizer 后的对抗前报告与红蓝对抗后的最终报告，做规则评测，
并统计分段耗时（报告生成 vs 对抗）、各模块真实 token 用量与成功阶段标记。

RAGAS 评测已从本脚本抽离至 eval_ragas.py，供后续离线补跑使用。

用法:
    python eval_json_parser_ablation.py -q "你的研究问题"
    python eval_json_parser_ablation.py --bench-id tech_001
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import random
import re
import sys
import time
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
from deep_research.observability import RunEventRecorder
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
# JSON 解析拦截器：直接解析 -> 自研启发式修复 -> json_repair
# =============================================================================
@dataclass
class JsonParseRecord:
    stage: str
    raw_output: str
    baseline_success: bool = False
    baseline_error: str = ""
    heuristic_success: bool = False
    json_repair_success: bool = False
    production_success: bool = False
    production_error: str = ""
    repaired_by: str = "failed"
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
                record = self._evaluate_layers(raw, stage)
                self.records.append(record)

                try:
                    result = original(*args, **kwargs)
                    record.production_success = self._is_success(result, stage)
                    record.production_error = ""
                    return result
                except Exception as e:
                    record.production_success = False
                    record.production_error = type(e).__name__
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
    # 基线 vs 自研启发式修复 vs json_repair
    # ------------------------------------------------------------------
    def _evaluate_layers(self, raw: str, stage: str) -> JsonParseRecord:
        raw = raw or ""
        record = JsonParseRecord(stage=stage, raw_output=raw[:8000], raw_len=len(raw))

        # 基线：直接 json.loads
        try:
            json.loads(raw)
            record.baseline_success = True
            record.repaired_by = "direct"
            return record
        except Exception as e:
            record.baseline_success = False
            record.baseline_error = type(e).__name__

        cleaned = self._heuristic_repair(raw)
        try:
            json.loads(cleaned)
            record.heuristic_success = True
            record.repaired_by = "heuristic"
            return record
        except Exception:
            pass

        try:
            repaired = json_repair_loads(cleaned)
            if isinstance(repaired, (dict, list)):
                record.json_repair_success = True
                record.repaired_by = "json_repair"
        except Exception:
            pass
        return record

    @staticmethod
    def _heuristic_repair(raw: str) -> str:
        """自研启发式修复：代码块/括号提取、全角归一化、注释和尾逗号清理。"""

        text = (raw or "").strip()
        fenced = re.search(
            r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if fenced:
            text = fenced.group(1).strip()

        for src, dst in (
            ("｛", "{"), ("｝", "}"), ("［", "["), ("］", "]"),
            ("：", ":"), ("，", ","), ("“", '"'), ("”", '"'),
            ("‘", "'"), ("’", "'"),
        ):
            text = text.replace(src, dst)

        starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
        if starts:
            start = min(starts)
            opener = text[start]
            closer = "}" if opener == "{" else "]"
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
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        text = text[start : i + 1]
                        break

        # 只删除字符串外的 // 注释，避免破坏 https:// URL。
        cleaned_lines = []
        for line in text.splitlines():
            in_str = False
            escape = False
            cut = len(line)
            for i, ch in enumerate(line):
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                elif (
                    ch == "/"
                    and not in_str
                    and i + 1 < len(line)
                    and line[i + 1] == "/"
                ):
                    cut = i
                    break
            cleaned_lines.append(line[:cut])
        text = "\n".join(cleaned_lines)
        return re.sub(r",(\s*[}\]])", r"\1", text).strip()

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
        if stage == "blue_agent_fix":
            if isinstance(result, dict):
                return bool(result)
            return False
        if stage == "issue_merger":
            return isinstance(result, list) and len(result) > 0
        return False


# =============================================================================
# 数据容器
# =============================================================================
@dataclass
class EvalData:
    query: str = ""
    report_before: ResearchReport | None = None
    report_after: ResearchReport | None = None
    sources: list[dict] = field(default_factory=list)
    json_records: list[JsonParseRecord] = field(default_factory=list)
    # SYNTHESIZING / ADVERSARIAL 完成时刻（time.time()），用于分段耗时统计
    t_synthesized: float | None = None
    t_adversarial_done: float | None = None
    # 按模块累计的 LLM 真实 API token 用量（由 _UsageTrackingClient 写入）
    token_by_module: dict = field(default_factory=dict)


# =============================================================================
# 辅助函数
# =============================================================================
def _p(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Windows GBK 控制台无法编码 emoji 等字符时降级为可打印形式
        print(msg.encode("gbk", "replace").decode("gbk"), flush=True)


def _create_tool_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(WebSearchTool())
    r.register(BrowserTool())
    r.register(ArxivReaderTool())
    r.register(FileReaderTool())
    r.register(CodeSandboxTool())
    r.register(CalculatorTool())
    r.register(NotepadTool())
    return r


# =============================================================================
# LLM 客户端包装：记录真实 API token 用量
# =============================================================================
class _UsageTrackingClient:
    """包装 LLMClient，委托所有属性，拦截同步 chat() 并累计 token 用量。"""

    def __init__(self, delegate, bucket: dict) -> None:
        self._delegate = delegate
        self._bucket = bucket

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @staticmethod
    def _usage_get(usage: Any, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return usage.get(key) or 0
        return getattr(usage, key, 0) or 0

    def chat(self, messages, tools=None, **kwargs):
        resp = self._delegate.chat(messages, tools=tools, **kwargs)
        usage = getattr(resp, "usage", None)
        prompt = self._usage_get(usage, "prompt_tokens")
        completion = self._usage_get(usage, "completion_tokens")
        total = self._usage_get(usage, "total_tokens")
        if not total:
            total = prompt + completion
        self._bucket["calls"] += 1
        self._bucket["prompt_tokens"] += prompt
        self._bucket["completion_tokens"] += completion
        self._bucket["total_tokens"] += total
        return resp


def _timing_payload(t0: float | None, ed: EvalData, total_seconds: float | None) -> dict[str, Any]:
    research = (
        ed.t_synthesized - t0
        if ed.t_synthesized is not None and t0 is not None
        else None
    )
    adversarial = (
        ed.t_adversarial_done - ed.t_synthesized
        if ed.t_adversarial_done is not None and ed.t_synthesized is not None
        else None
    )
    return {
        "research_seconds": research,
        "adversarial_seconds": adversarial,
        "total_seconds": total_seconds,
    }


def _token_usage_payload(ed: EvalData, estimated_task_tokens: int | None = None) -> dict[str, Any]:
    by_module = {k: dict(v) for k, v in ed.token_by_module.items()}
    return {
        "by_module": by_module,
        "total_tokens": sum(b.get("total_tokens", 0) for b in by_module.values()),
        "total_calls": sum(b.get("calls", 0) for b in by_module.values()),
        "estimated_task_tokens": estimated_task_tokens,
        "note": "by_module 为 API 真实用量；estimated_task_tokens 为 agent 记录的字符估算值，供对照",
    }


# =============================================================================
# Orchestrator 构建（注入保存 hook + token 用量统计）
# =============================================================================
def build_orchestrator(
    config: dict,
    session_memory: SessionMemory,
    knowledge_base: KnowledgeBase,
    ed: EvalData,
    event_sink=None,
    token_buckets: dict | None = None,
) -> Orchestrator:
    model_cfg = config.get("model", {})
    backend = model_cfg.get("backend", "deepseek")

    client_kwargs: dict = {}
    for k in ("base_model", "base_url", "api_key", "temperature", "top_p", "max_tokens"):
        if k in model_cfg:
            client_kwargs[k] = model_cfg[k]

    llm_client = ModelRouter.create_backend(backend, **client_kwargs)
    log.info("LLM: %s", backend)

    # 仅用透明代理记录 Planner 用量，不改变其后端、采样参数或调用路径。
    planner_client = llm_client
    if token_buckets is not None:
        planner_bucket = token_buckets.setdefault("planner", {
            "calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0,
        })
        planner_client = _UsageTrackingClient(llm_client, planner_bucket)
    planner = Planner(planner_client)
    registry = _create_tool_registry()

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
        client = ModelRouter.create_backend(mb, **m)
        if token_buckets is not None:
            bucket = token_buckets.setdefault(mod, {
                "calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
            })
            return _UsageTrackingClient(client, bucket)
        return client

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
        event_sink=event_sink,
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
        event_sink=event_sink,
    )

    # ---- Hook: SYNTHESIZING 完成后保存对抗前报告 ----
    orig_synth = orch._state_handlers[WorkflowState.SYNTHESIZING]
    async def synth_hook():
        r = await orig_synth()
        ed.t_synthesized = time.time()
        if orch._report is not None:
            ed.report_before = copy.deepcopy(orch._report)
            ed.sources = getattr(orch._report, "sources", [])
            ed.query = orch._query
            _p(f"  [hook] 对抗前报告已保存 (conf={getattr(orch._report, 'confidence', 0):.2f})")
        return r

    # ---- Hook: ADVERSARIAL 完成后保存对抗后报告 ----
    orig_adv = orch._state_handlers[WorkflowState.ADVERSARIAL]
    async def adv_hook():
        r = await orig_adv()
        ed.t_adversarial_done = time.time()
        if orch._report is not None:
            ed.report_after = copy.deepcopy(orch._report)
            _p(f"  [hook] 对抗后报告已保存 (conf={getattr(orch._report, 'confidence', 0):.2f}, "
               f"rounds={getattr(orch._report, 'adversarial_rounds', 0)})")
        return r

    orch._state_handlers[WorkflowState.SYNTHESIZING] = synth_hook
    orch._state_handlers[WorkflowState.ADVERSARIAL] = adv_hook

    return orch


# =============================================================================
# 规则评测
# =============================================================================
def run_rule_eval(report: ResearchReport | None, question_id: str, bench: ResearchBench) -> dict[str, Any]:
    if report is None:
        return {"error": "no report"}
    sources = list(getattr(report, "sources", []) or [])
    return bench.evaluate_report(report.content, question_id, sources=sources)


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
        heuristic_saved = sum(1 for r in rs if r.repaired_by == "heuristic")
        repair_saved = sum(1 for r in rs if r.repaired_by == "json_repair")
        heuristic_ok = baseline_ok + heuristic_saved
        final_ok = heuristic_ok + repair_saved
        summary["by_stage"][stage] = {
            "count": len(rs),
            "baseline_success_rate": round(baseline_ok / len(rs), 4),
            "heuristic_success_rate": round(heuristic_ok / len(rs), 4),
            "final_success_rate": round(final_ok / len(rs), 4),
            "saved_by_heuristic": heuristic_saved,
            "saved_by_json_repair": repair_saved,
            "failed": len(rs) - final_ok,
        }
    return summary


# =============================================================================
# 保存结果
# =============================================================================
def save_results(
    ed: EvalData,
    question_id: str,
    bench: ResearchBench,
    status: str,
    rule_before: dict[str, Any] | None,
    rule_after: dict[str, Any] | None,
    json_summary: dict[str, Any],
    out_dir: Path,
    run_id: str = "",
    stages: dict[str, bool] | None = None,
    timing: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
    error: str | None = None,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sq = "".join(c if c.isalnum() or c in "_-" else "_" for c in ed.query[:30])
    path = out_dir / f"eval_json_ablation_{sq}_{ts}.json"

    q_obj = next((q for q in bench.questions if q["id"] == question_id), None)
    domain = q_obj.get("domain", "") if q_obj else ""

    report_before_text = ed.report_before.content if ed.report_before else ""
    report_after_text = ed.report_after.content if ed.report_after else ""
    # 初始报告的 Red 评分 = 对抗第 1 轮评分（当时 Blue 尚未做任何修复）
    adv_history = getattr(ed.report_after, "adversarial_history", []) if ed.report_after else []
    first_round_red = adv_history[0] if adv_history else {}

    if stages is None:
        stages = {
            "report_generated": ed.report_before is not None,
            "adversarial_completed": ed.report_after is not None,
            "rule_eval_done": rule_after is not None,
        }

    payload: dict[str, Any] = {
        "question_id": question_id,
        "query": ed.query,
        "domain": domain,
        "status": status,
        "run_id": run_id,
        "saved_at": datetime.now().isoformat(),
        "stages": stages,
        "timing": timing or {},
        "token_usage": token_usage or {},
        "report_before": {
            "confidence": getattr(ed.report_before, "confidence", None),
            "length": len(report_before_text),
            "num_sources": len(getattr(ed.report_before, "sources", []) or []),
            "content": report_before_text,
            "sources": getattr(ed.report_before, "sources", []) or [],
            "red_overall_score": first_round_red.get("red_overall_score"),
            "red_dimension_scores": first_round_red.get("red_dimension_scores", {}),
        },
        "report_after": {
            "confidence": getattr(ed.report_after, "confidence", None),
            "length": len(report_after_text),
            "num_sources": len(getattr(ed.report_after, "sources", []) or []),
            "content": report_after_text,
            "sources": getattr(ed.report_after, "sources", []) or [],
            "adversarial_rounds": getattr(ed.report_after, "adversarial_rounds", 0),
            "final_score": getattr(ed.report_after, "final_score", None),
            "dimension_scores": {
                d.value: s for d, s in (getattr(ed.report_after, "dimension_scores", {}) or {}).items()
            },
        },
        "json_parse_records": [
            {
                "stage": r.stage,
                "baseline_success": r.baseline_success,
                "baseline_error": r.baseline_error,
                "heuristic_success": r.heuristic_success,
                "json_repair_success": r.json_repair_success,
                "repaired_by": r.repaired_by,
                "production_success": r.production_success,
                "production_error": r.production_error,
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
        "adversarial_history": adv_history,
    }
    if error:
        payload["error"] = error

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
) -> dict[str, Any]:
    """执行单个题目，返回结果摘要。"""

    _p()
    _p("=" * 60)
    _p(f" 题目: {question_id}")
    _p(f" 问题: {query[:80]}{'...' if len(query) > 80 else ''}")
    _p()

    ed = EvalData(query=query)

    recorder = RunEventRecorder(query)
    t0: float | None = None
    interceptor = JsonParseInterceptor()
    try:
        try:
            with interceptor:
                _p("[构建编排器 ...]")
                orch = build_orchestrator(
                    config, sm, kb, ed, event_sink=recorder,
                    token_buckets=ed.token_by_module,
                )
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

                _report = getattr(orch, "_report", None)
                recorder.emit("run_completed", {
                    "state": state.value,
                    "confidence": getattr(_report, "confidence", 0.0) if _report else 0.0,
                    "sources": len(getattr(_report, "sources", []) or []) if _report else 0,
                })
        finally:
            # 即使 orch.run() 因解析错误或其他异常中止，也保留已经采集的记录。
            ed.json_records = list(interceptor.records)

        _p(f"\n研究完成 ({elapsed:.0f}s, 状态={state.value})")

        timing = _timing_payload(t0, ed, elapsed)
        estimated_task_tokens = sum(
            getattr(r, "token_usage", 0) or 0 for r in getattr(orch, "_results", []) or []
        )
        token_usage = _token_usage_payload(ed, estimated_task_tokens)
        rs = f"{timing['research_seconds']:.0f}s" if timing["research_seconds"] is not None else "n/a"
        advs = f"{timing['adversarial_seconds']:.0f}s" if timing["adversarial_seconds"] is not None else "n/a"
        _p(f"  耗时: 报告生成={rs} 对抗={advs} | "
           f"token(真实)={token_usage['total_tokens']} 估算对照={estimated_task_tokens}")

        json_summary = summarize_json_records(ed.json_records)
        _p()
        _p("-" * 40)
        _p("JSON 解析消融结果:")
        _p(f"  总解析次数: {json_summary.get('total_records', 0)}")
        for stage, s in json_summary.get("by_stage", {}).items():
            _p(
                f"  [{stage}] n={s['count']}  基线={s['baseline_success_rate']:.2%}  "
                f"启发式后={s['heuristic_success_rate']:.2%}  "
                f"json_repair后={s['final_success_rate']:.2%}"
            )
            _p(
                f"           自研挽救={s['saved_by_heuristic']}  "
                f"json_repair挽救={s['saved_by_json_repair']}  失败={s['failed']}"
            )
        _p("-" * 40)

        if ed.report_before is None:
            _p("FAIL: 未生成对抗前报告")
            failed_path = save_results(
                ed, question_id, bench,
                status="failed",
                rule_before=None, rule_after=None,
                json_summary=json_summary,
                out_dir=out_dir,
                run_id=recorder.run_id,
                stages={
                    "report_generated": False,
                    "adversarial_completed": False,
                    "rule_eval_done": False,
                },
                timing=timing, token_usage=token_usage,
                error="no report_before",
            )
            return {
                "question_id": question_id,
                "status": "failed",
                "error": "no report_before",
                "run_id": recorder.run_id,
                "json_parse_summary": json_summary,
                "stages": {
                    "report_generated": False,
                    "adversarial_completed": False,
                    "rule_eval_done": False,
                },
                "timing": timing,
                "output_file": str(failed_path),
            }

        # 在“无对抗模式回退”之前记录真实阶段完成情况（report_before 此处必存在）
        stages = {
            "report_generated": True,
            "adversarial_completed": ed.report_after is not None,
            "rule_eval_done": True,
        }

        _p()
        _p("-" * 40)
        _p(f" 对抗前 | conf={getattr(ed.report_before, 'confidence', 0):.2f} | "
           f"len={len(ed.report_before.content)}")
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

        saved_path = save_results(
            ed, question_id, bench,
            status=state.value,
            rule_before=rule_before, rule_after=rule_after,
            json_summary=json_summary,
            out_dir=out_dir,
            run_id=recorder.run_id,
            stages=stages,
            timing=timing, token_usage=token_usage,
        )
        _p(f"\n结果已保存: {saved_path}")

        return {
            "question_id": question_id,
            "status": state.value,
            "elapsed_seconds": elapsed,
            "run_id": recorder.run_id,
            "json_parse_summary": json_summary,
            "rule_before": rule_before,
            "rule_after": rule_after,
            "stages": stages,
            "timing": timing,
            "token_usage": token_usage,
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
        err = f"{type(e).__name__}: {e}"
        recorder.emit("run_failed", {
            "state": "failed",
            "error": err,
        })
        timing = _timing_payload(t0, ed, (time.time() - t0) if t0 is not None else None)
        stages = {
            "report_generated": ed.report_before is not None,
            "adversarial_completed": ed.report_after is not None,
            "rule_eval_done": False,
        }
        rule_before = run_rule_eval(ed.report_before, question_id, bench) if ed.report_before else None
        rule_after = run_rule_eval(ed.report_after, question_id, bench) if ed.report_after else None
        token_usage = _token_usage_payload(ed)
        output_file = None
        try:
            output_file = str(save_results(
                ed, question_id, bench,
                status="failed",
                rule_before=rule_before, rule_after=rule_after,
                json_summary=summarize_json_records(ed.json_records),
                out_dir=out_dir,
                run_id=recorder.run_id,
                stages=stages,
                timing=timing, token_usage=token_usage,
                error=err,
            ))
        except Exception:
            logging.exception("题目 %s 的失败结果保存异常", question_id)
        return {
            "question_id": question_id,
            "status": "failed",
            "error": err,
            "run_id": recorder.run_id,
            "json_parse_summary": summarize_json_records(ed.json_records) if ed.json_records else {},
            "stages": stages,
            "timing": timing,
            "token_usage": token_usage,
            "rule_before": rule_before,
            "rule_after": rule_after,
            "output_file": output_file,
        }


# =============================================================================
# main
# =============================================================================
async def main() -> int:
    parser = argparse.ArgumentParser(description="JSON 解析消融 + 对抗效果端到端评估")
    parser.add_argument("-q", "--query", help="研究问题（与 --bench-id / --run-all 二选一）")
    parser.add_argument("--bench-id", help="ResearchBench 题目 ID，如 tech_001")
    parser.add_argument("--run-all", action="store_true", help="跑 ResearchBench 全部题目")
    parser.add_argument("--sample", type=int, help="从 ResearchBench 随机抽取指定数量题目")
    parser.add_argument("--seed", type=int, default=20261009, help="随机抽样种子")
    parser.add_argument("-c", "--config", default="config/experiment.yaml")
    parser.add_argument("-o", "--output", default="outputs")
    parser.add_argument("--log-level", default="INFO")
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
    selectors = sum(bool(x) for x in (args.run_all, args.bench_id, args.query, args.sample))
    if selectors != 1:
        _p("错误: -q/--query、--bench-id、--run-all、--sample 必须且只能提供一个")
        return 1

    if args.run_all:
        questions = [(q["id"], q["query"]) for q in bench.questions]
    elif args.sample:
        if args.sample < 1 or args.sample > len(bench.questions):
            _p(f"错误: --sample 必须在 1 到 {len(bench.questions)} 之间")
            return 1
        sampled = random.Random(args.seed).sample(bench.questions, args.sample)
        questions = [(q["id"], q["query"]) for q in sampled]
    elif args.bench_id:
        q_obj = next((q for q in bench.questions if q["id"] == args.bench_id), None)
        if q_obj is None:
            _p(f"错误: ResearchBench 中未找到 {args.bench_id}")
            return 1
        questions = [(q_obj["id"], q_obj["query"])]
    elif args.query:
        questions = [("custom", args.query)]

    _p()
    _p("=" * 60)
    _p(" JSON 解析消融 + 对抗效果端到端评估")
    _p("=" * 60)
    _p(f" 题目数: {len(questions)}")
    if args.sample:
        _p(f" 抽样种子: {args.seed}")
        _p(f" 抽中题目: {', '.join(qid for qid, _ in questions)}")
    _p(f" 对抗: {'开启' if not args.no_adversarial else '关闭'}")
    _p()

    # 加载配置
    cp = Path(args.config).resolve()
    config_bytes = cp.read_bytes() if cp.exists() else b""
    config = yaml.safe_load(config_bytes.decode("utf-8")) if config_bytes else {}
    config_hash = hashlib.sha256(config_bytes).hexdigest() if config_bytes else "missing"
    if "adversarial" not in config:
        config["adversarial"] = {}
    config["adversarial"]["enabled"] = not args.no_adversarial
    if not args.no_adversarial:
        config["adversarial"]["entry_confidence_threshold"] = 0.95
    _p(f"[1/3] 配置已加载: {cp}")
    _p(f"      SHA-256: {config_hash}")
    _p(f"      对抗: {'开启' if not args.no_adversarial else '关闭'}")

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
    _p(f" 报告生成成功率: {sum(1 for r in results if r.get('stages', {}).get('report_generated'))} / {len(results)}")
    _p(f" 对抗完成率: {sum(1 for r in results if r.get('stages', {}).get('adversarial_completed'))} / {len(results)}")

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
        "results": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"eval_json_ablation_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"\n汇总结果: {summary_path}")

    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
