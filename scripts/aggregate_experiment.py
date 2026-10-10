#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/aggregate_experiment.py
================================================================================
聚合端到端评测实验结果，产出 JSON + Markdown 报告。

数据源：
1. outputs/eval_json_ablation_summary_<ts>.json（汇总文件，--summary 指定或自动选最新）
2. 每题明细 JSON（results[*].output_file 指向，含 ragas 字段）
3. outputs/debug_runs/<run_id>/events.jsonl（事件流）

用法：
    python scripts/aggregate_experiment.py [--summary PATH] [-o outputs]
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deep_research.evaluation.metrics.stats import (  # noqa: E402
    bootstrap_ci_paired,
    cohens_d,
    paired_t_test,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (
        isinstance(x, float) and math.isnan(x)
    )


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # json.load 可解析非法 NaN token，后续用 _is_num 过滤


def _sanitize(obj):
    """递归将非有限浮点（inf/nan，如零方差配对 t 检验产生的 inf）转为 None。"""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _paired_stats(pairs: list[tuple[float, float]], invert: bool = False) -> dict:
    """对 (before, after) 配对列表做统计。n<2 时跳过 t 检验/效应量。

    invert=True 用于"越低越好"指标：显著性检验与效应量按改善方向
    （before - after）计算，展示的均值与 diff（after - before）保持原样。
    """
    pairs = [(b, a) for b, a in pairs if _is_num(b) and _is_num(a)]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "note": "无有效配对样本"}
    before = [b for b, _ in pairs]
    after = [a for _, a in pairs]
    diffs = [b - a if invert else a - b for b, a in pairs]
    result = {
        "n": n,
        "before_mean": round(mean(before), 4),
        "after_mean": round(mean(after), 4),
        "diff": round(mean(after) - mean(before), 4),
        "bootstrap": bootstrap_ci_paired(diffs),
    }
    if invert:
        result["direction"] = "lower_is_better，显著性检验按改善方向（before - after）计算"
    if n >= 2:
        result["cohens_d"] = round(
            cohens_d(before, after) if invert else cohens_d(after, before), 4
        )
        result["t_test"] = (
            paired_t_test(before, after) if invert else paired_t_test(after, before)
        )
    else:
        result["note"] = "n<2，跳过 Cohen's d 与配对 t 检验"
    return result


def _dist(values: list[float]) -> dict:
    vals = [v for v in values if _is_num(v)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def _sig_wording(stats: dict) -> str:
    boot = stats.get("bootstrap") or {}
    if boot.get("significant"):
        return f"显著提升（bootstrap p={boot.get('p_value')}）"
    p = boot.get("p_value", "—")
    return f"未检测到显著差异（bootstrap p={p}）"


# ---------------------------------------------------------------------------
# 1. 总览
# ---------------------------------------------------------------------------

def build_overview(summary: dict) -> dict:
    results = summary.get("results", [])
    status_counter = Counter(r.get("status", "unknown") for r in results)
    per_question = [
        {
            "question_id": r.get("question_id"),
            "status": r.get("status"),
            "run_id": r.get("run_id"),
            "elapsed_seconds": r.get("elapsed_seconds"),
        }
        for r in results
    ]
    num = summary.get("num_questions", len(results))
    success = summary.get("success_count", status_counter.get("done", 0))

    # 分阶段成功率（results[].stages，旧数据无此字段则跳过）
    staged = [r for r in results if isinstance(r.get("stages"), dict)]
    if staged:
        report_gen = [r for r in staged if r["stages"].get("report_generated")]
        adv_done = [r for r in report_gen if r["stages"].get("adversarial_completed")]
        stage_rates = {
            "n_with_stages": len(staged),
            "report_generated_count": len(report_gen),
            "report_generated_rate": round(len(report_gen) / len(staged), 4),
            "adversarial_completed_count": len(adv_done),
            "adversarial_completed_rate": (
                round(len(adv_done) / len(report_gen), 4) if report_gen else None
            ),
            "adversarial_rate_note": "对抗完成率的基数为 report_generated=true 的样本",
        }
    else:
        stage_rates = {
            "n_with_stages": 0,
            "note": "样本均无 stages 字段（旧数据），跳过报告生成/对抗成功率细分",
        }

    return {
        "num_questions": num,
        "success_count": success,
        "failed_count": summary.get("failed_count", num - success),
        "status_counts": dict(status_counter),
        "completion_rate": round(success / num, 4) if num else 0.0,
        "stage_rates": stage_rates,
        "total_elapsed_seconds": summary.get("total_elapsed_seconds"),
        "skip_ragas": bool(summary.get("skip_ragas")),
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# 2. 规则质量指标配对统计
# ---------------------------------------------------------------------------

# hallucination_rate 位于 rule_before/rule_after 顶层（非 metrics 内），且越低越好
TOP_LEVEL_RULE_KEYS = ("hallucination_rate",)

RULE_DIRECTION_NOTE = (
    "hallucination_rate 取自 rule_before/rule_after 顶层字段（非 metrics 内），"
    "该指标越低越好，diff 为负表示改善，该行显著性检验按改善方向（before − after）计算；"
    "其余指标越高越好，diff 为正表示改善"
)


def build_rule_metrics(results: list[dict]) -> dict:
    keys: list[str] = []
    for r in results:
        rb = r.get("rule_before") or {}
        for k in (rb.get("metrics") or {}):
            if k not in keys:
                keys.append(k)
    keys.append("composite_score")
    for top_key in TOP_LEVEL_RULE_KEYS:
        if any(top_key in (r.get("rule_before") or {}) or top_key in (r.get("rule_after") or {})
               for r in results):
            keys.append(top_key)

    out = {}
    for key in keys:
        pairs = []
        for r in results:
            rb, ra = r.get("rule_before"), r.get("rule_after")
            if not rb or not ra:
                continue
            if key == "composite_score":
                b, a = rb.get("composite_score"), ra.get("composite_score")
            elif key in TOP_LEVEL_RULE_KEYS:
                b, a = rb.get(key), ra.get(key)
            else:
                b = (rb.get("metrics") or {}).get(key)
                a = (ra.get("metrics") or {}).get(key)
            if _is_num(b) and _is_num(a):
                pairs.append((b, a))
        out[key] = _paired_stats(pairs, invert=key in TOP_LEVEL_RULE_KEYS)
    return out


# ---------------------------------------------------------------------------
# 2b. Red 对抗评分配对统计（report_before.red_* vs report_after.final/dimension_scores）
# ---------------------------------------------------------------------------

RED_DIMS = ("factual", "hallucination", "logic", "source", "coverage")

RED_NOTE = (
    "before = 对抗第 1 轮 Red 评分（评的是尚未修复的初始报告），"
    "after = 交付的历史最佳分数；对抗未执行导致分数为 null 的样本不进入配对"
)


def _norm_dim_key(k) -> str:
    """dimension_scores 的 key 可能是 enum 序列化字符串（如 'Dimension.FACTUAL'），归一化为小写维度名。"""
    return str(k).split(".")[-1].strip().lower()


def _load_detail(result: dict):
    """读取每题 output_file 明细 JSON；失败返回 None。"""
    output_file = result.get("output_file")
    if not output_file:
        return None
    path = Path(output_file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def build_red_adversarial(results: list[dict]) -> dict:
    overall_pairs: list[tuple[float, float]] = []
    dim_pairs: dict[str, list] = defaultdict(list)
    files_read = files_missing = 0

    for r in results:
        if r.get("status") != "done":
            continue
        detail = _load_detail(r)
        if detail is None:
            files_missing += 1
            detail = {}
        else:
            files_read += 1
        rb = detail.get("report_before") or r.get("report_before") or {}
        ra = detail.get("report_after") or r.get("report_after") or {}

        b, a = rb.get("red_overall_score"), ra.get("final_score")
        if _is_num(b) and _is_num(a):
            overall_pairs.append((b, a))

        bdims = {
            _norm_dim_key(k): v
            for k, v in (rb.get("red_dimension_scores") or {}).items()
        }
        adims = {
            _norm_dim_key(k): v
            for k, v in (ra.get("dimension_scores") or {}).items()
        }
        for dim in RED_DIMS:
            bv, av = bdims.get(dim), adims.get(dim)
            if _is_num(bv) and _is_num(av):
                dim_pairs[dim].append((bv, av))

    if not overall_pairs and not dim_pairs:
        return {
            "skipped": True,
            "reason": "所有成功样本的 red_overall_score/final_score 均为 null 或明细文件缺失",
            "files_read": files_read,
            "files_missing": files_missing,
        }
    return {
        "skipped": False,
        "files_read": files_read,
        "files_missing": files_missing,
        "overall": _paired_stats(overall_pairs),
        "dimensions": {dim: _paired_stats(dim_pairs.get(dim, [])) for dim in RED_DIMS},
    }


# ---------------------------------------------------------------------------
# 3. RAGAS 配对统计
# ---------------------------------------------------------------------------

ASPECT_NOTE = "AspectCritique 粒度较粗，仅作参考"


def build_ragas(results: list[dict], skip_ragas: bool) -> dict:
    if skip_ragas:
        return {"skipped": True, "reason": "summary 中 skip_ragas=true，跳过 RAGAS 统计"}

    metric_pairs: dict[str, list] = defaultdict(list)
    files_read = files_missing = 0
    for r in results:
        if r.get("status") != "done" or not r.get("output_file"):
            continue
        path = Path(r["output_file"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            files_missing += 1
            continue
        try:
            detail = _load_json(path)
        except Exception:
            files_missing += 1
            continue
        ragas = detail.get("ragas")
        if not isinstance(ragas, dict):
            continue
        files_read += 1
        for name in ("faithfulness", "answer_relevancy"):
            m = ragas.get(name) or {}
            b, a = m.get("before"), m.get("after")
            if _is_num(b) and _is_num(a):
                metric_pairs[name].append((b, a))
        ac = ragas.get("aspect_critique") or {}
        ac_before, ac_after = ac.get("before") or {}, ac.get("after") or {}
        if isinstance(ac_before, dict) and isinstance(ac_after, dict):
            for dim in set(ac_before) & set(ac_after):
                b, a = ac_before.get(dim), ac_after.get(dim)
                if _is_num(b) and _is_num(a):
                    metric_pairs[f"aspect_critique.{dim}"].append((b, a))

    if not metric_pairs:
        return {
            "skipped": True,
            "reason": "所有成功样本的 ragas 字段均缺失或无有效数值，跳过 RAGAS 统计",
            "files_read": files_read,
            "files_missing": files_missing,
        }

    out = {
        "skipped": False,
        "files_read": files_read,
        "files_missing": files_missing,
        "metrics": {},
    }
    for name, pairs in metric_pairs.items():
        stats = _paired_stats(pairs)
        if name.startswith("aspect_critique."):
            stats["note"] = ASPECT_NOTE
        out["metrics"][name] = stats
    return out


# ---------------------------------------------------------------------------
# 4. 过程指标（事件流）
# ---------------------------------------------------------------------------

def parse_events(path: Path) -> dict | None:
    if not path.exists():
        return None
    events = []
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    if not events:
        return None

    info: dict = {"bad_lines": bad_lines}
    info["total_elapsed_ms"] = events[-1].get("elapsed_ms")

    # 各状态累计耗时：相邻事件 elapsed_ms 差值归属到当前状态
    state_ms: dict[str, float] = defaultdict(float)
    current_state = "unknown"
    prev_elapsed = None
    for ev in events:
        elapsed = ev.get("elapsed_ms")
        if prev_elapsed is not None and _is_num(elapsed) and _is_num(prev_elapsed):
            state_ms[current_state] += max(elapsed - prev_elapsed, 0)
        if ev.get("type") == "state_transition":
            payload = ev.get("payload") or {}
            current_state = payload.get("to") or current_state
        elif current_state == "unknown" and ev.get("type") == "run_started":
            current_state = "idle"
        if _is_num(elapsed):
            prev_elapsed = elapsed
    info["state_durations_ms"] = dict(state_ms)

    # token / 任务状态 / 引用健康度
    token_total = 0
    task_status: Counter = Counter()
    citation_normalized = 0
    citation_invalid = 0
    tool_calls: Counter = Counter()
    tool_fails: Counter = Counter()
    task_started_ids: set = set()
    dag_task_count = None

    for ev in events:
        etype = ev.get("type")
        payload = ev.get("payload") or {}
        if etype == "task_completed":
            task_status[payload.get("status", "unknown")] += 1
            if _is_num(payload.get("token_usage")):
                token_total += payload["token_usage"]
            meta = payload.get("metadata") or {}
            diag = meta.get("citation_diagnostics") or {}
            if _is_num(diag.get("normalized_count")):
                citation_normalized += diag["normalized_count"]
            labels = diag.get("invalid_labels")
            if isinstance(labels, list):
                citation_invalid += len(labels)
        elif etype == "agent_loop_event":
            event = payload.get("event") or {}
            if event.get("role") == "tool":
                name = event.get("name") or "unknown"
                tool_calls[name] += 1
                if event.get("failed"):
                    tool_fails[name] += 1
        elif etype == "task_started":
            tid = payload.get("task_id")
            if tid:
                task_started_ids.add(tid)
        elif etype == "dag_created":
            tasks = payload.get("tasks")
            if isinstance(tasks, dict):
                dag_task_count = len(tasks)
            elif isinstance(tasks, list):
                dag_task_count = len(tasks)
            elif isinstance(payload.get("dag"), dict):
                nodes = payload["dag"].get("nodes")
                if isinstance(nodes, list):
                    dag_task_count = len(nodes)

    info["token_total"] = token_total
    info["task_status"] = dict(task_status)
    info["citation_normalized_total"] = citation_normalized
    info["citation_invalid_total"] = citation_invalid
    info["tool_calls"] = dict(tool_calls)
    info["tool_fails"] = dict(tool_fails)
    info["subtask_count"] = (
        dag_task_count if dag_task_count is not None else len(task_started_ids)
    )
    info["subtask_count_source"] = (
        "dag_created" if dag_task_count is not None else "task_started"
    )
    return info


def resolve_events_path(result: dict) -> tuple[Path | None, str]:
    """按新目录布局解析 events.jsonl，依次尝试：
    1. result["run_dir"]/events.jsonl
    2. result["question_dir"]/events/*/events.jsonl（glob 取唯一/最新）
    3. 老位置 outputs/debug_runs/<run_id>/events.jsonl
    返回 (路径, 来源标签)；都找不到返回 (None, "missing")。
    """
    run_dir = result.get("run_dir")
    if run_dir:
        p = Path(run_dir)
        if not p.is_absolute():
            p = ROOT / p
        f = p / "events.jsonl"
        if f.exists():
            return f, "run_dir"

    question_dir = result.get("question_dir")
    if question_dir:
        p = Path(question_dir)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            candidates = sorted(p.glob("events/*/events.jsonl"))
            if candidates:
                return candidates[-1], "question_dir_glob"

    run_id = result.get("run_id")
    if run_id:
        f = ROOT / "outputs" / "debug_runs" / run_id / "events.jsonl"
        if f.exists():
            return f, "debug_runs_legacy"

    return None, "missing"


def build_process_metrics(results: list[dict]) -> dict:
    per_question = []
    missing_events = 0
    for r in results:
        path, events_source = resolve_events_path(r)
        info = parse_events(path) if path else None
        if info is None:
            missing_events += 1
            continue
        summary_s = r.get("elapsed_seconds")
        events_s = (
            round(info["total_elapsed_ms"] / 1000, 3)
            if _is_num(info.get("total_elapsed_ms"))
            else None
        )
        per_question.append(
            {
                "question_id": r.get("question_id"),
                "run_id": r.get("run_id"),
                "events_source": events_source,
                "summary_elapsed_seconds": summary_s,
                "events_elapsed_seconds": events_s,
                "elapsed_diff_seconds": (
                    round(events_s - summary_s, 3)
                    if _is_num(events_s) and _is_num(summary_s)
                    else None
                ),
                **info,
            }
        )

    # 跨题聚合
    n = len(per_question)
    state_acc: dict[str, list[float]] = defaultdict(list)
    for q in per_question:
        for state, ms in (q.get("state_durations_ms") or {}).items():
            state_acc[state].append(ms / 1000.0)
    state_avg = {
        state: {
            "mean_seconds": round(mean(v), 3),
            "n_questions": len(v),
        }
        for state, v in sorted(state_acc.items())
    }

    tool_calls_total: Counter = Counter()
    tool_fails_total: Counter = Counter()
    for q in per_question:
        tool_calls_total.update(q.get("tool_calls") or {})
        tool_fails_total.update(q.get("tool_fails") or {})
    tools = {
        name: {
            "calls": tool_calls_total[name],
            "failures": tool_fails_total.get(name, 0),
            "failure_rate": round(
                tool_fails_total.get(name, 0) / tool_calls_total[name], 4
            ),
        }
        for name in sorted(tool_calls_total)
    }

    task_status_total: Counter = Counter()
    for q in per_question:
        task_status_total.update(q.get("task_status") or {})

    # 阶段耗时（results[].timing 优先，缺失时回退到每题明细 JSON 顶层 timing）
    timing_research: list[float] = []
    timing_adv: list[float] = []
    timing_total: list[float] = []
    timing_adv_null = 0
    n_with_timing = 0
    # token 真实用量（results[].token_usage 优先，缺失时回退明细 JSON 顶层）
    tok_by_module: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    tok_per_q_total: list[float] = []
    tok_est_ratios: list[float] = []
    n_with_token = 0

    for r in results:
        detail_cache = None

        timing = r.get("timing")
        if not isinstance(timing, dict):
            detail_cache = _load_detail(r)
            timing = (detail_cache or {}).get("timing")
        if isinstance(timing, dict):
            n_with_timing += 1
            v = timing.get("research_seconds")
            if _is_num(v):
                timing_research.append(v)
            v = timing.get("adversarial_seconds")
            if _is_num(v):
                timing_adv.append(v)
            else:
                timing_adv_null += 1
            v = timing.get("total_seconds")
            if _is_num(v):
                timing_total.append(v)

        tu = r.get("token_usage")
        if not isinstance(tu, dict):
            if detail_cache is None:
                detail_cache = _load_detail(r)
            tu = (detail_cache or {}).get("token_usage")
        if isinstance(tu, dict):
            n_with_token += 1
            for mod, m in (tu.get("by_module") or {}).items():
                if not isinstance(m, dict):
                    continue
                agg = tok_by_module[mod]
                for f in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                    if _is_num(m.get(f)):
                        agg[f] += m[f]
            total = tu.get("total_tokens")
            if _is_num(total):
                tok_per_q_total.append(total)
            est = tu.get("estimated_task_tokens")
            if _is_num(est) and _is_num(total) and total > 0:
                tok_est_ratios.append(est / total)

    timing_block = {
        "questions_with_data": n_with_timing,
        "research_seconds": _dist(timing_research),
        "adversarial_seconds": {
            **_dist(timing_adv),
            "null_count": timing_adv_null,
            "note": "adversarial_seconds 为 null（对抗未执行）的样本不进入对抗耗时统计",
        },
        "total_seconds": _dist(timing_total),
    }
    if n_with_timing == 0:
        timing_block["note"] = "样本均无 timing 字段（旧数据），阶段耗时统计不可用"

    if n_with_token:
        token_block = {
            "source": "api",
            "note": "by_module 为 API 真实用量；estimated_task_tokens 为 agent 记录的字符估算值，供对照",
            "questions_with_data": n_with_token,
            "by_module": {m: dict(a) for m, a in sorted(tok_by_module.items())},
            "total_tokens": _dist(tok_per_q_total),
            "estimated_vs_real": {
                "ratio_mean": round(mean(tok_est_ratios), 4) if tok_est_ratios else None,
                "n": len(tok_est_ratios),
                "note": "estimated_task_tokens / total_tokens 的跨题均值，>1 表示估算偏高",
            },
        }
    else:
        token_block = {
            "source": "events_estimate",
            "note": "除 Blue SEARCH 修复外为 字符数÷3 估算值（样本无 token_usage 字段，回退到 events.jsonl 口径）",
            **_dist([q["token_total"] for q in per_question]),
        }

    # 对抗指标（来自 summary 的 report_after，仅成功样本）
    adv_rounds, final_scores = [], []
    for r in results:
        if r.get("status") != "done":
            continue
        ra = r.get("report_after") or {}
        if _is_num(ra.get("adversarial_rounds")):
            adv_rounds.append(ra["adversarial_rounds"])
        if _is_num(ra.get("final_score")):
            final_scores.append(ra["final_score"])

    return {
        "questions_with_events": n,
        "missing_events_count": missing_events,
        "per_question": per_question,
        "elapsed_seconds": _dist(
            [q["events_elapsed_seconds"] for q in per_question]
        ),
        "timing": timing_block,
        "state_durations_avg": state_avg,
        "token_usage": token_block,
        "tools": tools,
        "subtask_count": _dist([q["subtask_count"] for q in per_question]),
        "task_status_distribution": dict(task_status_total),
        "citation_health": {
            "normalized_count_total": sum(
                q["citation_normalized_total"] for q in per_question
            ),
            "invalid_labels_total": sum(
                q["citation_invalid_total"] for q in per_question
            ),
        },
        "adversarial": {
            "rounds_distribution": dict(
                sorted(Counter(adv_rounds).items(), key=lambda kv: kv[0])
            ),
            "final_score": _dist(final_scores),
        },
    }


# ---------------------------------------------------------------------------
# 5. JSON 解析鲁棒性
# ---------------------------------------------------------------------------

def build_json_parse_stats(results: list[dict]) -> dict:
    stages: dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "baseline_w": 0.0,
            "heuristic_w": 0.0,
            "final_w": 0.0,
            "saved_by_heuristic": 0,
            "saved_by_json_repair": 0,
            "failed": 0,
        }
    )
    questions_with_data = 0
    for r in results:
        jps = r.get("json_parse_summary") or {}
        by_stage = jps.get("by_stage") or {}
        if not by_stage:
            continue
        questions_with_data += 1
        for stage, s in by_stage.items():
            agg = stages[stage]
            count = s.get("count") or 0
            if not _is_num(count):
                count = 0
            agg["count"] += count
            for field, key in (
                ("baseline_success_rate", "baseline_w"),
                ("heuristic_success_rate", "heuristic_w"),
                ("final_success_rate", "final_w"),
            ):
                rate = s.get(field)
                if _is_num(rate):
                    agg[key] += rate * count
            for f in ("saved_by_heuristic", "saved_by_json_repair", "failed"):
                v = s.get(f)
                if _is_num(v):
                    agg[f] += v

    out = {"questions_with_data": questions_with_data, "by_stage": {}}
    for stage, agg in sorted(stages.items()):
        count = agg["count"]
        entry = {
            "total_count": count,
            "saved_by_heuristic": agg["saved_by_heuristic"],
            "saved_by_json_repair": agg["saved_by_json_repair"],
            "failed": agg["failed"],
        }
        for key, name in (
            ("baseline_w", "baseline_success_rate_weighted"),
            ("heuristic_w", "heuristic_success_rate_weighted"),
            ("final_w", "final_success_rate_weighted"),
        ):
            entry[name] = round(agg[key] / count, 4) if count else None
        out["by_stage"][stage] = entry
    return out


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------

def _md_paired_table(title: str, metrics: dict, extra_note: str = "") -> list[str]:
    lines = [f"### {title}", ""]
    if extra_note:
        lines += [f"> {extra_note}", ""]
    lines.append(
        "| 指标 | n | before 均值 | after 均值 | diff | 95% CI | bootstrap p | Cohen's d | t 检验 p | 结论 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, s in metrics.items():
        if s.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | — | — | — | — | — | — | — | 无有效配对样本 |")
            continue
        boot = s.get("bootstrap") or {}
        ci = f"[{boot.get('ci_lower')}, {boot.get('ci_upper')}]"
        d = s.get("cohens_d", "—")
        t_p = (s.get("t_test") or {}).get("p_value", "—")
        lines.append(
            f"| {name} | {s['n']} | {s['before_mean']} | {s['after_mean']} | "
            f"{s['diff']} | {ci} | {boot.get('p_value', '—')} | {d} | {t_p} | {_sig_wording(s)} |"
        )
    lines.append("")
    return lines


def render_markdown(report: dict) -> str:
    L: list[str] = []
    ov = report["overview"]
    L.append(f"# 实验聚合报告（{report['generated_at']}）")
    L.append("")
    L.append(f"- 汇总文件：`{report['summary_file']}`")
    L.append(f"- 实验时间戳：{report.get('experiment_timestamp', '—')}")
    L.append("")

    # 1. 总览
    L.append("## 1. 总览")
    L.append("")
    L.append(f"- 题数：{ov['num_questions']}")
    L.append(
        f"- 成功：{ov['success_count']}；失败：{ov['failed_count']}；"
        f"完成率：{ov['completion_rate'] * 100:.1f}%"
    )
    L.append(f"- 状态分布：{json.dumps(ov['status_counts'], ensure_ascii=False)}")
    sr = ov.get("stage_rates") or {}
    if sr.get("n_with_stages"):
        L.append(
            f"- 报告生成成功率：{sr['report_generated_count']}/{sr['n_with_stages']}"
            f"（{sr['report_generated_rate'] * 100:.1f}%）"
        )
        if sr.get("adversarial_completed_rate") is not None:
            L.append(
                f"- 对抗完成率（基数为报告已生成样本）：{sr['adversarial_completed_count']}/{sr['report_generated_count']}"
                f"（{sr['adversarial_completed_rate'] * 100:.1f}%）"
            )
    else:
        L.append(f"- 分阶段成功率：{sr.get('note', '不可用')}")
    L.append(f"- 端到端 done 率：{ov['completion_rate'] * 100:.1f}%")
    if _is_num(ov.get("total_elapsed_seconds")):
        L.append(f"- 总耗时：{ov['total_elapsed_seconds']:.1f} 秒")
    L.append("")
    L.append("| question_id | 状态 | run_id | 耗时（秒） |")
    L.append("|---|---|---|---|")
    for q in ov["per_question"]:
        el = q.get("elapsed_seconds")
        L.append(
            f"| {q.get('question_id')} | {q.get('status')} | {q.get('run_id') or '—'} | "
            f"{f'{el:.1f}' if _is_num(el) else '—'} |"
        )
    L.append("")

    # 2. 规则质量指标
    L.append("## 2. 规则质量指标配对统计（rule_before vs rule_after）")
    L.append("")
    rule_display = {
        (f"{k} *" if k in TOP_LEVEL_RULE_KEYS else k): v
        for k, v in report["rule_metrics"].items()
    }
    L += _md_paired_table(
        "规则指标",
        rule_display,
        extra_note=f"* {RULE_DIRECTION_NOTE}。",
    )

    # 2b. Red 对抗评分
    L.append("## 3. Red 对抗评分配对统计")
    L.append("")
    red = report["red_adversarial"]
    if red.get("skipped"):
        L.append(f"本节跳过：{red.get('reason')}")
        L.append("")
    else:
        L += _md_paired_table(
            "总体评分（red_overall_score vs final_score）",
            {"overall_score": red["overall"]},
            extra_note=RED_NOTE,
        )
        L += _md_paired_table(
            "分维度评分（red_dimension_scores vs dimension_scores）",
            red["dimensions"],
        )
        L.append(
            f"（读取明细文件 {red.get('files_read', 0)} 个，缺失/不可读 {red.get('files_missing', 0)} 个）"
        )
        L.append("")

    # 3. RAGAS
    L.append("## 4. RAGAS 配对统计")
    L.append("")
    ragas = report["ragas"]
    if ragas.get("skipped"):
        L.append(f"本节跳过：{ragas.get('reason')}")
        L.append("")
    else:
        core = {
            k: v
            for k, v in ragas["metrics"].items()
            if not k.startswith("aspect_critique.")
        }
        aspect = {
            k: v
            for k, v in ragas["metrics"].items()
            if k.startswith("aspect_critique.")
        }
        if core:
            L += _md_paired_table("Faithfulness / Answer Relevancy", core)
        if aspect:
            L += _md_paired_table(
                "AspectCritique 各维度",
                aspect,
                extra_note=ASPECT_NOTE + "；NaN 已过滤。",
            )
        L.append(
            f"（读取明细文件 {ragas.get('files_read', 0)} 个，缺失/不可读 {ragas.get('files_missing', 0)} 个）"
        )
        L.append("")

    # 4. 过程指标
    pm = report["process_metrics"]
    L.append("## 5. 过程指标（事件流）")
    L.append("")
    L.append(
        f"- 覆盖题目：{pm['questions_with_events']}；"
        f"缺少 events.jsonl 而跳过：{pm['missing_events_count']}"
    )
    el = pm["elapsed_seconds"]
    if el.get("n"):
        L.append(
            f"- 每题总耗时（事件流，秒）：均值 {el['mean']}，中位 {el['median']}，"
            f"范围 [{el['min']}, {el['max']}]"
        )
    L.append("")

    if pm["per_question"]:
        L.append("### 5.1 每题耗时对照（summary vs 事件流）")
        L.append("")
        L.append("| question_id | summary（秒） | events（秒） | 差值（秒） |")
        L.append("|---|---|---|---|")
        for q in pm["per_question"]:
            s = q.get("summary_elapsed_seconds")
            e = q.get("events_elapsed_seconds")
            d = q.get("elapsed_diff_seconds")
            L.append(
                f"| {q['question_id']} | {f'{s:.1f}' if _is_num(s) else '—'} | "
                f"{f'{e:.1f}' if _is_num(e) else '—'} | "
                f"{f'{d:.1f}' if _is_num(d) else '—'} |"
            )
        L.append("")

    # 阶段耗时（timing 字段，优先于事件流分解）
    tb = pm.get("timing") or {}
    L.append("### 5.2 阶段耗时统计（results[].timing）")
    L.append("")
    if tb.get("questions_with_data"):
        L.append("| 阶段 | n | 均值（秒） | 中位（秒） | 最小（秒） | 最大（秒） |")
        L.append("|---|---|---|---|---|---|")
        for label, key in (
            ("报告生成（research_seconds）", "research_seconds"),
            ("对抗（adversarial_seconds）", "adversarial_seconds"),
            ("总计（total_seconds）", "total_seconds"),
        ):
            d = tb.get(key) or {}
            if d.get("n"):
                L.append(
                    f"| {label} | {d['n']} | {d['mean']} | {d['median']} | {d['min']} | {d['max']} |"
                )
            else:
                L.append(f"| {label} | 0 | — | — | — | — |")
        L.append("")
        L.append(
            f"> 注：{tb['adversarial_seconds']['note']}（本次 null 样本 {tb['adversarial_seconds'].get('null_count', 0)} 个）。"
        )
        L.append("")
    else:
        L.append(f"{tb.get('note', 'timing 字段不可用')}")
        L.append("")

    if pm["state_durations_avg"]:
        L.append("### 5.3 各状态累计耗时（事件流分解，补充）")
        L.append("")
        L.append("| 状态 | 平均耗时（秒） | 覆盖题数 |")
        L.append("|---|---|---|")
        for state, s in pm["state_durations_avg"].items():
            L.append(f"| {state} | {s['mean_seconds']} | {s['n_questions']} |")
        L.append("")

    tok = pm["token_usage"]
    L.append("### 5.4 Token 用量")
    L.append("")
    L.append(f"> 注：{tok['note']}。")
    L.append("")
    if tok.get("source") == "api":
        tt = tok["total_tokens"]
        if tt.get("n"):
            L.append(
                f"- 每题 total_tokens（真实 API 用量）：均值 {tt['mean']}，中位 {tt['median']}，"
                f"范围 [{tt['min']}, {tt['max']}]（n={tt['n']}）"
            )
        er = tok.get("estimated_vs_real") or {}
        if er.get("ratio_mean") is not None:
            L.append(
                f"- estimated_task_tokens / total_tokens 跨题均值比值：{er['ratio_mean']}"
                f"（n={er['n']}；>1 表示字符估算偏高）"
            )
        L.append("")
        if tok.get("by_module"):
            L.append("| 模块 | calls | prompt_tokens | completion_tokens | total_tokens |")
            L.append("|---|---|---|---|---|")
            for mod, a in tok["by_module"].items():
                L.append(
                    f"| {mod} | {a['calls']} | {a['prompt_tokens']} | "
                    f"{a['completion_tokens']} | {a['total_tokens']} |"
                )
            L.append("")
    else:
        if tok.get("n"):
            L.append(
                f"- 每题 token 总量（events.jsonl 估算口径）：均值 {tok['mean']}，中位 {tok['median']}，"
                f"范围 [{tok['min']}, {tok['max']}]（n={tok['n']}）"
            )
            L.append("")

    if pm["tools"]:
        L.append("### 5.5 工具调用")
        L.append("")
        L.append("| 工具 | 调用次数 | 失败次数 | 失败率 |")
        L.append("|---|---|---|---|")
        for name, t in pm["tools"].items():
            L.append(f"| {name} | {t['calls']} | {t['failures']} | {t['failure_rate'] * 100:.1f}% |")
        L.append("")

    sub = pm["subtask_count"]
    L.append("### 5.6 子任务与任务状态")
    L.append("")
    if sub.get("n"):
        L.append(
            f"- 每题子任务数：均值 {sub['mean']}，范围 [{sub['min']}, {sub['max']}]（n={sub['n']}）"
        )
    L.append(
        f"- task_completed 状态分布：{json.dumps(pm['task_status_distribution'], ensure_ascii=False)}"
    )
    L.append("")

    ch = pm["citation_health"]
    L.append("### 5.7 引用健康度")
    L.append("")
    L.append(f"- citation normalized_count 合计：{ch['normalized_count_total']}")
    L.append(f"- invalid_labels 总数：{ch['invalid_labels_total']}")
    L.append("")

    adv = pm["adversarial"]
    L.append("### 5.8 对抗评审")
    L.append("")
    L.append(
        f"- adversarial_rounds 分布：{json.dumps(adv['rounds_distribution'], ensure_ascii=False)}"
    )
    fs = adv["final_score"]
    if fs.get("n"):
        L.append(
            f"- final_score：均值 {fs['mean']}，中位 {fs['median']}，"
            f"范围 [{fs['min']}, {fs['max']}]（n={fs['n']}）"
        )
    L.append("")

    # 5. JSON 解析鲁棒性
    jp = report["json_parse"]
    L.append("## 6. JSON 解析鲁棒性")
    L.append("")
    L.append(f"- 含解析统计的题目数：{jp['questions_with_data']}")
    L.append("")
    if jp["by_stage"]:
        L.append(
            "| stage | 总记录数 | baseline 成功率（加权） | heuristic 成功率（加权） | final 成功率（加权） | saved_by_heuristic | saved_by_json_repair | failed |"
        )
        L.append("|---|---|---|---|---|---|---|---|")
        for stage, s in jp["by_stage"].items():
            L.append(
                f"| {stage} | {s['total_count']} | {s['baseline_success_rate_weighted']} | "
                f"{s['heuristic_success_rate_weighted']} | {s['final_success_rate_weighted']} | "
                f"{s['saved_by_heuristic']} | {s['saved_by_json_repair']} | {s['failed']} |"
            )
        L.append("")

    # 6. 失败样本清单
    L.append("## 7. 失败样本清单")
    L.append("")
    if report["failures"]:
        L.append("| question_id | status | run_id | error |")
        L.append("|---|---|---|---|")
        for f in report["failures"]:
            err = str(f.get("error") or "—").replace("|", "\\|")
            L.append(f"| {f.get('question_id')} | {f.get('status')} | {f.get('run_id') or '—'} | {err} |")
    else:
        L.append("无失败样本。")
    L.append("")

    # 7. 已知限制
    L.append("## 8. 已知限制")
    L.append("")
    if pm["token_usage"].get("source") == "api":
        L.append("- token 为 API 真实用量（按模块包装客户端统计），estimated_task_tokens 为字符估算对照。")
    else:
        L.append("- token 用量为估算值（除 Blue SEARCH 修复外为 字符数÷3 估算值），非真实 tokenizer 计数。")
    L.append(
        "- 每题仅有单一样本，无题内方差；显著性检验基于 n≤20 的题间配对，统计功效有限，"
        "p 值不显著时仅表示“未检测到显著差异”，不代表无实际差异。"
    )
    L.append("- AspectCritique 粒度较粗，结果仅作参考。")
    L.append("- judge LLM 与被评系统同为 deepseek，存在自评偏差风险。")
    L.append("- 超时/失败样本只计入完成率，不进入配对统计。")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def find_latest_summary(outputs_dir: Path) -> Path:
    candidates = sorted(outputs_dir.glob("eval_json_ablation_summary_*.json"))
    candidates += sorted(
        (outputs_dir / "evaluation").glob("eval_json_ablation_summary_*.json")
    )
    candidates = sorted(candidates)
    if not candidates:
        sys.exit(
            f"未在 {outputs_dir} 及其 evaluation/ 子目录找到 "
            "eval_json_ablation_summary_*.json，请用 --summary 指定"
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合端到端评测实验结果")
    parser.add_argument("--summary", default=None, help="汇总 JSON 路径（缺省自动选最新）")
    parser.add_argument("-o", default=str(ROOT / "outputs"), help="报告输出目录")
    args = parser.parse_args()

    summary_path = (
        Path(args.summary) if args.summary else find_latest_summary(ROOT / "outputs")
    )
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path = summary_path.resolve()
    summary = _load_json(summary_path)
    results = summary.get("results", [])

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_file": str(summary_path),
        "experiment_timestamp": summary.get("timestamp"),
        "overview": build_overview(summary),
        "rule_metrics": build_rule_metrics(results),
        "red_adversarial": build_red_adversarial(results),
        "ragas": build_ragas(results, bool(summary.get("skip_ragas"))),
        "process_metrics": build_process_metrics(results),
        "json_parse": build_json_parse_stats(results),
        "failures": [
            {
                "question_id": r.get("question_id"),
                "status": r.get("status"),
                "run_id": r.get("run_id"),
                "error": r.get("error"),
            }
            for r in results
            if r.get("status") != "done"
        ],
    }

    # summary 位于 outputs/evaluation/ 时，报告写到同一目录；否则用 -o 指定目录
    if summary_path.parent == (ROOT / "outputs" / "evaluation").resolve():
        out_dir = summary_path.parent
    else:
        out_dir = Path(args.o)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"experiment_report_{ts}.json"
    md_path = out_dir / f"experiment_report_{ts}.md"

    report = _sanitize(report)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"JSON 报告：{json_path}")
    print(f"Markdown 报告：{md_path}")


if __name__ == "__main__":
    main()
