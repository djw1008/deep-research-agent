#!/usr/bin/env python3
"""
对抗模块消融实验：对比 adversarial=off vs adversarial=on。
运行前3题，配对比较 + Bootstrap CI + Cohen's d。

用法: python scripts/run_ablation.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml

from deep_research.agents import AgentPool
from deep_research.core import Orchestrator, ResearchContext
from deep_research.evaluation.benchmarks import ResearchBench
from deep_research.evaluation.metrics import bootstrap_ci_paired, cohens_d
from deep_research.memory import KnowledgeBase, SessionMemory
from deep_research.memory.embedder import MemoryEmbedder
from deep_research.models import ModelRouter
from deep_research.planner import Planner
from deep_research.tools import (
    ArxivReaderTool, BrowserTool, CalculatorTool, CodeSandboxTool,
    FileReaderTool, NotepadTool, ToolRegistry, WebSearchTool,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("openai", "httpx", "httpcore", "primp", "urllib3", "aiosqlite"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("ablation")

NUM_QUESTIONS = 3


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_tool_registry(_config: dict) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(BrowserTool())
    registry.register(ArxivReaderTool())
    registry.register(FileReaderTool())
    registry.register(CodeSandboxTool())
    registry.register(CalculatorTool())
    registry.register(NotepadTool())
    return registry


def build_orchestrator(config: dict, session_memory, knowledge_base) -> Orchestrator:
    model_cfg = config.get("model", {})
    backend = model_cfg.get("backend", "deepseek")

    client_kwargs: dict = {}
    for key in ("base_model", "base_url", "api_key", "temperature", "top_p", "max_tokens"):
        if key in model_cfg:
            client_kwargs[key] = model_cfg[key]

    llm_client = ModelRouter.create_backend(backend, **client_kwargs)
    logger.info("LLM: %s", backend)

    planner = Planner(llm_client)
    registry = create_tool_registry(config)

    sampling_cfg = config.get("model", {}).get("backend_sampling", {})
    backend_defaults = sampling_cfg.get(backend, {})
    module_overrides = sampling_cfg.get("modules", {})
    backend_map = config.get("model", {}).get("backend_mapping", {})

    _TASK_TO_MODULE = {
        "search": "solver", "analyze": "solver", "verify": "solver",
        "synthesize": "summarizer", "red_agent": "red_agent",
        "blue_agent": "blue_agent", "issue_arbiter": "issue_arbiter",
    }

    def policy_factory(task_type: str = "search"):
        module = _TASK_TO_MODULE.get(task_type, "solver")
        merged = dict(backend_defaults)
        merged.update(client_kwargs)
        if module in module_overrides:
            merged.update(module_overrides[module])
        mod_backend = backend_map.get(module, backend)
        return ModelRouter.create_backend(mod_backend, **merged)

    def tools_factory():
        return list(registry._tools.values())

    orch_cfg = config.get("orchestrator", {})
    agent_pool = AgentPool(
        policy_factory=policy_factory, tools_factory=tools_factory,
        max_idle=orch_cfg.get("max_concurrent", 5), config=config,
        session_memory=session_memory, knowledge_base=knowledge_base,
    )

    issue_arbiter_client = None
    if config.get("adversarial", {}).get("issue_arbiter", {}).get("enabled", True):
        try:
            issue_arbiter_client = policy_factory("issue_arbiter")
        except Exception:
            pass

    return Orchestrator(
        planner=planner, agent_pool=agent_pool,
        session_memory=session_memory, knowledge_base=knowledge_base,
        config=config, issue_arbiter_client=issue_arbiter_client,
    )


async def run_single_query(query: str, config: dict, sm, kb) -> tuple:
    orch = build_orchestrator(config, sm, kb)
    await orch.knowledge_base.initialize()
    await orch.session_memory.initialize()

    adversarial = config.get("adversarial", {}).get("enabled", False)
    context = ResearchContext(
        topic=query,
        max_iterations=config.get("orchestrator", {}).get("max_replan_rounds", 3),
        enable_adversarial=adversarial,
    )

    try:
        timeout = config.get("orchestrator", {}).get("global_timeout_seconds", None)
        final_state = await orch.run(context, timeout_seconds=timeout,
                                     session_id=None, round=1, previous_session_context="")
        report = getattr(orch, "_report", None)
        report_text = getattr(report, "content", "") if report else ""
        return final_state, report_text, report
    finally:
        await orch.knowledge_base.close()
        await orch.session_memory.close()


async def main():
    config = load_config(str(PROJECT_ROOT / "config" / "default.yaml"))
    bench = ResearchBench()
    questions = bench.get_questions(n=NUM_QUESTIONS)

    memory_cfg = config.get("memory", {})
    session_memory = SessionMemory(db_path=memory_cfg.get("session_db_path", "data/session_memory.db"))
    knowledge_base = KnowledgeBase(
        db_path=memory_cfg.get("knowledge_db_path", "data/knowledge_base.db"),
        embedder=MemoryEmbedder(), config=memory_cfg,
    )
    await session_memory.initialize()
    await knowledge_base.initialize()

    results = {"adversarial_off": [], "adversarial_on": []}

    try:
        for mode, label in [(False, "OFF"), (True, "ON")]:
            config.setdefault("adversarial", {})["enabled"] = mode
            logger.info("\n" + "█" * 60)
            logger.info(" 对抗模块: %s", label)
            logger.info("█" * 60)

            for i, q in enumerate(questions, 1):
                logger.info("\n[%s] [%d/%d] %s", label, i, len(questions), q["id"])
                state, report_text, report_obj = await run_single_query(
                    q["query"], config, session_memory, knowledge_base,
                )

                num_sources = len(getattr(report_obj, "sources", [])) if report_obj else 0
                eval_result = bench.evaluate_report(report_text, q["id"], num_sources=num_sources)

                results[f"adversarial_{'on' if mode else 'off'}"].append({
                    "question_id": q["id"],
                    "state": state.value,
                    "report_length": len(report_text),
                    "num_sources": num_sources,
                    "composite_score": eval_result["composite_score"],
                    "metrics": eval_result["metrics"],
                    "hallucination_rate": eval_result["hallucination_rate"],
                })

                logger.info("  score=%.3f | factual=%.2f | halluc=%.2f | logic=%.2f",
                            eval_result["composite_score"],
                            eval_result["metrics"]["factual_accuracy"],
                            eval_result["hallucination_rate"],
                            eval_result["metrics"]["logical_consistency"])

    finally:
        await session_memory.close()
        await knowledge_base.close()

    # =========================================================================
    # 统计对比
    # =========================================================================
    off_scores = [r["composite_score"] for r in results["adversarial_off"]]
    on_scores = [r["composite_score"] for r in results["adversarial_on"]]
    diffs = [a - b for a, b in zip(on_scores, off_scores)]

    ci = bootstrap_ci_paired(diffs)
    d = cohens_d(on_scores, off_scores)

    print("\n" + "=" * 60)
    print(" 对抗模块消融结果")
    print("=" * 60)
    print(f"\n{'题目':<12} {'OFF':>8} {'ON':>8} {'Δ':>8}")
    print("-" * 36)
    for i, q in enumerate(questions):
        print(f"{q['id']:<12} {off_scores[i]:>8.3f} {on_scores[i]:>8.3f} {diffs[i]:>+8.3f}")
    print("-" * 36)
    avg_off = sum(off_scores) / len(off_scores)
    avg_on = sum(on_scores) / len(on_scores)
    print(f"{'平均':<12} {avg_off:>8.3f} {avg_on:>8.3f} {avg_on - avg_off:>+8.3f}")

    print(f"\n统计检验:")
    print(f"  均值差异:        {ci['mean_diff']:+.3f}")
    print(f"  Bootstrap 95% CI: [{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}]")
    print(f"  p-value:          {ci['p_value']:.4f}")
    print(f"  统计显著:         {'是 ✅' if ci['significant'] else '否 ❌'}")
    print(f"  Cohen's d:        {d:+.3f}")

    # 幻觉率对比
    off_halluc = [r["hallucination_rate"] for r in results["adversarial_off"]]
    on_halluc = [r["hallucination_rate"] for r in results["adversarial_on"]]
    print(f"\n幻觉率对比:")
    print(f"  OFF: {sum(off_halluc)/len(off_halluc):.3f}  →  ON: {sum(on_halluc)/len(on_halluc):.3f}")

    # 保存
    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_questions": NUM_QUESTIONS,
        "adversarial_off": results["adversarial_off"],
        "adversarial_on": results["adversarial_on"],
        "statistics": ci,
        "cohens_d": d,
    }
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ablation_adversarial_{output['timestamp']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {path}")


if __name__ == "__main__":
    asyncio.run(main())
