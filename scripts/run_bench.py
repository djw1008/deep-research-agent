#!/usr/bin/env python3
"""
运行 ResearchBench 评测：前5题，对抗模块关闭。

用法: python scripts/run_bench.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# 允许未安装包时直接运行
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

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("openai", "httpx", "httpcore", "primp", "urllib3", "aiosqlite"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# 辅助函数（从 run.py 精简）
# ---------------------------------------------------------------------------
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


def build_orchestrator(
    config: dict,
    session_memory: SessionMemory,
    knowledge_base: KnowledgeBase,
) -> Orchestrator:
    model_cfg = config.get("model", {})
    backend = model_cfg.get("backend", "deepseek")

    client_kwargs: dict = {}
    if "base_model" in model_cfg:
        client_kwargs["model_name"] = model_cfg["base_model"]
    if "base_url" in model_cfg:
        client_kwargs["base_url"] = model_cfg["base_url"]
    if "api_key" in model_cfg:
        client_kwargs["api_key"] = model_cfg["api_key"]
    for key in ("temperature", "top_p", "max_tokens"):
        if key in model_cfg:
            client_kwargs[key] = model_cfg[key]

    try:
        llm_client = ModelRouter.create_backend(backend, **client_kwargs)
        logger.info("LLM 后端已初始化: %s", backend)
    except ValueError as e:
        logger.error("LLM 后端初始化失败: %s", e)
        raise SystemExit(1)

    planner = Planner(llm_client)
    registry = create_tool_registry(config)

    sampling_cfg = config.get("model", {}).get("backend_sampling", {})
    backend_defaults = sampling_cfg.get(backend, {})
    module_overrides = sampling_cfg.get("modules", {})
    backend_map = config.get("model", {}).get("backend_mapping", {})

    _TASK_TO_MODULE = {
        "search": "solver",
        "analyze": "solver",
        "verify": "solver",
        "synthesize": "summarizer",
        "red_agent": "red_agent",
        "blue_agent": "blue_agent",
        "issue_arbiter": "issue_arbiter",
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
        policy_factory=policy_factory,
        tools_factory=tools_factory,
        max_idle=orch_cfg.get("max_concurrent", 5),
        config=config,
        session_memory=session_memory,
        knowledge_base=knowledge_base,
    )

    # IssueMerger 的 LLM 仲裁
    issue_arbiter_client = None
    arbiter_cfg = config.get("adversarial", {}).get("issue_arbiter", {})
    if arbiter_cfg.get("enabled", True):
        try:
            issue_arbiter_client = policy_factory("issue_arbiter")
        except Exception:
            logger.exception("Issue 仲裁 LLM 初始化失败")

    return Orchestrator(
        planner=planner,
        agent_pool=agent_pool,
        session_memory=session_memory,
        knowledge_base=knowledge_base,
        config=config,
        issue_arbiter_client=issue_arbiter_client,
    )


async def run_single_query(
    query: str,
    config: dict,
    session_memory: SessionMemory,
    knowledge_base: KnowledgeBase,
) -> tuple:
    """执行单次研究，返回 (final_state, report_text)。"""
    orch = build_orchestrator(config, session_memory, knowledge_base)
    await orch.knowledge_base.initialize()
    await orch.session_memory.initialize()

    context = ResearchContext(
        topic=query,
        max_iterations=config.get("orchestrator", {}).get("max_replan_rounds", 3),
        enable_adversarial=config.get("adversarial", {}).get("enabled", False),
    )

    try:
        orch_cfg = config.get("orchestrator", {})
        timeout = orch_cfg.get("global_timeout_seconds", None)

        final_state = await orch.run(
            context,
            timeout_seconds=timeout,
            session_id=None,
            round=1,
            previous_session_context="",
        )
        report = getattr(orch, "_report", None)
        report_text = getattr(report, "content", "") if report else ""
        return final_state, report_text, report
    finally:
        await orch.knowledge_base.close()
        await orch.session_memory.close()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main():
    config = load_config(str(PROJECT_ROOT / "config" / "default.yaml"))

    # 强制关闭对抗模块
    config.setdefault("adversarial", {})["enabled"] = False
    logger.info("对抗模块: 已关闭")

    # 加载数据集
    bench = ResearchBench()
    questions = bench.get_questions(n=5)
    logger.info("共 %d 道评测题", len(questions))
    for q in questions:
        logger.info("  [%s] %s", q["id"], q["query"][:60])

    # 初始化持久化存储
    memory_cfg = config.get("memory", {})
    session_memory = SessionMemory(
        db_path=memory_cfg.get("session_db_path", "data/session_memory.db"),
    )
    knowledge_base = KnowledgeBase(
        db_path=memory_cfg.get("knowledge_db_path", "data/knowledge_base.db"),
        embedder=MemoryEmbedder(),
        config=memory_cfg,
    )
    await session_memory.initialize()
    await knowledge_base.initialize()

    results = []
    try:
        for i, q in enumerate(questions, 1):
            logger.info("\n" + "=" * 60)
            logger.info("[%d/%d] 开始研究: %s", i, len(questions), q["id"])
            logger.info("问题: %s", q["query"][:80])
            logger.info("=" * 60)

            state, report_text, report_obj = await run_single_query(
                q["query"], config, session_memory, knowledge_base,
            )

            logger.info("状态: %s", state.value)
            logger.info("报告长度: %d 字符", len(report_text))

            # 从 report 对象提取来源数
            num_sources = len(getattr(report_obj, "sources", [])) if report_obj else 0
            logger.info("来源数: %d", num_sources)

            # 评测
            eval_result = bench.evaluate_report(report_text, q["id"], num_sources=num_sources)

            results.append({
                "question_id": q["id"],
                "domain": q["domain"],
                "query": q["query"],
                "state": state.value,
                "report_length": len(report_text),
                "num_sources": num_sources,
                "report": report_text[:3000],  # 截断保存前3000字符
                "eval": eval_result,
            })

            logger.info("综合得分: %.3f", eval_result["composite_score"])
            logger.info(
                "  factual=%.3f, halluc=%.3f, sources=%.3f, logic=%.3f, comp=%.3f",
                eval_result["metrics"]["factual_accuracy"],
                eval_result["hallucination_rate"],
                eval_result["metrics"]["source_adequacy"],
                eval_result["metrics"]["logical_consistency"],
                eval_result["metrics"]["comprehensiveness"],
            )

    finally:
        await session_memory.close()
        await knowledge_base.close()

    # -----------------------------------------------------------------------
    # 汇总
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 评测汇总")
    print("=" * 60)

    scores = [r["eval"]["composite_score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"题目数:     {len(results)}")
    print(f"平均综合分: {avg_score:.3f}")
    print()
    for r in results:
        s = r["eval"]["composite_score"]
        bar = "█" * int(s * 20)
        print(f"  [{r['question_id']}] ({r['domain']}) {s:.3f} {bar}")

    # 按领域汇总
    from collections import defaultdict
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r["eval"]["composite_score"])
    print()
    for domain, scores_list in sorted(by_domain.items()):
        avg = sum(scores_list) / len(scores_list)
        print(f"  {domain}: avg={avg:.3f} ({len(scores_list)}题)")

    # 保存结果
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bench_result_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "num_questions": len(questions),
        "average_composite": avg_score,
        "adversarial_enabled": False,
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
