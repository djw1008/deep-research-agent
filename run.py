#!/usr/bin/env python3
"""DeepResearchAgent 入口脚本。

支持两种模式：
1. 非交互模式：python run.py --query "你的研究问题"
2. 交互模式：python run.py

交互模式下，完成一轮研究后可选择"继续当前研究"或"开启新研究"。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# 允许在未安装包的情况下直接运行
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
from deep_research.memory import (
    KnowledgeBase,
    SessionMemory,
)
from deep_research.memory.embedder import MemoryEmbedder
from deep_research.models import ModelRouter
from deep_research.observability import RunEventRecorder
from deep_research.planner import Planner
from deep_research.tools import (
    ArxivReaderTool,
    BrowserBatchTool,
    BrowserTool,
    CalculatorTool,
    CodeSandboxTool,
    FileReaderTool,
    NotepadTool,
    ToolRegistry,
    WebSearchTool,
)


def setup_logging(level: str = "INFO") -> None:
    """配置日志输出。第三方库的日志级别自动提高，避免干扰。"""
    user_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=user_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 降低第三方库的日志噪音
    for noisy in ("openai", "httpx", "httpcore", "primp", "urllib3", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(config_path: str | None) -> dict:
    """加载 YAML 配置文件，若不存在则返回空字典。"""
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        logging.warning("配置文件不存在: %s", config_path)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_tool_registry(_config: dict) -> ToolRegistry:
    """创建并注册所有可用工具。"""
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    browser = BrowserTool()
    registry.register(browser)
    max_batch_urls = int(_config.get("researcher", {}).get("max_browser_urls_per_round", 3))
    registry.register(BrowserBatchTool(browser, max_urls=max_batch_urls))
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
    event_sink=None,
) -> Orchestrator:
    """根据配置构建完整的 Orchestrator。"""
    # 1. 读取模型后端配置
    model_cfg = config.get("model", {})
    backend = model_cfg.get("backend", "deepseek")

    # 2. LLMClient 参数（优先从环境变量读取，YAML 显式配置作为 fallback）
    client_kwargs: dict = {}
    if "base_model" in model_cfg:
        client_kwargs["model_name"] = model_cfg["base_model"]
    if "base_url" in model_cfg:
        client_kwargs["base_url"] = model_cfg["base_url"]
    if "api_key" in model_cfg:
        client_kwargs["api_key"] = model_cfg["api_key"]
    # 采样参数（Orchestrator + Agent policy 共用）
    for key in ("temperature", "top_p", "max_tokens"):
        if key in model_cfg:
            client_kwargs[key] = model_cfg[key]

    try:
        llm_client = ModelRouter.create_backend(backend, **client_kwargs)
        logging.info("LLM 后端已初始化: %s", backend)
    except ValueError as e:
        logging.error(
            "LLM 后端初始化失败: %s\n"
            "请确保以下任一方式已配置:\n"
            "  1. 环境变量: DEEPSEEK_API_KEY=your_key\n"
            "  2. YAML 配置: config/default.yaml -> model.api_key\n",
            e,
        )
        raise SystemExit(1)

    # 3. 创建 Planner
    planner = Planner(llm_client)

    # 4. 创建 AgentPool（按模块分配合适的 sampling 参数）
    registry = create_tool_registry(config)

    # 模块名 → {backend, temperature, max_tokens, ...}
    sampling_cfg = config.get("model", {}).get("backend_sampling", {})
    backend_defaults = sampling_cfg.get(backend, {})      # 后端全局默认
    module_overrides = sampling_cfg.get("modules", {})     # 模块级覆盖
    backend_map = config.get("model", {}).get("backend_mapping", {})

    # task_type → 模块名
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
        # 合并优先级：后端全局默认 < client_kwargs < 模块覆盖
        # 模块级配置（如 summarizer.max_tokens）优先级最高
        merged = dict(backend_defaults)
        merged.update(client_kwargs)
        if module in module_overrides:
            merged.update(module_overrides[module])
        # 模块有独立后端时优先
        mod_backend = backend_map.get(module, backend)
        return ModelRouter.create_backend(mod_backend, **merged)

    def tools_factory():
        # 返回工具实例列表（ResearchAgent 会注入）
        return list(registry._tools.values())

    orch_cfg = config.get("orchestrator", {})

    agent_pool = AgentPool(
        policy_factory=policy_factory,
        tools_factory=tools_factory,
        max_idle=orch_cfg.get("max_concurrent", 5),
        config=config,
        session_memory=session_memory,
        knowledge_base=knowledge_base,
        event_sink=event_sink,
    )

    # 7. 可选：IssueMerger 的 LLM 仲裁客户端
    issue_arbiter_client = None
    arbiter_cfg = config.get("adversarial", {}).get("issue_arbiter", {})
    if arbiter_cfg.get("enabled", True):
        try:
            issue_arbiter_client = policy_factory("issue_arbiter")
            logging.info("Issue 仲裁 LLM 后端已初始化")
        except Exception:
            logging.exception("Issue 仲裁 LLM 初始化失败，将使用 rule-based 合并")

    # 9. 创建 Orchestrator
    return Orchestrator(
        planner=planner,
        agent_pool=agent_pool,
        session_memory=session_memory,
        knowledge_base=knowledge_base,
        config=config,
        issue_arbiter_client=issue_arbiter_client,
        event_sink=event_sink,
    )


async def run_research(
    query: str,
    config: dict,
    timeout_seconds: int | None = 300,
    session_id: str | None = None,
    round: int = 1,
    previous_session_context: str = "",
    session_memory: SessionMemory | None = None,
    knowledge_base: KnowledgeBase | None = None,
    event_recorder: RunEventRecorder | None = None,
    adversarial: bool = False,
) -> tuple:
    """执行单次研究并返回最终状态和报告。"""
    recorder = event_recorder or RunEventRecorder(query)
    orch = None
    try:
        orch = build_orchestrator(
            config,
            session_memory=session_memory or SessionMemory("data/session_memory.db"),
            knowledge_base=knowledge_base or KnowledgeBase("data/knowledge_base.db"),
            event_sink=recorder,
        )
        if orch.knowledge_base is not None:
            await orch.knowledge_base.initialize()
        if orch.session_memory is not None:
            await orch.session_memory.initialize()

        context = ResearchContext(
            topic=query,
            max_iterations=config.get("orchestrator", {}).get("max_replan_rounds", 3),
            enable_adversarial=adversarial or config.get("adversarial", {}).get("enabled", False),
        )
        final_state = await orch.run(
            context,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            round=round,
            previous_session_context=previous_session_context,
        )
        report = getattr(orch, "_report", None)
        recorder.emit("run_completed", {
            "state": final_state.value,
            "confidence": getattr(report, "confidence", 0.0) if report else 0.0,
            "sources": len(getattr(report, "sources", [])) if report else 0,
        })
        return final_state, report, orch
    except Exception as exc:
        recorder.emit("run_failed", {
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        if orch is not None and orch.knowledge_base is not None:
            await orch.knowledge_base.close()
        if orch is not None and orch.session_memory is not None:
            await orch.session_memory.close()


def format_dag(dag_dict: dict) -> str:
    """Format a serialized DAG dict as a concise human-readable string."""
    tasks = dag_dict.get("tasks", {})
    edges = dag_dict.get("edges", {})
    nodes = dag_dict.get("nodes", [])

    if tasks:
        lines = ["子任务:"]
        for task_id in sorted(tasks.keys()):
            task = tasks[task_id]
            ttype = task.get("task_type", "search")
            desc = task.get("description", "")
            lines.append(f"  - {task_id} [{ttype}] {desc}")
        if edges:
            lines.append("依赖:")
            for src, dsts in edges.items():
                if dsts:
                    lines.append(f"  {src} -> {', '.join(dsts)}")
        return "\n".join(lines)

    # 兼容旧数据：只有 nodes/edges 时没有 description
    if not nodes:
        return "（无计划）"
    lines = [f"节点: {', '.join(nodes)}"]
    if edges:
        lines.append("依赖:")
        for src, dsts in edges.items():
            if dsts:
                lines.append(f"  {src} -> {', '.join(dsts)}")
    return "\n".join(lines)


def build_continuation_context(
    history: list,
    max_chars_per_report: int = 2000,
) -> str:
    """Build the context text for a continuation plan from session history."""
    if not history:
        return ""

    parts = ["以下是目前会话的多轮研究历史："]
    for entry in history:
        report_snippet = entry.report[:max_chars_per_report]
        if len(entry.report) > max_chars_per_report:
            report_snippet += "\n...（已截断）"
        parts.append(
            f"\n### 第 {entry.round} 轮\n"
            f"用户问题：{entry.query}\n"
            f"研究计划：\n{format_dag(entry.dag)}\n"
            f"研究报告：\n{report_snippet}"
        )
    return "\n\n".join(parts)


async def prompt_session_selection(session_memory: SessionMemory) -> tuple[str | None, str]:
    """展示已有 session 供用户选择，返回 (session_id, query)。

    如果用户选择新建研究，返回 (None, new_query)。
    如果用户选择已有 session，返回 (selected_session_id, follow_up_query)。
    """
    sessions = await session_memory.list_sessions()

    print("\n" + "=" * 60)
    print(" DeepResearchAgent 会话选择")
    print("=" * 60)
    print("  [0] 新建研究")
    for i, s in enumerate(sessions, 1):
        topic = s.get("topic", "")[:40]
        print(f"  [{i}] {topic}  (session: {s['session_id'][:8]}..., {s['count']} 轮)")
    print("=" * 60)

    while True:
        choice = input("请选择会话编号: ").strip()
        if choice == "0":
            query = input("请输入新的研究问题: ").strip()
            if not query:
                print("问题不能为空，请重新选择。")
                continue
            return None, query

        try:
            idx = int(choice)
            if idx < 1 or idx > len(sessions):
                print("编号无效，请重新输入。")
                continue
        except ValueError:
            print("请输入数字编号。")
            continue

        selected = sessions[idx - 1]
        session_id = selected["session_id"]
        print(f"\n已选择会话: {selected['topic'][:40]}")
        follow_up = input("请输入后续问题（直接回车使用原问题）: ").strip()
        query = follow_up or selected["topic"]
        return session_id, query


def ask_continue_or_new() -> str:
    """Ask the user whether to continue the current research or start a new one."""
    print("\n" + "=" * 60)
    print(" 研究完成。请选择：")
    print("  [1] 继续当前研究")
    print("  [2] 开启新的研究")
    print("  [3] 退出")
    print("=" * 60)
    while True:
        choice = input("请输入选项 (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("无效输入，请重新输入。")


def save_report(query: str, report, config: dict) -> Path | None:
    """将报告保存到工作目录。"""
    if report is None:
        return None

    work_dir = Path(config.get("system", {}).get("work_dir", "./outputs"))
    work_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:30])
    filename = work_dir / f"report_{safe_query}_{timestamp}.md"

    content_lines = [
        f"# 研究报告: {query}",
        "",
        f"- **生成时间**: {datetime.now().isoformat()}",
        f"- **整体置信度**: {getattr(report, 'confidence', 'N/A')}",
        f"- **引用来源数**: {len(getattr(report, 'sources', []))}",
        f"- **搜索次数**: {getattr(report, 'num_searches', 'N/A')}",
    ]
    # 对抗评分只有 Red/Blue 阶段实际运行后才会填充；
    # 未启用时直接写"未启用"，避免输出 0.0 / {} 这类占位默认值
    dimension_scores = getattr(report, "dimension_scores", None) or {}
    adversarial_rounds = getattr(report, "adversarial_rounds", 0) or 0
    if adversarial_rounds or dimension_scores:
        dimension_scores_text = ", ".join(
            f"{getattr(dim, 'value', dim)}: {score}"
            for dim, score in dimension_scores.items()
        )
        content_lines += [
            f"- **对抗轮数**: {adversarial_rounds}",
            f"- **最终评分**: {getattr(report, 'final_score', 'N/A')}",
            f"- **五维评分**: {dimension_scores_text}",
        ]
    else:
        content_lines.append("- **对抗优化**: 未启用")
    content_lines += [
        "",
        "---",
        "",
        getattr(report, "content", "(无内容)"),
        "",
        "---",
        "",
        "## 参考链接",
        "",
    ]
    for src in getattr(report, "sources", []):
        title = src.get("title", "")
        url = src.get("url", "")
        content_lines.append(f"- [{title}]({url})")

    content = "\n".join(content_lines)
    filename.write_text(content, encoding="utf-8")

    # Dashboard 启动的 run：额外写一份到 debug_runs/<run_id>/report.md 供报告接口读取
    run_id = os.environ.get("DEEP_RESEARCH_RUN_ID")
    if run_id:
        debug_report = work_dir / "debug_runs" / run_id / "report.md"
        if debug_report.parent.is_dir():
            debug_report.write_text(content, encoding="utf-8")
    return filename


def parse_report_markdown(report_path: Path):
    """从 save_report 生成的 report.md 重建 ResearchReport。"""
    from deep_research.core.schema import ResearchReport

    lines = report_path.read_text(encoding="utf-8").splitlines()
    query = ""
    confidence = 0.5
    if lines and lines[0].startswith("# 研究报告: "):
        query = lines[0][len("# 研究报告: "):].strip()
    for line in lines:
        if line.startswith("- **整体置信度**:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.5
            break

    # 正文：第一条 --- 之后、最后一条 ---（## 参考链接 之前）之间
    separators = [i for i, line in enumerate(lines) if line.strip() == "---"]
    content = ""
    if len(separators) >= 2:
        content = "\n".join(lines[separators[0] + 1:separators[-1]]).strip()

    link_re = re.compile(r"^- \[(?P<title>[^\]]*)\]\((?P<url>[^)]*)\)\s*$")
    sources = [
        {"title": match.group("title"), "url": match.group("url")}
        for match in (link_re.match(line.strip()) for line in lines)
        if match
    ]
    return ResearchReport(query=query, content=content, confidence=confidence, sources=sources)


async def upgrade_adversarial_run(
    run_id: str,
    config: dict,
    session_memory: SessionMemory,
    knowledge_base: KnowledgeBase,
) -> int:
    """对已完成的报告只跑 Red/Blue 对抗阶段（--upgrade-adversarial 模式）。"""
    work_dir = Path(config.get("system", {}).get("work_dir", "./outputs"))
    report_path = work_dir / "debug_runs" / run_id / "report.md"
    if not report_path.exists():
        logging.error("未找到报告文件: %s", report_path)
        return 1

    report = parse_report_markdown(report_path)
    query = report.query or run_id
    recorder = RunEventRecorder(query)
    recorder.emit("state_transition", {"from": "done", "to": "adversarial"})

    orch = None
    try:
        orch = build_orchestrator(
            config,
            session_memory=session_memory,
            knowledge_base=knowledge_base,
            event_sink=recorder,
        )
        upgraded = await orch.run_adversarial_upgrade(report, session_id=f"upgrade-{run_id}")
        recorder.emit("run_completed", {
            "state": "done",
            "confidence": getattr(upgraded, "confidence", 0.0),
            "sources": len(getattr(upgraded, "sources", [])),
        })
        saved = save_report(query, upgraded, config)
        if saved:
            logging.info("对抗升级报告已保存: %s", saved)
        return 0
    except Exception as exc:
        logging.exception("对抗升级失败")
        recorder.emit("run_failed", {
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        })
        return 1
    finally:
        if orch is not None and orch.knowledge_base is not None:
            await orch.knowledge_base.close()
        if orch is not None and orch.session_memory is not None:
            await orch.session_memory.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description="DeepResearchAgent 会话式研究入口")
    parser.add_argument("-q", "--query", help="研究问题（非交互模式下使用）")
    parser.add_argument("-c", "--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--adversarial", action="store_true", help="启用 Red/Blue 对抗优化")
    parser.add_argument("--upgrade-adversarial", metavar="RUN_ID", help="对已完成 run 的报告只运行 Red/Blue 对抗优化阶段")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)

    if config:
        logging.info("已加载配置: %s", args.config)
    else:
        logging.info("未找到配置文件，使用默认参数")

    memory_cfg = config.get("memory", {})
    session_memory = SessionMemory(
        db_path=memory_cfg.get("session_db_path", "data/session_memory.db"),
    )
    knowledge_base = KnowledgeBase(
        db_path=memory_cfg.get("knowledge_db_path", "data/knowledge_base.db"),
        embedder=MemoryEmbedder(),
        config=memory_cfg,
    )

    try:
        await session_memory.initialize()
        await knowledge_base.initialize()

        if args.upgrade_adversarial:
            return await upgrade_adversarial_run(
                args.upgrade_adversarial, config, session_memory, knowledge_base,
            )

        if args.query:
            session_id: str | None = None
            query = args.query
        else:
            session_id, query = await prompt_session_selection(session_memory)

        orch_cfg = config.get("orchestrator", {})
        global_timeout = orch_cfg.get("global_timeout_seconds", 300)
        if global_timeout is None:
            logging.info("全局超时已关闭")

        logging.info("研究问题: %s", query)
        if session_id:
            logging.info("复用会话: %s", session_id)

        current_round = 1
        current_session_id = session_id
        current_query = query

        # 如果从入口选择了已有会话，初始 round 应为已有轮数+1，
        # 这样第一轮就会触发 continuation prompt 并加载历史上下文。
        if current_session_id is not None:
            history = await session_memory.get_session_history(current_session_id)
            current_round = len(history) + 1

        while True:
            previous_context = ""
            if current_session_id is not None and current_round > 1:
                history = await session_memory.get_session_history(current_session_id)
                previous_context = build_continuation_context(history)

            final_state, report, _orch = await run_research(
                query=current_query,
                config=config,
                timeout_seconds=global_timeout,
                session_id=current_session_id,
                round=current_round,
                previous_session_context=previous_context,
                session_memory=session_memory,
                knowledge_base=knowledge_base,
                adversarial=args.adversarial,
            )

            # Ensure session_id is stable for the next round
            if current_session_id is None:
                current_session_id = getattr(_orch, "_session_id", None)

            print(f"\n{'='*60}")
            print(f"最终状态: {final_state.value}")
            print(f"{'='*60}")

            if report is not None:
                print(f"\n报告置信度: {getattr(report, 'confidence', 'N/A')}")
                print(f"引用来源: {len(getattr(report, 'sources', []))} 条")
                print(f"搜索次数: {getattr(report, 'num_searches', 'N/A')}")

                saved = save_report(current_query, report, config)
                if saved:
                    print(f"\n报告已保存: {saved}")
            else:
                print("\n未生成报告（流程可能提前终止或失败）。")

            # Non-interactive mode exits after one round
            if args.query:
                return 0 if final_state.value == "done" else 1

            choice = ask_continue_or_new()
            if choice == "3":
                return 0 if final_state.value == "done" else 1
            if choice == "2":
                current_session_id = None
                current_round = 1
                current_query = input("请输入新的研究问题: ").strip()
                if not current_query:
                    print("问题不能为空，退出。")
                    return 1
                continue

            # choice == "1": continue current research
            current_round += 1
            follow_up = input("请输入继续研究的方向: ").strip()
            current_query = follow_up or current_query

    except SystemExit:
        return 1
    except Exception:
        logging.exception("研究流程异常终止")
        return 1
    finally:
        await session_memory.close()
        await knowledge_base.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
