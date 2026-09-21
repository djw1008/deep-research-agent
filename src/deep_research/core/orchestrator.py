"""Orchestrator — sole controller of the workflow state machine."""

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..memory.knowledge_store import KnowledgeBase
from ..memory.models import KnowledgeEntry, SessionMemoryEntry
from ..memory.session_store import SessionMemory
from .schema import (
    AttackDimension,
    DimensionAttack,
    FixType,
    HandlerResult,
    Issue,
    RedAttackResult,
    ResearchContext,
    ResearchReport,
    Severity,
    WorkflowState,
)
from .issue_merger import IssueMerger
from .workflow import InvalidTransitionError, NoHandlerRegisteredError, StateTransition, TRANSITIONS

logger = logging.getLogger(__name__)

# Type alias for a state handler
StateHandler = Callable[["Orchestrator", ResearchContext], Any]


class Orchestrator:
    """
    Sole controller of the research workflow.

    Owns the main loop, state transition rules, handler registry and
    concurrent task tracking.  Handlers only return HandlerResult;
    Orchestrator decides where to go next.

    调度器直接控制所有任务：
      - __init__ 注入 planner + agent_pool
      - _do_dispatching 内部创建 Semaphore，按 DAG 层调度，直接 get/release Agent
      - 状态处理器收拢为 Orchestrator 内部方法
    """

    def __init__(
        self,
        planner=None,
        agent_pool=None,
        session_memory: SessionMemory | None = None,
        knowledge_base: KnowledgeBase | None = None,
        config: dict[str, Any] | None = None,
        issue_arbiter_client=None,
        event_sink=None,
    ) -> None:
        # 核心依赖（外部注入，便于测试时 Mock）
        self.planner = planner
        self.agent_pool = agent_pool
        self.session_memory = session_memory
        self.knowledge_base = knowledge_base
        self.config = config or {}
        # 可选：IssueMerger 的 LLM 仲裁客户端
        self.issue_arbiter_client = issue_arbiter_client
        self.event_sink = event_sink

        # 状态机状态
        self._state = WorkflowState.IDLE
        self._history: List[StateTransition] = []
        self._handlers: Dict[WorkflowState, StateHandler] = {}
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

        # 运行时状态（单次 run 生命周期）
        self._results: list = []
        self._dag = None
        self._task_map: dict = {}
        self._memory_store: dict = {}
        self._query: str = ""
        self._session_id: str = ""
        self._round: int = 1
        self._replan_count: int = 0
        self._report: Any = None
        self._context: Optional[ResearchContext] = None
        self._previous_session_context: str = ""

        # 写入知识库的置信度阈值
        self._knowledge_confidence_threshold: float = float(
            self.config.get("memory", {}).get("min_confidence", 0.6)
        )

        # 状态处理器映射（默认内部方法）
        self._state_handlers = {
            WorkflowState.PLANNING: self._do_planning,
            WorkflowState.DISPATCHING: self._do_dispatching,
            WorkflowState.COLLECTING: self._do_collecting,
            WorkflowState.SYNTHESIZING: self._do_synthesizing,
            WorkflowState.ADVERSARIAL: self._do_adversarial,
            WorkflowState.REPLANNING: self._do_replanning,
            WorkflowState.DONE: self._on_done,
            WorkflowState.FAILED: self._on_failed,
        }

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def state(self) -> WorkflowState:
        return self._state

    @property
    def history(self) -> List[StateTransition]:
        return self._history.copy()

    @property
    def is_terminal(self) -> bool:
        return self._state in (WorkflowState.DONE, WorkflowState.FAILED)

    # ------------------------------------------------------------------
    # State-machine guards
    # ------------------------------------------------------------------
    def can_transition(self, target: WorkflowState) -> bool:
        return target in TRANSITIONS.get(self._state, set())

    def get_allowed_transitions(self) -> Set[WorkflowState]:
        return TRANSITIONS.get(self._state, set()).copy()

    # ------------------------------------------------------------------
    # Core transition (atomic, with lock)
    # ------------------------------------------------------------------
    async def transition(self, target: WorkflowState, context: Optional[ResearchContext] = None) -> None:
        async with self._lock:
            if not self.can_transition(target):
                raise InvalidTransitionError(
                    f"Cannot transition from {self._state.value} to {target.value}"
                )

            old = self._state
            self._state = target
            iter_count = getattr(context, "iteration_count", 0) if context else 0
            self._history.append(
                StateTransition(
                    from_state=old,
                    to_state=target,
                    context={"iteration": iter_count},
                )
            )
            logger.info("状态转换: %s -> %s", old.value, target.value)
            self._emit("state_transition", {
                "from": old.value, "to": target.value, "iteration": iter_count,
            })

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish a dashboard event without making observability mission-critical."""
        if self.event_sink is None:
            return
        try:
            self.event_sink(event_type, payload)
        except Exception:
            logger.exception("Dashboard event recording failed: %s", event_type)

    def _emit_agent_result(self, task, result) -> None:
        """Mirror a non-research agent's compact trajectory and final state."""
        for step in result.trajectory or []:
            self._emit("agent_loop_event", {"task_id": task.id, "event": step})
        self._emit("task_completed", {
            "task_id": task.id,
            "status": result.status.value,
            "confidence": result.confidence,
            "token_usage": result.token_usage,
            "output": result.output,
            "metadata": result.metadata,
        })

    # ------------------------------------------------------------------
    # Handler registry (for extension / custom override)
    # ------------------------------------------------------------------
    def on_state(self, state: WorkflowState, handler: StateHandler) -> None:
        """注册自定义 handler 覆盖默认行为。"""
        self._handlers[state] = handler

    # ------------------------------------------------------------------
    # Handler execution (for extension mode)
    # ------------------------------------------------------------------
    async def execute_handler(self, state: WorkflowState, context: ResearchContext) -> HandlerResult:
        handler = self._handlers.get(state)
        if handler is None:
            raise NoHandlerRegisteredError(f"No handler registered for state: {state.value}")
        return await handler(self, context)

    # ------------------------------------------------------------------
    # Concurrency helpers
    # ------------------------------------------------------------------
    def spawn_task(self, name: str, coro: Any) -> asyncio.Task:
        task: asyncio.Task = asyncio.create_task(coro, name=name)
        self._pending_tasks[name] = task
        logger.debug("启动任务 %s", name)
        return task

    async def wait_pending_tasks(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for name, task in list(self._pending_tasks.items()):
            try:
                results[name] = await task
            except Exception as exc:
                results[name] = exc
                logger.error("任务 %s 失败: %s", name, exc)
        self._pending_tasks.clear()
        return results

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(
        self,
        context: ResearchContext,
        timeout_seconds: int | None = 300,
        session_id: str | None = None,
        round: int = 1,
        previous_session_context: str = "",
    ) -> WorkflowState:
        """
        Drive the workflow from IDLE to a terminal state.

        Args:
            context: 研究上下文
            timeout_seconds: 全局超时（秒），默认 300。设为 None 或 null 则关闭超时。
                超过后强制转入 FAILED。
            session_id: 复用的会话 ID；为 None 时自动生成新的 UUID。
            round: 当前会话的第几轮研究，从 1 开始。
            previous_session_context: 继续研究时传入的历史上下文文本。

        两种方式执行状态逻辑：
          1. 如果注册了自定义 handler（on_state），优先调用
          2. 否则使用 Orchestrator 内置的 _state_handlers
        """
        if self._state != WorkflowState.IDLE:
            raise RuntimeError(
                f"Workflow must start from IDLE, currently {self._state.value}"
            )

        self._round = round
        self._previous_session_context = previous_session_context

        if timeout_seconds is None:
            return await self._run_loop(context, session_id=session_id)

        try:
            return await asyncio.wait_for(
                self._run_loop(context, session_id=session_id), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error("全局超时: 研究流程超过 %s 秒，强制终止", timeout_seconds)
            if self.can_transition(WorkflowState.FAILED):
                await self.transition(WorkflowState.FAILED, context)
            return self._state

    async def _run_loop(self, context: ResearchContext, session_id: str | None = None) -> WorkflowState:
        """核心状态循环（不含超时包装）。"""
        self._context = context
        self._query = context.topic
        self._session_id = session_id or str(uuid.uuid4())
        await self.transition(WorkflowState.PLANNING, context)

        while not self.is_terminal:
            try:
                # 优先使用用户注册的自定义 handler
                if self._state in self._handlers:
                    result = await self.execute_handler(self._state, context)
                    next_state = self._decide_next_state(self._state, result, context)
                    if next_state is None:
                        logger.info("编排器在 %s 暂停", self._state.value)
                        break
                    await self.transition(next_state, context)
                else:
                    # 使用内置状态处理器
                    handler = self._state_handlers.get(self._state)
                    if handler is None:
                        raise RuntimeError(f"No handler for state: {self._state.value}")
                    next_state = await handler()
                    if next_state is not None and next_state != self._state:
                        await self.transition(next_state, context)
            except Exception:
                logger.exception("状态 %s 处理时发生异常", self._state.value)
                if self.can_transition(WorkflowState.FAILED):
                    await self.transition(WorkflowState.FAILED, context)
                break

        # Execute terminal handlers so cleanup (e.g. Session Memory persistence) runs.
        if self._state == WorkflowState.DONE:
            try:
                await self._on_done()
            except Exception:
                logger.exception("终端状态 DONE 处理异常")
        elif self._state == WorkflowState.FAILED:
            try:
                await self._on_failed()
            except Exception:
                logger.exception("终端状态 FAILED 处理异常")

        return self._state

    # ------------------------------------------------------------------
    # 内置状态处理器（调度器直接控制）
    # ------------------------------------------------------------------
    async def _do_planning(self) -> WorkflowState:
        """PLANNING：召回相关记忆并调用 Planner 生成 DAG。"""
        if self.planner is None:
            raise RuntimeError("Planner not configured")
        try:
            memory_context = ""
            continuation_context = self._previous_session_context

            if self._round > 1:
                # Continue research: use continuation prompt even if no history was loaded.
                if not continuation_context:
                    continuation_context = "（用户选择继续当前研究，但未找到历史研究记录。请基于当前问题继续规划。）"
                dag, subtasks = await self.planner.generate_continuation_plan(
                    self._query,
                    continuation_context=continuation_context,
                    memory_context=memory_context,
                )
            else:
                dag, subtasks = await self.planner.generate_plan(
                    self._query, memory_context=memory_context
                )

            self._dag = dag
            self._task_map = {st.id: st for st in subtasks}
            self._emit("dag_created", {
                "dag": dag.to_dict(),
                "layers": dag.get_parallel_groups(),
                "tasks": [st.model_dump() for st in subtasks],
            })
            logger.info("规划完成: %d 个子任务, %d 层", len(dag), len(dag.get_parallel_groups()))
            logger.debug("\n%s", dag.to_ascii())
        except Exception as e:
            logger.exception("规划失败")
            return WorkflowState.FAILED
        return WorkflowState.DISPATCHING

    async def _do_dispatching(self) -> WorkflowState:
        """DISPATCHING：拓扑排序 + 并发调度子任务。

        调度器直接控制：
          1. 创建 Semaphore 限制并发
          2. 按 DAG 层遍历，层内并发
          3. 从 AgentPool get_agent / release_agent
          4. 单任务超时保护
        """
        if self.agent_pool is None:
            raise RuntimeError("AgentPool not configured")
        if self._dag is None or len(self._dag) == 0:
            return WorkflowState.COLLECTING

        semaphore = asyncio.Semaphore(3)  # 默认并发数 3
        parallel_groups = self._dag.get_parallel_groups()
        all_results = []

        # 调度器内部维护 agent_results：每层结束后写入，下一层开始前读取上游依赖
        agent_results: dict[str, AgentResult] = {}

        for layer_idx, group in enumerate(parallel_groups):
            logger.info("调度层 %d/%d: %s", layer_idx + 1, len(parallel_groups), group)

            async def _run_one(task_id: str):
                async with semaphore:
                    subtask = self._task_map.get(task_id)
                    if subtask is None:
                        from ..core.schema import AgentResult, AgentStatus
                        return AgentResult(task_id=task_id, status=AgentStatus.FAILED, output=f"Task '{task_id}' not found")

                    # 根据 DAG 依赖从 agent_results 构造当前任务的上下文
                    ctx: dict[str, Any] = {"query": self._query}
                    for dep_id in subtask.dependencies:
                        dep_result = agent_results.get(dep_id)
                        if dep_result is not None:
                            output = dep_result.output
                            ctx[f"dep:{dep_id}"] = output if isinstance(output, str) else str(output)

                    # 从 AgentPool 借 Agent
                    self._emit("task_started", {
                        "task_id": task_id,
                        "task_type": subtask.task_type,
                        "description": subtask.description,
                        "dependencies": subtask.dependencies,
                        "layer": layer_idx,
                    })
                    agent = await self.agent_pool.get_agent(subtask.task_type)
                    configured_timeout = int(
                        self.config.get("researcher", {}).get("subtask_timeout_seconds", 360)
                    )
                    effective_timeout = max(subtask.timeout_seconds, configured_timeout)
                    try:
                        result = await asyncio.wait_for(
                            agent.run(subtask, ctx),
                            timeout=effective_timeout,
                        )
                    except asyncio.TimeoutError:
                        from ..core.schema import AgentResult, AgentStatus
                        result = AgentResult(
                            task_id=task_id, status=AgentStatus.TIMEOUT,
                            output=f"Task timed out after {effective_timeout}s",
                        )
                    except Exception as e:
                        from ..core.schema import AgentResult, AgentStatus
                        result = AgentResult(
                            task_id=task_id, status=AgentStatus.FAILED,
                            output=f"Exception: {type(e).__name__}: {e}",
                        )
                    finally:
                        await self.agent_pool.release_agent(agent)

                    # 把本层结果写入 agent_results，供下游任务使用
                    agent_results[task_id] = result
                    self._emit("task_completed", {
                        "task_id": task_id,
                        "status": result.status.value,
                        "confidence": result.confidence,
                        "token_usage": result.token_usage,
                        "output": result.output,
                        "metadata": result.metadata,
                    })
                    return result

            coros = [_run_one(tid) for tid in group]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)

            for lr in layer_results:
                if isinstance(lr, Exception):
                    from ..core.schema import AgentResult, AgentStatus
                    all_results.append(AgentResult(task_id="unknown", status=AgentStatus.FAILED, output=f"Dispatch exception: {lr}"))
                else:
                    all_results.append(lr)

        self._results = all_results
        return WorkflowState.COLLECTING

    async def _do_collecting(self) -> WorkflowState:
        """COLLECTING：收集结果、写入记忆、导出 trajectory、判断是否重规划。"""
        from ..core.schema import AgentStatus

        # 保存结果
        for r in self._results:
            self._memory_store[f"result:{r.task_id}"] = r

        # 将高置信度、非召回得到的成功子任务结果写入知识库
        if self.knowledge_base is not None:
            for r in self._results:
                if (
                    r.status == AgentStatus.SUCCESS
                    and r.confidence >= self._knowledge_confidence_threshold
                    and not r.metadata.get("from_memory")
                ):
                    await self._persist_to_knowledge_base(r)

        # 导出每个子任务的 trajectory
        self._export_trajectories()

        success_count = sum(1 for r in self._results if r.status == AgentStatus.SUCCESS)
        total_count = len(self._results)
        fail_count = total_count - success_count
        logger.info("收集完成: %d/%d 成功 (%d 失败)", success_count, total_count, fail_count)

        # 打印每个失败任务的详情
        for r in self._results:
            if r.status != AgentStatus.SUCCESS:
                output = r.output if isinstance(r.output, str) else str(r.output)
                logger.warning(
                    "  失败任务 [%s] 状态=%s 原因=%s",
                    r.task_id, r.status.value, output[:200]
                )

        # 判断是否重规划
        if self._should_replan(self._results):
            self._replan_count += 1
            logger.info("触发重规划 (第 %d 次)", self._replan_count)
            return WorkflowState.REPLANNING

        return WorkflowState.SYNTHESIZING

    def _export_trajectories(self) -> None:
        """将所有子任务的 trajectory 导出为 JSON 文件。"""
        import json
        from datetime import datetime
        from pathlib import Path

        if not self._results:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in self._query[:20])
        out_dir = Path(f"outputs/trajectories_{safe_query}_{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for r in self._results:
            meta = r.metadata or {}
            if not meta:
                st = self._task_map.get(r.task_id)
                meta = {
                    "from_memory": False,
                    "task_description": st.description if st else None,
                }
            data = {
                "task_id": r.task_id,
                "status": r.status.value,
                "confidence": r.confidence,
                "token_usage": r.token_usage,
                "output": r.output if isinstance(r.output, str) else str(r.output)[:500],
                "trajectory": r.trajectory,
                "metadata": meta,
            }
            file_path = out_dir / f"{r.task_id}.json"
            file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Trajectory 已导出: %s (%d 个任务)", out_dir, len(self._results))

    async def _persist_to_knowledge_base(self, result: "AgentResult") -> None:
        """把单个非召回的成功子任务结果写入全局知识库。"""
        if self.knowledge_base is None:
            return

        subtask = self._task_map.get(result.task_id)
        content = result.output if isinstance(result.output, str) else str(result.output)
        task_type = subtask.task_type if subtask else "search"
        task_description = subtask.description if subtask else ""

        # 从 trajectory 中提取引用来源
        sources: list[dict] = []
        for step in result.trajectory:
            if step.get("role") != "tool":
                continue
            res = step.get("result")
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict) and "url" in item:
                        sources.append({
                            "url": item["url"],
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "task_id": result.task_id,
                        })
            elif isinstance(res, dict):
                if "results" in res and isinstance(res["results"], list):
                    for item in res["results"]:
                        if isinstance(item, dict) and "url" in item:
                            sources.append({
                                "url": item["url"],
                                "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""),
                                "task_id": result.task_id,
                            })
                elif "papers" in res and isinstance(res["papers"], list):
                    for paper in res["papers"]:
                        if isinstance(paper, dict) and "pdf_url" in paper:
                            sources.append({
                                "url": paper["pdf_url"],
                                "title": paper.get("title", ""),
                                "snippet": paper.get("summary", "")[:200],
                                "task_id": result.task_id,
                            })

        # 去重
        seen = set()
        unique_sources = []
        for s in sources:
            key = s.get("url", "")
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(s)

        try:
            metadata = dict(result.metadata or {})
            metadata.update({
                "token_usage": result.token_usage,
                "status": result.status.value,
                "task_description": task_description,
                "from_memory": False,
                "generated_by": "researcher",
            })

            entry = KnowledgeEntry(
                id=KnowledgeBase.make_id(self._query, task_description),
                content=content,
                task_type=task_type,
                topic=self._query,
                confidence=result.confidence,
                sources=unique_sources,
                metadata=metadata,
            )
            stored = await self.knowledge_base.add(entry)
            action = stored.metadata.get("action", "inserted")
            logger.info(
                "子任务结果已写入知识库 [%s] confidence=%.2f action=%s",
                result.task_id, result.confidence, action
            )
        except Exception:
            logger.exception("写入知识库失败 [%s]", result.task_id)

    async def _do_synthesizing(self) -> WorkflowState:
        """SYNTHESIZING：调用 SummarizerAgent 合成报告。"""
        if self.agent_pool is None:
            logger.error("AgentPool 未配置，无法合成报告")
            return WorkflowState.FAILED

        from ..core.schema import AgentStatus, SubTask

        # 构建合成任务的上下文
        synthesize_context = {
            "query": self._query,
            "results": self._results,
            "session_id": self._session_id,
        }

        # 创建虚拟合成任务
        synth_task = SubTask(
            id="synthesize",
            description="合成最终研究报告",
            task_type="synthesize",
        )

        self._emit("task_started", {
            "task_id": synth_task.id,
            "task_type": synth_task.task_type,
            "description": synth_task.description,
            "dependencies": [task_id for task_id in self._task_map],
        })
        agent = await self.agent_pool.get_agent("synthesize")
        try:
            result = await agent.run(synth_task, synthesize_context)
        except Exception as e:
            logger.exception("合成报告失败")
            return WorkflowState.FAILED
        finally:
            await self.agent_pool.release_agent(agent)

        if result.status != AgentStatus.SUCCESS:
            logger.error("SummarizerAgent 返回失败状态: %s", result.status.value)
            return WorkflowState.FAILED

        self._emit_agent_result(synth_task, result)

        report = result.output
        self._report = report
        logger.info("报告合成完成 (confidence=%.2f, sources=%d)", getattr(report, "confidence", 0.0), len(getattr(report, "sources", [])))

        # 判断是否进入对抗优化阶段
        enable_adversarial = False
        if self._context is not None and self._context.enable_adversarial:
            enable_adversarial = True
        elif self.config.get("adversarial", {}).get("enabled", False):
            enable_adversarial = True

        if enable_adversarial:
            return WorkflowState.ADVERSARIAL
        return WorkflowState.DONE

    async def _do_adversarial(self) -> WorkflowState:
        """ADVERSARIAL：5 维 Red 并发攻击 + IssueMerger 合并仲裁 + Blue 按 fix_type 批量修复。

        每轮流程：
          1. 并发调用 5 个维度的 Red Agent。
          2. 用 IssueMerger（rule-based + 可选 LLM 仲裁）合并、去重、消解冲突。
          3. 按 REMOVAL → SEARCH → IN_PLACE 顺序批量调用 Blue Agent 修复。
          4. 计算 overall_score，检查终止条件。
        """
        report = self._report
        if report is None:
            logger.error("ADVERSARIAL 状态未找到报告")
            return WorkflowState.FAILED

        cfg = self.config.get("adversarial", {})
        max_rounds = int(cfg.get("max_rounds", 5))
        score_threshold = float(cfg.get("score_threshold", 9.0))
        entry_confidence_threshold = float(cfg.get("entry_confidence_threshold", 0.8))
        delta_threshold = float(cfg.get("delta_threshold", 0.2))
        enable_oscillation_check = bool(cfg.get("enable_oscillation_check", True))
        save_logs = bool(cfg.get("save_logs", True))
        min_severity = Severity(str(cfg.get("min_severity_to_fix", "major")).lower())

        # 入口条件：合成报告置信度已足够高时，无需再进入对抗修复
        report_confidence = float(getattr(report, "confidence", 0.0))
        if report_confidence >= entry_confidence_threshold:
            logger.info(
                "报告置信度 %.2f 已达到入口阈值 %.2f，跳过对抗优化",
                report_confidence,
                entry_confidence_threshold,
            )
            report.adversarial_rounds = 0
            report.final_score = 0.0
            report.adversarial_history = []
            report.dimension_scores = {}
            self._report = report
            return WorkflowState.DONE

        severity_rank = {Severity.CRITICAL: 3, Severity.MAJOR: 2, Severity.MINOR: 1}
        min_rank = severity_rank.get(min_severity, 2)

        history: list[dict] = []
        historical_issues: list[dict] = []
        prev_score = 0.0

        for round_no in range(1, max_rounds + 1):
            logger.info("开始第 %d/%d 轮对抗优化", round_no, max_rounds)
            dimension_attacks: dict[AttackDimension, DimensionAttack] = {}

            # 1. 五个维度并发 Red 攻击
            red_tasks = [
                self._run_red_dimension(report, dimension, round_no)
                for dimension in AttackDimension
            ]
            red_results: list[DimensionAttack | BaseException] = []
            try:
                red_results = await asyncio.gather(*red_tasks, return_exceptions=True)
            except Exception:
                logger.exception("Red Agent 并发调用失败")
                return WorkflowState.FAILED

            all_issues: list[Issue] = []
            for dimension, dim_attack in zip(AttackDimension, red_results):
                if isinstance(dim_attack, Exception):
                    logger.exception("Red Agent 维度 %s 调用失败", dimension.value)
                    return WorkflowState.FAILED
                if not isinstance(dim_attack, DimensionAttack):
                    logger.error("Red Agent 维度 %s 返回类型错误: %s", dimension.value, type(dim_attack))
                    return WorkflowState.FAILED

                dimension_attacks[dimension] = dim_attack
                logger.info(
                    "第 %d 轮 - 维度 %s Red 攻击完成: score=%.2f, issues=%d, summary=%s",
                    round_no,
                    dimension.value,
                    dim_attack.dimension_score,
                    len(dim_attack.issues),
                    dim_attack.analysis_summary,
                )

                issues = [
                    i for i in dim_attack.issues
                    if severity_rank.get(i.severity, 0) >= min_rank
                ]
                all_issues.extend(issues)

            # 2. 合并去重 + 冲突仲裁（可选 LLM 二次仲裁），再按 fix_type 分组批量修复
            round_blue_fixes: list[dict] = []
            round_blue_new_issues: list[dict] = []
            if all_issues:
                merged_issues = await IssueMerger.merge_issues(
                    all_issues,
                    llm_client=self.issue_arbiter_client,
                    query=self._query,
                    report_content=report.content,
                )
                logger.info(
                    "第 %d 轮 - Red 共发现 %d 个问题，合并后剩余 %d 个",
                    round_no,
                    len(all_issues),
                    len(merged_issues),
                )

                # 按 fix_type 分组
                fix_type_groups: dict[FixType, list[Issue]] = defaultdict(list)
                for issue in merged_issues:
                    fix_type_groups[issue.fix_type].append(issue)

                # 按 fix_type 保守程度排序：先删除，再补来源，最后改措辞
                fix_type_order = {FixType.REMOVAL: 0, FixType.SEARCH: 1, FixType.IN_PLACE: 2}
                for fix_type in sorted(fix_type_groups.keys(), key=lambda ft: fix_type_order.get(ft, 99)):
                    group = fix_type_groups[fix_type]
                    try:
                        blue_result = await self._run_blue_agent(report, group[0].dimension, group)
                    except Exception:
                        logger.exception(
                            "Blue Agent 批量修复 %s 类型失败", fix_type.value
                        )
                        return WorkflowState.FAILED
                    if not isinstance(blue_result.output, ResearchReport):
                        logger.error("Blue Agent 返回类型错误: %s", type(blue_result.output))
                        return WorkflowState.FAILED

                    report = blue_result.output
                    if blue_result.trajectory:
                        traj = blue_result.trajectory[0]
                        round_blue_fixes.extend(traj.get("fixes", []))
                        round_blue_new_issues.extend(traj.get("self_verify_new_issues", []))
            else:
                logger.info("第 %d 轮 - 无突出问题，跳过修复", round_no)

            # 3. 用权重算法得到本轮整体评分（无 Judge LLM）
            red_result = RedAttackResult(
                round_no=round_no,
                dimension_attacks=dimension_attacks,
            )
            overall_score = red_result.compute_overall_score()
            red_result.overall_score = overall_score
            red_result.overall_summary = self._build_red_summary(dimension_attacks)

            outstanding = red_result.outstanding_issues(min_severity)
            current_issue_dicts = [
                {
                    "dimension": i.dimension.value,
                    "severity": i.severity.value,
                    "location": i.location,
                    "description": i.description,
                    "fix_type": i.fix_type.value,
                    "evidence": i.evidence,
                }
                for i in outstanding
            ]

            logger.info(
                "第 %d 轮 Red 攻击完成：overall_score=%.2f，突出问题 %d 个",
                round_no,
                overall_score,
                len(outstanding),
            )

            history.append({
                "round": round_no,
                "red_overall_score": overall_score,
                "red_dimension_scores": {
                    d.value: da.dimension_score
                    for d, da in dimension_attacks.items()
                },
                "outstanding_issues": current_issue_dicts,
                "blue_fixes": round_blue_fixes,
                "blue_new_issues": round_blue_new_issues,
            })

            logger.info(
                "第 %d 轮 Blue 修复完成：%d 批修复，自检发现 %d 个新问题，报告长度 %d 字符",
                round_no,
                len(round_blue_fixes),
                len(round_blue_new_issues),
                len(report.content),
            )

            # 终止条件 1：评分达到目标阈值
            if overall_score >= score_threshold:
                logger.info(
                    "报告整体评分 %.2f 达到阈值 %.2f，对抗结束",
                    overall_score,
                    score_threshold,
                )
                break

            # 终止条件 3：无突出问题
            if not outstanding:
                logger.info("Red 未发现需修复的突出问题，对抗提前结束")
                break

            # 终止条件 4：Issue 震荡检测
            if enable_oscillation_check and historical_issues:
                oscillated = [
                    issue for issue in current_issue_dicts
                    if issue in historical_issues
                ]
                if oscillated:
                    logger.info(
                        "检测到 %d 个完全相同的问题重复出现，判定为震荡，对抗结束",
                        len(oscillated),
                    )
                    history[-1]["terminated_by_oscillation"] = True
                    history[-1]["oscillated_issues"] = oscillated
                    break

            # 累积历史 issue 用于下一轮震荡检测
            historical_issues.extend(current_issue_dicts)

            # 终止条件 5：相邻轮评分变化收敛
            if round_no > 1 and abs(overall_score - prev_score) < delta_threshold:
                logger.info(
                    "评分变化 %.2f 小于阈值 %.2f，对抗收敛",
                    abs(overall_score - prev_score),
                    delta_threshold,
                )
                break
            prev_score = overall_score

        # 更新最终报告
        report.adversarial_rounds = len(history)
        report.final_score = history[-1]["red_overall_score"] if history else 0.0
        report.adversarial_history = history
        report.dimension_scores = {
            d: history[-1]["red_dimension_scores"].get(d.value, 0.0)
            for d in AttackDimension
        } if history else {}
        self._report = report

        logger.info(
            "对抗优化完成：共 %d 轮，最终评分 %.2f",
            report.adversarial_rounds,
            report.final_score,
        )

        if save_logs:
            try:
                await self._save_adversarial_logs(history)
            except Exception:
                logger.exception("保存对抗日志失败")

        return WorkflowState.DONE

    async def _do_replanning(self) -> WorkflowState:
        """REPLANNING：调用 Planner 增量重规划。"""
        if self.planner is None:
            return WorkflowState.FAILED

        from ..core.schema import AgentStatus
        failed_tasks = []
        for r in self._results:
            if r.status != AgentStatus.SUCCESS:
                st = self._task_map.get(r.task_id)
                if st:
                    failed_tasks.append(st)

        preserved = {r.task_id: r.output for r in self._results if r.status == AgentStatus.SUCCESS}
        reason = self._build_failure_reason(self._results)

        try:
            dag, subtasks = await self.planner.replan(
                self._query, failed_tasks, preserved, reason
            )
            self._dag = dag
            self._task_map = {st.id: st for st in subtasks}
            self._results = []
            logger.info("重规划完成: %d 个子任务", len(dag))
        except Exception as e:
            logger.exception("重规划失败")
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
                return WorkflowState.SYNTHESIZING
            return WorkflowState.FAILED

        return WorkflowState.DISPATCHING

    async def _on_done(self) -> WorkflowState:
        """Persist the final report to Session Memory before finishing."""
        if self.session_memory is not None and self._report is not None:
            try:
                report_content = getattr(self._report, "content", "")
                if report_content:
                    dag_dict = self._dag.to_dict() if self._dag is not None else {}
                    # 把 task 详情也写入 session memory，便于后续继续研究时复用
                    if self._task_map:
                        dag_dict["tasks"] = {
                            task_id: {
                                "task_id": task.id,
                                "description": task.description,
                                "task_type": task.task_type,
                                "dependencies": task.dependencies,
                                "search_hints": task.search_hints,
                                "timeout_seconds": task.timeout_seconds,
                                "priority": task.priority,
                                "expected_type": task.expected_type,
                            }
                            for task_id, task in self._task_map.items()
                        }
                    await self.session_memory.add_round(
                        session_id=self._session_id,
                        round=self._round,
                        query=self._query,
                        dag=dag_dict,
                        report=report_content,
                    )
                    logger.info(
                        "Session round persisted [session=%s round=%d]",
                        self._session_id,
                        self._round,
                    )
            except Exception:
                logger.exception("持久化会话轮次失败")
        return WorkflowState.DONE

    async def _on_failed(self) -> WorkflowState:
        return WorkflowState.FAILED

    # ------------------------------------------------------------------
    # Adversarial helpers
    # ------------------------------------------------------------------
    def _adversarial_step_timeout(self) -> float | None:
        """读取单步超时配置。返回 None 表示不限制单步时间。"""
        cfg = self.config.get("adversarial", {}).get("per_step_timeout_seconds")
        if cfg is None:
            return None
        return float(cfg)

    async def _run_red_dimension(
        self,
        report: ResearchReport,
        dimension: AttackDimension,
        round_no: int,
    ) -> DimensionAttack:
        """调用 RedTeamAgent 攻击单个维度，带单步超时。"""
        if self.agent_pool is None:
            raise RuntimeError("AgentPool not configured")

        from ..core.schema import AgentStatus, SubTask

        task = SubTask(
            id=f"red_{round_no}_{dimension.value}",
            description=f"第 {round_no} 轮 {dimension.value} 维度攻击",
            task_type="red_agent",
        )
        self._emit("task_started", {
            "task_id": task.id,
            "task_type": task.task_type,
            "description": task.description,
            "dependencies": ["synthesize"],
        })
        agent = await self.agent_pool.get_agent("red_agent")
        step_timeout = self._adversarial_step_timeout()
        try:
            if step_timeout is None:
                result = await agent.run(
                    task,
                    {
                        "report": report,
                        "query": self._query,
                        "dimension": dimension,
                        "round_no": round_no,
                    },
                )
            else:
                result = await asyncio.wait_for(
                    agent.run(
                        task,
                        {
                            "report": report,
                            "query": self._query,
                            "dimension": dimension,
                            "round_no": round_no,
                        },
                    ),
                    timeout=step_timeout,
                )
        except asyncio.TimeoutError:
            logger.error(
                "Red Agent 维度 %s 调用超时（>%s 秒）",
                dimension.value,
                step_timeout,
            )
            raise
        finally:
            await self.agent_pool.release_agent(agent)

        if result.status != AgentStatus.SUCCESS:
            raise RuntimeError(f"Red agent failed: {result.output}")
        self._emit_agent_result(task, result)
        return result.output

    async def _run_blue_agent(
        self,
        report: ResearchReport,
        dimension: AttackDimension,
        issues: list,
    ) -> "AgentResult":
        """调用 BlueTeamAgent 批量修复同一 fix_type 的 issues，带单步超时。"""
        if self.agent_pool is None:
            raise RuntimeError("AgentPool not configured")

        from ..core.schema import AgentStatus, SubTask

        fix_type = issues[0].fix_type.value if issues else "none"
        task = SubTask(
            id=f"blue_{fix_type}_{len(issues)}",
            description=f"批量修复 {fix_type} 类型共 {len(issues)} 个问题",
            task_type="blue_agent",
        )
        self._emit("task_started", {
            "task_id": task.id,
            "task_type": task.task_type,
            "description": task.description,
            "dependencies": ["synthesize"],
        })
        agent = await self.agent_pool.get_agent("blue_agent")
        step_timeout = self._adversarial_step_timeout()
        try:
            if step_timeout is None:
                result = await agent.run(
                    task,
                    {
                        "report": report,
                        "dimension": dimension,
                        "issues": issues,
                        "query": self._query,
                        "results": self._results,
                    },
                )
            else:
                result = await asyncio.wait_for(
                    agent.run(
                        task,
                        {
                            "report": report,
                            "dimension": dimension,
                            "issues": issues,
                            "query": self._query,
                            "results": self._results,
                        },
                    ),
                    timeout=step_timeout,
                )
        except asyncio.TimeoutError:
            logger.error(
                "Blue Agent 维度 %s 调用超时（>%s 秒）",
                dimension.value,
                step_timeout,
            )
            raise
        finally:
            await self.agent_pool.release_agent(agent)

        if result.status != AgentStatus.SUCCESS:
            raise RuntimeError(f"Blue agent failed: {result.output}")
        self._emit_agent_result(task, result)
        return result

    async def _run_blue_fix(
        self,
        report: ResearchReport,
        issue: Issue,
    ) -> "AgentResult":
        """调用 BlueTeamAgent 修复单个 issue。"""
        return await self._run_blue_agent(report, issue.dimension, [issue])

    def _build_red_summary(self, dimension_attacks: dict[AttackDimension, DimensionAttack]) -> str:
        """构建本轮 Red 攻击汇总。"""
        parts = ["本轮 Red 攻击汇总："]
        for dim in AttackDimension:
            da = dimension_attacks.get(dim)
            if da is None:
                continue
            issue_count = len(da.issues)
            parts.append(
                f"- {dim.value}: 维度得分 {da.dimension_score:.1f}/10，发现 {issue_count} 个问题"
            )
        return "\n".join(parts)

    async def _save_adversarial_logs(self, history: list[dict]) -> None:
        """保存每轮对抗日志到 outputs/adversarial/{session_id}/。"""
        from datetime import datetime

        work_dir = Path(self.config.get("system", {}).get("work_dir", "./outputs"))
        out_dir = work_dir / "adversarial" / self._session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for record in history:
            round_no = record["round"]
            red_path = out_dir / f"round_{round_no}_red.json"
            red_path.write_text(
                json.dumps(
                    {
                        "round": round_no,
                        "red_overall_score": record["red_overall_score"],
                        "red_dimension_scores": record["red_dimension_scores"],
                        "outstanding_issues": record["outstanding_issues"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            blue_path = out_dir / f"round_{round_no}_blue.json"
            blue_path.write_text(
                json.dumps(
                    {
                        "round": round_no,
                        "fixes": record.get("blue_fixes", []),
                        "self_verify_new_issues": record.get("blue_new_issues", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        summary_path = out_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "session_id": self._session_id,
                    "query": self._query,
                    "timestamp": datetime.now().isoformat(),
                    "total_rounds": len(history),
                    "final_score": history[-1]["red_overall_score"] if history else 0.0,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info("对抗日志已保存: %s", out_dir)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    def _should_replan(self, results: list) -> bool:
        from ..core.schema import AgentStatus
        if not results:
            return False
        total = len(results)
        failed = sum(1 for r in results if r.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT))
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)

        if failed / total > 0.5:
            return True
        if success / total < 0.3 and failed > 0:
            return True
        return False

    def _build_failure_reason(self, results: list) -> str:
        from ..core.schema import AgentStatus
        reasons = []
        timeout_count = sum(1 for r in results if r.status == AgentStatus.TIMEOUT)
        failed_count = sum(1 for r in results if r.status == AgentStatus.FAILED)
        if timeout_count > 0:
            reasons.append(f"{timeout_count} 个任务超时")
        if failed_count > 0:
            reasons.append(f"{failed_count} 个任务失败")
        return "; ".join(reasons) if reasons else "未知失败"

    # ------------------------------------------------------------------
    # Hard-coded state-transition decision logic (for custom handler mode)
    # ------------------------------------------------------------------
    def _decide_next_state(
        self,
        current: WorkflowState,
        result: HandlerResult,
        context: ResearchContext,
    ) -> Optional[WorkflowState]:
        if current == WorkflowState.PLANNING:
            return WorkflowState.DISPATCHING if result.status == "success" else WorkflowState.FAILED

        if current == WorkflowState.DISPATCHING:
            return WorkflowState.COLLECTING

        if current == WorkflowState.COLLECTING:
            if result.status == "success":
                return WorkflowState.SYNTHESIZING
            if result.status == "partial_failure":
                return WorkflowState.REPLANNING
            return WorkflowState.FAILED

        if current == WorkflowState.REPLANNING:
            return WorkflowState.PLANNING if result.status == "success" else WorkflowState.FAILED

        if current == WorkflowState.SYNTHESIZING:
            if result.status != "success":
                return WorkflowState.FAILED
            return WorkflowState.ADVERSARIAL if context.enable_adversarial else WorkflowState.DONE

        if current == WorkflowState.ADVERSARIAL:
            if result.status == "success":
                return WorkflowState.DONE
            if result.status == "partial_failure":
                return WorkflowState.SYNTHESIZING
            return WorkflowState.FAILED

        return None
