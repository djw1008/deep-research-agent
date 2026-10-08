"""Orchestrator — sole controller of the workflow state machine."""

import asyncio
import copy
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
from ..observability import _safe as _json_safe

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

    def _emit_merge_summary(
        self,
        round_no: int,
        red_result: "RedAttackResult",
        all_issues: list,
        merged_issues: list,
        fix_type_groups: dict,
    ) -> None:
        """Emit a synthetic per-round merge node so the dashboard timeline shows
        the structured attack aggregate and how issues are dispatched to Blue."""
        task_id = f"merge_round_{round_no}"
        dimension_scores = {
            d.value: da.dimension_score
            for d, da in red_result.dimension_attacks.items()
        }
        dispatch = {ft.value: len(group) for ft, group in fix_type_groups.items()}
        summary = {
            "round_no": round_no,
            "overall_score": red_result.overall_score,
            "dimension_scores": dimension_scores,
            "issues_before_merge": len(all_issues),
            "issues_after_merge": len(merged_issues),
            "fix_type_dispatch": dispatch,
        }
        self._emit("task_started", {
            "task_id": task_id,
            "task_type": "merge",
            "description": f"第 {round_no} 轮攻击结果汇总与分发",
            "dependencies": [f"red_{round_no}_{d.value}" for d in AttackDimension],
        })
        self._emit("agent_loop_event", {
            "task_id": task_id,
            "event": {
                "turn": 0,
                "role": "assistant",
                "log": True,
                "content": (
                    f"第 {round_no} 轮汇总：加权总评分 {red_result.overall_score:.2f}/10；"
                    f"原始 issue {len(all_issues)} 个，合并去重后 {len(merged_issues)} 个；"
                    f"按修复类型分发给 Blue：{dispatch if dispatch else '无需修复'}"
                ),
            },
        })
        self._emit("agent_loop_event", {
            "task_id": task_id,
            "event": {
                "turn": 0,
                "role": "tool",
                "name": "issue_merger",
                "result": {**summary, "merged_issues": merged_issues},
            },
        })
        self._emit("task_completed", {
            "task_id": task_id,
            "status": "success",
            "confidence": red_result.overall_score / 10.0,
            "token_usage": 0,
            "output": summary,
            "metadata": {},
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
                            # Dependency citations are task-local.  Pass their
                            # bindings with the text so the downstream agent can
                            # cite them through an unambiguous task namespace.
                            dep_sources = dep_result.metadata.get("sources", [])
                            ctx[f"dep_sources:{dep_id}"] = (
                                dep_sources if isinstance(dep_sources, list) else []
                            )

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
                "trajectory": _json_safe(r.trajectory),
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

        # ResearchAgent 已将列表收敛为 output 实际引用的来源。
        sources = result.metadata.get("sources", [])
        unique_sources = (
            [
                dict(source)
                for source in sources
                if isinstance(source, dict) and str(source.get("url", "")).strip()
            ]
            if isinstance(sources, list)
            else []
        )

        try:
            metadata = dict(result.metadata or {})
            # sources 有独立存储列，不在 metadata 中重复保存。
            metadata.pop("sources", None)
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
          1. 并发调用 5 个维度的 Red Agent，对当前报告评分。
          2. 评分达标/无问题/震荡/不再提升时直接终止（评的就是当前报告，分数与制品一致）。
          3. 否则用 IssueMerger（rule-based + 可选 LLM 仲裁）合并去重，按 REMOVAL → SEARCH → IN_PLACE 分批修复。
        循环最多 max_rounds 次修复，之后追加一轮只评不修的 Red 复评。
        循环结束后若历史最佳版本的分数高于最终版本，回退到最佳版本交付。
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
        # （用户显式触发对抗升级时通过 _force_adversarial 绕过该判断）
        report_confidence = float(getattr(report, "confidence", 0.0))
        if report_confidence >= entry_confidence_threshold and not getattr(self, "_force_adversarial", False):
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
        # 最佳版本快照：每轮 Red 评分对应的是本轮修复前的报告，
        # 若后续轮次的修复反而拉低分数，循环结束后回退到历史最佳版本交付
        best_score = -1.0
        best_report: Any | None = None
        best_round = 0

        # max_rounds 限制 Blue 修复次数。由于每次修复后的报告都必须再由 Red
        # 评分，循环最多包含 max_rounds + 1 次 Red 评估；最后一次只评估、不修复。
        for round_no in range(1, max_rounds + 2):
            logger.info("开始第 %d 轮对抗评估（最多修复 %d 轮）", round_no, max_rounds)
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

            # 2. 汇总本轮攻击：加权评分 + 合并去重 + 冲突仲裁，再按 fix_type 分发修复
            red_result = RedAttackResult(
                round_no=round_no,
                dimension_attacks=dimension_attacks,
            )
            overall_score = red_result.compute_overall_score()
            red_result.overall_score = overall_score
            red_result.overall_summary = self._build_red_summary(dimension_attacks)

            # 在本轮 Blue 修复前打快照：overall_score 评的是当前这个版本的报告
            if overall_score > best_score:
                best_score = overall_score
                best_report = copy.deepcopy(report)
                best_round = round_no

            # 2. 先依据 Red 结果决定是否需要修复。评分和问题均对应当前 report，
            # 因此达标时必须在 Blue 修改前退出，避免最终报告与 final_score 不一致。
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
                "blue_fixes": [],
            })

            # 终止条件 1：评分达到目标阈值，不再调用 Blue
            if overall_score >= score_threshold:
                logger.info(
                    "报告整体评分 %.2f 达到阈值 %.2f，对抗结束",
                    overall_score,
                    score_threshold,
                )
                break

            # 终止条件 2：无突出问题
            if not outstanding:
                logger.info("Red 未发现需修复的突出问题，对抗提前结束")
                break

            # 终止条件 3：Issue 震荡检测
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

            # 终止条件 4：评分不再明显提升（含下降）即收敛；
            # 不能用 abs()——分数下降说明修复在帮倒忙，更应该停
            if round_no > 1 and (overall_score - prev_score) < delta_threshold:
                logger.info(
                    "评分提升 %.2f 小于阈值 %.2f（或无提升），对抗收敛",
                    overall_score - prev_score,
                    delta_threshold,
                )
                break

            # max_rounds 次 Blue 修复已用完。当前轮是最后一次修复后的 Red
            # 复评，只记录真实最终分数，不再产生未经复评的新修改。
            if round_no > max_rounds:
                logger.info("已完成 %d 轮 Blue 修复及最终 Red 复评，对抗结束", max_rounds)
                history[-1]["final_verification"] = True
                break

            historical_issues.extend(current_issue_dicts)
            prev_score = overall_score

            # 3. 未达标时才合并问题，并按 fix_type 分发给 Blue 修复
            round_blue_fixes: list[dict] = []
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

            fix_type_groups: dict[FixType, list[Issue]] = defaultdict(list)
            for issue in merged_issues:
                fix_type_groups[issue.fix_type].append(issue)

            self._emit_merge_summary(round_no, red_result, all_issues, merged_issues, fix_type_groups)

            # 按 fix_type 保守程度排序：先删除，再补来源，最后改措辞
            fix_type_order = {FixType.REMOVAL: 0, FixType.SEARCH: 1, FixType.IN_PLACE: 2}
            for fix_type in sorted(fix_type_groups.keys(), key=lambda ft: fix_type_order.get(ft, 99)):
                group = fix_type_groups[fix_type]
                try:
                    blue_result = await self._run_blue_agent(report, group[0].dimension, group)
                except Exception:
                    logger.exception("Blue Agent 批量修复 %s 类型失败", fix_type.value)
                    return WorkflowState.FAILED
                if not isinstance(blue_result.output, ResearchReport):
                    logger.error("Blue Agent 返回类型错误: %s", type(blue_result.output))
                    return WorkflowState.FAILED

                report = blue_result.output
                if blue_result.metadata:
                    round_blue_fixes.extend(blue_result.metadata.get("fixes", []))

            history[-1]["blue_fixes"] = round_blue_fixes
            logger.info(
                "第 %d 轮 Blue 修复完成：%d 批修复，报告长度 %d 字符",
                round_no,
                len(round_blue_fixes),
                len(report.content),
            )

        # 更新最终报告：若后续轮次的修复反而拉低了分数，回退到历史最佳版本交付，
        # 保证交付的报告与 final_score 一致
        final_round_score = history[-1]["red_overall_score"] if history else 0.0
        if history:
            history[-1]["best_round"] = best_round
            history[-1]["best_score"] = best_score
        if best_report is not None and best_score > final_round_score:
            logger.info(
                "历史最佳版本出现在第 %d 轮（%.2f > %.2f），回退到该版本交付",
                best_round,
                best_score,
                final_round_score,
            )
            if history:
                history[-1]["restored_best"] = True
            report = best_report
            delivered_score = best_score
            delivered_dimension_scores = {
                d: history[best_round - 1]["red_dimension_scores"].get(d.value, 0.0)
                for d in AttackDimension
            }
        else:
            delivered_score = final_round_score
            delivered_dimension_scores = {
                d: history[-1]["red_dimension_scores"].get(d.value, 0.0)
                for d in AttackDimension
            } if history else {}

        report.adversarial_rounds = len(history)
        report.final_score = delivered_score
        report.adversarial_history = history
        report.dimension_scores = delivered_dimension_scores
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

    async def run_adversarial_upgrade(
        self,
        report: ResearchReport,
        session_id: str | None = None,
    ) -> ResearchReport:
        """只对已有报告执行 Red/Blue 对抗优化，不重新跑研究链路。

        供 `run.py --upgrade-adversarial` 使用：调用方负责从 report.md 重建
        ResearchReport 并注入 event_sink。`_do_adversarial` 依赖的实例属性
        （_report/_query/_session_id/_results）在此设置为安全值；其余属性
        沿用 __init__ 默认值即可（_dag/_task_map/_round/_context 不参与对抗流程）。
        """
        self._report = report
        self._query = report.query
        self._session_id = session_id or f"upgrade-{uuid.uuid4().hex[:12]}"
        self._results = []
        self._context = ResearchContext(topic=report.query, enable_adversarial=True)
        self._force_adversarial = True  # 绕过 _do_adversarial 的高置信度入口跳过
        await self._do_adversarial()
        return self._report

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
                        sources=list(getattr(self._report, "sources", []) or []),
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
