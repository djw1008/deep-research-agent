"""Tests for the Orchestrator."""

import asyncio

import pytest

from deep_research.core import (
    HandlerResult,
    InvalidTransitionError,
    Orchestrator,
    ResearchContext,
    SubTask,
    WorkflowState,
)
from deep_research.core.schema import (
    AgentResult,
    AgentStatus,
    AttackDimension,
    DimensionAttack,
    RedAttackResult,
    ResearchReport,
)


@pytest.fixture
def orch():
    return Orchestrator()


@pytest.fixture
def context():
    return ResearchContext(topic="AI safety")


@pytest.mark.asyncio
async def test_knowledge_persistence_uses_cited_sources_from_metadata():
    class CaptureKnowledgeBase:
        def __init__(self):
            self.entry = None

        async def add(self, entry):
            self.entry = entry
            return entry

    knowledge_base = CaptureKnowledgeBase()
    orchestrator = Orchestrator(knowledge_base=knowledge_base)
    orchestrator._query = "topic"
    orchestrator._task_map = {
        "research_1": SubTask(
            id="research_1",
            description="research task",
            task_type="search",
        )
    }
    cited_sources = [{
        "source_label": "SRC-1",
        "url": "https://example.com/used",
        "title": "Used source",
        "snippet": "evidence",
    }]
    result = AgentResult(
        task_id="research_1",
        status=AgentStatus.SUCCESS,
        output="Finding [SRC-1].",
        confidence=0.9,
        metadata={"from_memory": False, "sources": cited_sources},
    )

    await orchestrator._persist_to_knowledge_base(result)

    assert knowledge_base.entry.sources == cited_sources
    assert "sources" not in knowledge_base.entry.metadata


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_idle_allowed_transitions(orch):
    assert orch.get_allowed_transitions() == {
        WorkflowState.PLANNING,
        WorkflowState.FAILED,
    }


def test_terminal_states_have_no_exits(orch, context):
    orch._state = WorkflowState.DONE
    assert orch.get_allowed_transitions() == set()
    orch._state = WorkflowState.FAILED
    assert orch.get_allowed_transitions() == set()


@pytest.mark.asyncio
async def test_illegal_transition_raises(orch, context):
    with pytest.raises(InvalidTransitionError):
        await orch.transition(WorkflowState.DONE, context)


# ---------------------------------------------------------------------------
# Happy path (adversarial OFF)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_run_success_adversarial_off(orch, context):
    context.enable_adversarial = False

    async def plan(orch, ctx):
        ctx.plan = ["q1", "q2"]
        return HandlerResult(status="success")

    async def dispatch(orch, ctx):
        return HandlerResult(status="success")

    async def collect(orch, ctx):
        ctx.metadata["results"] = {"q1": "a1"}
        return HandlerResult(status="success")

    async def synthesize(orch, ctx):
        ctx.report_draft = "# Report"
        return HandlerResult(status="success")

    orch.on_state(WorkflowState.PLANNING, plan)
    orch.on_state(WorkflowState.DISPATCHING, dispatch)
    orch.on_state(WorkflowState.COLLECTING, collect)
    orch.on_state(WorkflowState.SYNTHESIZING, synthesize)

    final = await orch.run(context)

    assert final == WorkflowState.DONE
    assert len(orch.history) == 5  # I→P→D→C→S→Done
    assert context.report_draft == "# Report"


# ---------------------------------------------------------------------------
# Replanning loop
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_replanning_loop(orch, context):
    plan_count = 0
    dispatch_count = 0

    async def plan(orch, ctx):
        nonlocal plan_count
        plan_count += 1
        ctx.subtasks = [SubTask(id=f"t{plan_count}", description=f"task{plan_count}")]
        return HandlerResult(status="success")

    async def dispatch(orch, ctx):
        nonlocal dispatch_count
        dispatch_count += 1
        return HandlerResult(status="success")

    async def collect(orch, ctx):
        if ctx.iteration_count == 0:
            ctx.iteration_count += 1
            return HandlerResult(status="partial_failure", message="Majority failed")
        return HandlerResult(status="success")

    async def replan(orch, ctx):
        ctx.metadata["preserved"] = ctx.metadata.get("results", {})
        return HandlerResult(status="success")

    async def synthesize(orch, ctx):
        return HandlerResult(status="success")

    orch.on_state(WorkflowState.PLANNING, plan)
    orch.on_state(WorkflowState.DISPATCHING, dispatch)
    orch.on_state(WorkflowState.COLLECTING, collect)
    orch.on_state(WorkflowState.REPLANNING, replan)
    orch.on_state(WorkflowState.SYNTHESIZING, synthesize)

    final = await orch.run(context)

    assert final == WorkflowState.DONE
    assert plan_count == 2
    assert dispatch_count == 2
    assert context.metadata["preserved"] is not None


# ---------------------------------------------------------------------------
# Adversarial ON
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_adversarial_path_on(orch, context):
    context.enable_adversarial = True

    async def plan(orch, ctx):
        return HandlerResult(status="success")

    async def dispatch(orch, ctx):
        return HandlerResult(status="success")

    async def collect(orch, ctx):
        return HandlerResult(status="success")

    async def synthesize(orch, ctx):
        return HandlerResult(status="success")

    async def adversarial(orch, ctx):
        return HandlerResult(status="success")

    orch.on_state(WorkflowState.PLANNING, plan)
    orch.on_state(WorkflowState.DISPATCHING, dispatch)
    orch.on_state(WorkflowState.COLLECTING, collect)
    orch.on_state(WorkflowState.SYNTHESIZING, synthesize)
    orch.on_state(WorkflowState.ADVERSARIAL, adversarial)

    final = await orch.run(context)
    assert final == WorkflowState.DONE
    assert len(orch.history) == 6  # I→P→D→C→S→A→Done


# ---------------------------------------------------------------------------
# Adversarial partial failure loops back to synthesizing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_adversarial_partial_failure_loop(orch, context):
    context.enable_adversarial = True

    synthesize_count = 0
    adversarial_count = 0

    async def plan(orch, ctx):
        return HandlerResult(status="success")

    async def dispatch(orch, ctx):
        return HandlerResult(status="success")

    async def collect(orch, ctx):
        return HandlerResult(status="success")

    async def synthesize(orch, ctx):
        nonlocal synthesize_count
        synthesize_count += 1
        return HandlerResult(status="success")

    async def adversarial(orch, ctx):
        nonlocal adversarial_count
        adversarial_count += 1
        if adversarial_count == 1:
            return HandlerResult(status="partial_failure")
        return HandlerResult(status="success")

    orch.on_state(WorkflowState.PLANNING, plan)
    orch.on_state(WorkflowState.DISPATCHING, dispatch)
    orch.on_state(WorkflowState.COLLECTING, collect)
    orch.on_state(WorkflowState.SYNTHESIZING, synthesize)
    orch.on_state(WorkflowState.ADVERSARIAL, adversarial)

    final = await orch.run(context)
    assert final == WorkflowState.DONE
    assert synthesize_count == 2
    assert adversarial_count == 2


# ---------------------------------------------------------------------------
# Concurrency helpers
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_spawn_and_wait(orch, context):
    async def mock_agent(name: str, delay: float):
        await asyncio.sleep(delay)
        return f"result-{name}"

    async def dispatch(orch, ctx):
        orch.spawn_task("a1", mock_agent("a1", 0.01))
        orch.spawn_task("a2", mock_agent("a2", 0.02))
        return HandlerResult(status="success")

    async def collect(orch, ctx):
        results = await orch.wait_pending_tasks()
        ctx.metadata["results"] = results
        return HandlerResult(status="success")

    async def synthesize(orch, ctx):
        return HandlerResult(status="success")

    orch.on_state(WorkflowState.DISPATCHING, dispatch)
    orch.on_state(WorkflowState.COLLECTING, collect)
    orch.on_state(WorkflowState.SYNTHESIZING, synthesize)

    # Manually drive to DISPATCHING and execute handlers
    await orch.transition(WorkflowState.PLANNING, context)
    await orch.transition(WorkflowState.DISPATCHING, context)
    await orch.execute_handler(WorkflowState.DISPATCHING, context)
    await orch.transition(WorkflowState.COLLECTING, context)
    await orch.execute_handler(WorkflowState.COLLECTING, context)

    assert context.metadata["results"] == {
        "a1": "result-a1",
        "a2": "result-a2",
    }


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handler_crash_goes_to_failed(orch, context):
    async def boom(orch, ctx):
        raise RuntimeError("boom")

    orch.on_state(WorkflowState.PLANNING, boom)
    final = await orch.run(context)
    assert final == WorkflowState.FAILED
    assert orch.history[-1].to_state == WorkflowState.FAILED


@pytest.mark.asyncio
async def test_planning_failure_goes_to_failed(orch, context):
    async def bad_plan(orch, ctx):
        return HandlerResult(status="failure", message="cannot plan")

    orch.on_state(WorkflowState.PLANNING, bad_plan)
    final = await orch.run(context)
    assert final == WorkflowState.FAILED


# ---------------------------------------------------------------------------
# Built-in adversarial loop
# ---------------------------------------------------------------------------

class FakeRedAgent:
    def __init__(self, dimension_scores=None, issues_per_dimension=None):
        self.dimension_scores = dimension_scores or {
            AttackDimension.FACTUAL: 8.0,
            AttackDimension.HALLUCINATION: 8.0,
            AttackDimension.LOGIC: 8.0,
            AttackDimension.SOURCE: 8.0,
            AttackDimension.COVERAGE: 8.0,
        }
        self.issues_per_dimension = issues_per_dimension or []

    async def run(self, task, context):
        dimension = context.get("dimension")
        round_no = context.get("round_no", 1)

        if isinstance(dimension, AttackDimension):
            # 单维度模式：Orchestrator 按维度串行调用
            dim_attack = DimensionAttack(
                dimension=dimension,
                dimension_score=self.dimension_scores.get(dimension, 8.0),
                issues=list(self.issues_per_dimension),
            )
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.SUCCESS,
                output=dim_attack,
                trajectory=[{"round": round_no, "dimension": dimension.value}],
                token_usage=0,
                confidence=dim_attack.dimension_score / 10.0,
            )

        # 全维度模式：兼容旧测试/入口
        dimension_attacks = {
            dim: DimensionAttack(
                dimension=dim,
                dimension_score=score,
                issues=list(self.issues_per_dimension),
            )
            for dim, score in self.dimension_scores.items()
        }
        red_result = RedAttackResult(
            round_no=round_no,
            dimension_attacks=dimension_attacks,
        )
        red_result.overall_score = red_result.compute_overall_score()
        return AgentResult(
            task_id=task.id,
            status=AgentStatus.SUCCESS,
            output=red_result,
            trajectory=[{"round": round_no, "dimensions": [d.value for d in dimension_attacks]}],
            token_usage=0,
            confidence=red_result.overall_score / 10.0,
        )


class SequencedFakeRedAgent(FakeRedAgent):
    """按轮次返回不同评分/问题，用于验证修复后的最终复评。"""

    def __init__(self, round_scores, first_round_issues):
        super().__init__()
        self.round_scores = round_scores
        self.first_round_issues = first_round_issues
        self.calls = 0

    async def run(self, task, context):
        round_no = context.get("round_no", 1)
        dimension = context.get("dimension")
        self.calls += 1
        score = self.round_scores[min(round_no - 1, len(self.round_scores) - 1)]
        issues = self.first_round_issues if round_no == 1 else []
        dim_attack = DimensionAttack(
            dimension=dimension,
            dimension_score=score,
            issues=list(issues),
        )
        return AgentResult(
            task_id=task.id,
            status=AgentStatus.SUCCESS,
            output=dim_attack,
            trajectory=[{"round": round_no, "dimension": dimension.value}],
            token_usage=0,
            confidence=score / 10.0,
        )


class FakeBlueAgent:
    async def run(self, task, context):
        report = context.get("report")
        return AgentResult(
            task_id=task.id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=[],
            token_usage=0,
            confidence=report.confidence,
        )


class RecordingFakeBlueAgent:
    """记录每次调用的 dimension 和 issues，用于验证串行按维度修复。"""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    async def run(self, task, context):
        report = context.get("report")
        dimension = context.get("dimension")
        issues = context.get("issues", [])
        self.calls.append((dimension.value if dimension else None, len(issues)))
        return AgentResult(
            task_id=task.id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=[],
            token_usage=0,
            confidence=report.confidence,
        )


class FakeAgentPool:
    def __init__(self, red_agent=None, blue_agent=None):
        self.red_agent = red_agent or FakeRedAgent()
        self.blue_agent = blue_agent or FakeBlueAgent()

    async def get_agent(self, task_type):
        if task_type == "red_agent":
            return self.red_agent
        if task_type == "blue_agent":
            return self.blue_agent
        return None

    async def release_agent(self, agent):
        pass


@pytest.mark.asyncio
async def test_builtin_adversarial_loop_reaches_done():
    """测试内置 ADVERSARIAL handler 能正确跑完 Red→Blue→DONE。"""
    config = {
        "adversarial": {
            "enabled": True,
            "max_rounds": 2,
            "score_threshold": 7.0,
            "delta_threshold": 0.1,
            "min_severity_to_fix": "major",
            "save_logs": False,
        },
        "system": {"work_dir": "./outputs"},
    }

    orch = Orchestrator(agent_pool=FakeAgentPool(), config=config)
    context = ResearchContext(topic="AI safety", enable_adversarial=True)

    # 预置一份报告，直接测试 ADVERSARIAL handler
    # 置信度低于默认入口阈值 0.8，才会进入对抗
    orch._report = ResearchReport(query="AI safety", content="Test report", confidence=0.5)
    orch._state = WorkflowState.ADVERSARIAL

    final_state = await orch._do_adversarial()
    assert final_state == WorkflowState.DONE
    assert orch._report.adversarial_rounds >= 1
    assert orch._report.final_score > 0


@pytest.mark.asyncio
async def test_builtin_adversarial_loop_stops_at_threshold():
    """测试当 Red 评分达到阈值时，对抗循环提前结束。"""
    dimension_scores = {dim: 10.0 for dim in AttackDimension}
    blue_agent = RecordingFakeBlueAgent()
    pool = FakeAgentPool(
        red_agent=FakeRedAgent(dimension_scores=dimension_scores),
        blue_agent=blue_agent,
    )

    config = {
        "adversarial": {
            "enabled": True,
            "max_rounds": 5,
            "score_threshold": 9.5,
            "delta_threshold": 0.1,
            "min_severity_to_fix": "major",
            "save_logs": False,
        },
        "system": {"work_dir": "./outputs"},
    }

    orch = Orchestrator(agent_pool=pool, config=config)
    context = ResearchContext(topic="AI safety", enable_adversarial=True)
    # 置信度低于默认入口阈值 0.8，进入对抗
    orch._report = ResearchReport(query="AI safety", content="Test report", confidence=0.5)
    orch._state = WorkflowState.ADVERSARIAL

    final_state = await orch._do_adversarial()

    assert final_state == WorkflowState.DONE
    assert orch._report.adversarial_rounds == 1
    assert orch._report.final_score == 10.0
    assert len(blue_agent.calls) == 0


@pytest.mark.asyncio
async def test_builtin_adversarial_rescores_after_last_blue_fix():
    """最后一次 Blue 修复后必须再由 Red 评分，final_score 才对应最终报告。"""
    from deep_research.core.schema import Issue, Severity, FixType

    issue = Issue(
        dimension=AttackDimension.FACTUAL,
        severity=Severity.MAJOR,
        location="第1段",
        description="需要修复的问题",
        fix_type=FixType.IN_PLACE,
    )
    red_agent = SequencedFakeRedAgent([8.0, 10.0], [issue])
    blue_agent = RecordingFakeBlueAgent()
    pool = FakeAgentPool(red_agent=red_agent, blue_agent=blue_agent)
    config = {
        "adversarial": {
            "enabled": True,
            "max_rounds": 1,
            "score_threshold": 9.0,
            "entry_confidence_threshold": 0.8,
            "delta_threshold": 0.1,
            "min_severity_to_fix": "major",
            "save_logs": False,
        },
        "system": {"work_dir": "./outputs"},
    }

    orch = Orchestrator(agent_pool=pool, config=config)
    orch._report = ResearchReport(query="AI safety", content="Test report", confidence=0.5)
    orch._state = WorkflowState.ADVERSARIAL

    final_state = await orch._do_adversarial()

    assert final_state == WorkflowState.DONE
    assert len(blue_agent.calls) == 1
    assert red_agent.calls == len(AttackDimension) * 2
    assert orch._report.adversarial_rounds == 2
    assert orch._report.final_score == 10.0
    assert orch._report.adversarial_history[-1]["blue_fixes"] == []


@pytest.mark.asyncio
async def test_builtin_adversarial_batches_issues_by_fix_type():
    """测试 Orchestrator 并发 Red 攻击后，按 fix_type 批量调用 Blue 修复。"""
    from deep_research.core.schema import Issue, Severity, FixType

    dimension_scores = {dim: 8.0 for dim in AttackDimension}
    issues_per_dimension = [
        Issue(
            dimension=AttackDimension.FACTUAL,
            severity=Severity.MAJOR,
            location="第1段",
            description="removal issue",
            fix_type=FixType.REMOVAL,
        ),
        Issue(
            dimension=AttackDimension.SOURCE,
            severity=Severity.MAJOR,
            location="第2段",
            description="search issue",
            fix_type=FixType.SEARCH,
        ),
        Issue(
            dimension=AttackDimension.LOGIC,
            severity=Severity.MAJOR,
            location="第3段",
            description="in_place issue",
            fix_type=FixType.IN_PLACE,
        ),
    ]
    blue_agent = RecordingFakeBlueAgent()
    pool = FakeAgentPool(
        red_agent=FakeRedAgent(
            dimension_scores=dimension_scores,
            issues_per_dimension=issues_per_dimension,
        ),
        blue_agent=blue_agent,
    )

    config = {
        "adversarial": {
            "enabled": True,
            "max_rounds": 1,
            "score_threshold": 9.5,
            "entry_confidence_threshold": 0.5,  # 设低一点，确保能进入对抗
            "delta_threshold": 0.1,
            "min_severity_to_fix": "major",
            "save_logs": False,
        },
        "system": {"work_dir": "./outputs"},
    }

    orch = Orchestrator(agent_pool=pool, config=config)
    context = ResearchContext(topic="AI safety", enable_adversarial=True)
    # confidence 低于入口阈值，确保进入对抗
    orch._report = ResearchReport(query="AI safety", content="Test report", confidence=0.3)
    orch._state = WorkflowState.ADVERSARIAL

    final_state = await orch._do_adversarial()
    assert final_state == WorkflowState.DONE

    # 5 个维度各产生 3 个 issue，经合并/去重后按 fix_type 批量修复，共 3 批
    assert len(blue_agent.calls) == 3
    assert all(call[1] == 1 for call in blue_agent.calls)


@pytest.mark.asyncio
async def test_builtin_adversarial_skips_when_confidence_above_threshold():
    """测试报告 confidence 达到入口阈值时，直接跳过对抗优化。"""
    blue_agent = RecordingFakeBlueAgent()
    pool = FakeAgentPool(blue_agent=blue_agent)

    config = {
        "adversarial": {
            "enabled": True,
            "max_rounds": 5,
            "score_threshold": 9.5,
            "entry_confidence_threshold": 0.8,
            "delta_threshold": 0.1,
            "min_severity_to_fix": "major",
            "save_logs": False,
        },
        "system": {"work_dir": "./outputs"},
    }

    orch = Orchestrator(agent_pool=pool, config=config)
    context = ResearchContext(topic="AI safety", enable_adversarial=True)
    # confidence 高于入口阈值，应跳过对抗
    orch._report = ResearchReport(query="AI safety", content="Test report", confidence=0.9)
    orch._state = WorkflowState.ADVERSARIAL

    final_state = await orch._do_adversarial()
    assert final_state == WorkflowState.DONE
    assert orch._report.adversarial_rounds == 0
    assert orch._report.final_score == 0.0
    # Red 和 Blue 都没被调用
    assert len(blue_agent.calls) == 0


@pytest.mark.asyncio
async def test_builtin_adversarial_terminates_on_oscillation():
    """测试完全相同 issue 重复出现时触发震荡终止。"""
    from deep_research.core.schema import Issue, Severity, FixType

    dimension_scores = {dim: 7.0 for dim in AttackDimension}
    sample_issue = Issue(
        dimension=AttackDimension.FACTUAL,
        severity=Severity.MAJOR,
        location="第1段",
        description="震荡测试问题",
        fix_type=FixType.IN_PLACE,
    )
    blue_agent = RecordingFakeBlueAgent()
    pool = FakeAgentPool(
        red_agent=FakeRedAgent(
            dimension_scores=dimension_scores,
            issues_per_dimension=[sample_issue],
        ),
        blue_agent=blue_agent,
    )

    config = {
        "adversarial": {
            "enabled": True,
            "max_rounds": 5,
            "score_threshold": 9.5,
            "entry_confidence_threshold": 0.5,
            "delta_threshold": 0.1,
            "min_severity_to_fix": "major",
            "enable_oscillation_check": True,
            "save_logs": False,
        },
        "system": {"work_dir": "./outputs"},
    }

    orch = Orchestrator(agent_pool=pool, config=config)
    context = ResearchContext(topic="AI safety", enable_adversarial=True)
    # confidence 低于入口阈值 0.5，进入对抗
    orch._report = ResearchReport(query="AI safety", content="Test report", confidence=0.3)
    orch._state = WorkflowState.ADVERSARIAL

    final_state = await orch._do_adversarial()
    assert final_state == WorkflowState.DONE
    # Round 1: 进入修复；Round 2: 发现相同 issue，震荡终止
    assert orch._report.adversarial_rounds == 2
    assert orch._report.adversarial_history[-1].get("terminated_by_oscillation") is True


@pytest.mark.asyncio
async def test_builtin_adversarial_restores_best_scoring_report():
    """分数先升后降时，应回退到历史最佳版本交付，final_score 与制品一致。"""
    from deep_research.core.schema import Issue, Severity, FixType

    issue = Issue(
        dimension=AttackDimension.FACTUAL,
        severity=Severity.MAJOR,
        location="第1段",
        description="需要修复的问题",
        fix_type=FixType.IN_PLACE,
    )

    class IssuesUntilRound2RedAgent(FakeRedAgent):
        def __init__(self):
            super().__init__()
            self.round_scores = [6.0, 8.0, 7.0]

        async def run(self, task, context):
            round_no = context.get("round_no", 1)
            dimension = context.get("dimension")
            score = self.round_scores[min(round_no - 1, len(self.round_scores) - 1)]
            issues = [issue] if round_no <= 2 else []
            dim_attack = DimensionAttack(
                dimension=dimension,
                dimension_score=score,
                issues=list(issues),
            )
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.SUCCESS,
                output=dim_attack,
                trajectory=[],
                token_usage=0,
                confidence=score / 10.0,
            )

    class MutatingFakeBlueAgent:
        def __init__(self):
            self.calls = 0

        async def run(self, task, context):
            import copy as _copy

            report = _copy.deepcopy(context.get("report"))
            self.calls += 1
            report.content += f" fix{self.calls}"
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.SUCCESS,
                output=report,
                trajectory=[],
                token_usage=0,
                confidence=report.confidence,
            )

    pool = FakeAgentPool(
        red_agent=IssuesUntilRound2RedAgent(),
        blue_agent=MutatingFakeBlueAgent(),
    )
    config = {
        "adversarial": {
            "enabled": True,
            "max_rounds": 5,
            "score_threshold": 9.5,
            "entry_confidence_threshold": 0.5,
            "delta_threshold": 0.1,
            "min_severity_to_fix": "major",
            "enable_oscillation_check": False,
            "save_logs": False,
        },
        "system": {"work_dir": "./outputs"},
    }

    orch = Orchestrator(agent_pool=pool, config=config)
    orch._report = ResearchReport(query="AI safety", content="Test report", confidence=0.3)
    orch._state = WorkflowState.ADVERSARIAL

    final_state = await orch._do_adversarial()

    assert final_state == WorkflowState.DONE
    # Round 3 评分 7.0 < 上轮 8.0，提升为负触发收敛；最佳版本是第 2 轮的 8.0
    assert orch._report.adversarial_history[-1].get("restored_best") is True
    assert orch._report.final_score == 8.0
    # 交付的是第 2 轮评分对应的报告（只含第 1 轮修复），不是最终轮改坏的版本
    assert orch._report.content == "Test report fix1"
