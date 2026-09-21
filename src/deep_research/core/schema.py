"""Core data models for the deep research workflow."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowState(str, Enum):
    """9-state finite state machine for deep research workflow."""

    IDLE = "idle"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    COLLECTING = "collecting"
    SYNTHESIZING = "synthesizing"
    ADVERSARIAL = "adversarial"
    REPLANNING = "replanning"
    DONE = "done"
    FAILED = "failed"


class SubTask(BaseModel):
    """A single sub-task in the research plan."""

    id: str
    description: str
    task_type: str = "search"  # search | analyze | verify | synthesize | red_agent | blue_agent
    dependencies: List[str] = Field(default_factory=list)
    search_hints: List[str] = Field(default_factory=list)
    timeout_seconds: int = 360
    priority: int = 1
    expected_type: str = "factual"  # factual | opinion | comparison
    status: str = "pending"  # pending | running | done | failed
    result: Optional[Any] = None
    error: Optional[str] = None


class ResearchContext(BaseModel):
    """Shared context passed through all workflow states."""

    topic: str
    plan: Optional[Any] = None
    subtasks: List[SubTask] = Field(default_factory=list)
    report_draft: Optional[str] = None
    final_report: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 3
    enable_adversarial: bool = False  # 对抗开关，默认关闭
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class HandlerResult:
    """Result returned by a state handler — only describes execution outcome."""

    status: str  # success | partial_failure | failure
    data: Any = None
    message: str = ""


class AgentStatus(str, Enum):
    """Agent 执行状态。"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentResult:
    """Agent 执行子任务的结果。"""

    task_id: str
    status: AgentStatus
    output: Any
    trajectory: list[dict] = field(default_factory=list)
    token_usage: int = 0
    confidence: float = 0.0
    # 额外元数据，例如是否来自记忆召回、任务描述等。
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adversarial loop data models
# ---------------------------------------------------------------------------


class AttackDimension(str, Enum):
    """Red team 攻击维度。"""

    FACTUAL = "factual"
    HALLUCINATION = "hallucination"
    LOGIC = "logic"
    SOURCE = "source"
    COVERAGE = "coverage"


class Severity(str, Enum):
    """问题严重程度。"""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class FixType(str, Enum):
    """Blue team 修复类型，仅三种。"""

    IN_PLACE = "in_place"    # 原地修改：重写、降置信度、扩展、补反方、加待核实标注等
    SEARCH = "search"        # 需要进一步搜索/查证才能修复
    REMOVAL = "removal"      # 删除无依据论断


@dataclass
class Issue:
    """Red Agent 发现的单条问题。"""

    dimension: AttackDimension
    severity: Severity
    location: str
    description: str
    fix_type: FixType
    evidence: str = ""        # 攻击依据/原文引用/来源比对结果


@dataclass
class DimensionAttack:
    """单个维度的攻击结果。"""

    dimension: AttackDimension
    dimension_score: float = 0.0    # 0-10，10 分最好
    issues: List[Issue] = field(default_factory=list)
    analysis_summary: str = ""      # 该维度整体分析


@dataclass
class RedAttackResult:
    """一轮 Red 攻击的完整结果（包含 5 个维度）。"""

    round_no: int
    dimension_attacks: Dict[AttackDimension, DimensionAttack] = field(default_factory=dict)
    overall_score: float = 0.0
    overall_summary: str = ""

    # 维度权重：来源可信度排第一，事实/逻辑/覆盖并重，幻觉次之
    _DIMENSION_WEIGHTS = {
        AttackDimension.SOURCE: 0.25,
        AttackDimension.FACTUAL: 0.20,
        AttackDimension.LOGIC: 0.20,
        AttackDimension.COVERAGE: 0.20,
        AttackDimension.HALLUCINATION: 0.15,
    }

    def compute_overall_score(self) -> float:
        """加权平均五个维度的 dimension_score。"""
        total_weight = 0.0
        weighted_sum = 0.0
        for dim in AttackDimension:
            da = self.dimension_attacks.get(dim)
            if da is None:
                continue
            weight = self._DIMENSION_WEIGHTS.get(dim, 0.2)
            weighted_sum += da.dimension_score * weight
            total_weight += weight
        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 2)

    def outstanding_issues(self, min_severity: Severity = Severity.MAJOR) -> List[Issue]:
        """Blue 修复前过滤突出问题。

        默认只返回 critical 和 major。
        """
        severity_rank = {Severity.CRITICAL: 3, Severity.MAJOR: 2, Severity.MINOR: 1}
        min_rank = severity_rank.get(min_severity, 2)
        issues: List[Issue] = []
        for da in self.dimension_attacks.values():
            for issue in da.issues:
                if severity_rank.get(issue.severity, 0) >= min_rank:
                    issues.append(issue)
        return issues


@dataclass
class ResearchReport:
    """最终交付给用户的研究报告。

    由 SummarizerAgent 合成，可能经过 Adversarial Loop 优化。
    """

    query: str
    content: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    num_searches: int = 0
    num_replan: int = 0
    adversarial_rounds: int = 0
    final_score: float = 0.0
    adversarial_history: list[dict] = field(default_factory=list)
    dimension_scores: Dict[AttackDimension, float] = field(default_factory=dict)
