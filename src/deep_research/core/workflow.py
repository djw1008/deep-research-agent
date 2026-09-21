"""Workflow constants, transition graph and exceptions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Set

from .schema import WorkflowState


# ---------------------------------------------------------------------------
# Transition graph
# ---------------------------------------------------------------------------
TRANSITIONS: Dict[WorkflowState, Set[WorkflowState]] = {
    WorkflowState.IDLE: {WorkflowState.PLANNING, WorkflowState.FAILED},
    WorkflowState.PLANNING: {WorkflowState.DISPATCHING, WorkflowState.FAILED},
    WorkflowState.DISPATCHING: {WorkflowState.COLLECTING, WorkflowState.FAILED},
    WorkflowState.COLLECTING: {
        WorkflowState.SYNTHESIZING,
        WorkflowState.REPLANNING,
        WorkflowState.FAILED,
    },
    WorkflowState.SYNTHESIZING: {
        WorkflowState.ADVERSARIAL,
        WorkflowState.DONE,
        WorkflowState.FAILED,
    },
    WorkflowState.ADVERSARIAL: {
        WorkflowState.DONE,
        WorkflowState.SYNTHESIZING,
        WorkflowState.FAILED,
    },
    WorkflowState.REPLANNING: {WorkflowState.PLANNING, WorkflowState.DISPATCHING, WorkflowState.FAILED},
    WorkflowState.DONE: set(),
    WorkflowState.FAILED: set(),
}


# ---------------------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------------------
@dataclass
class StateTransition:
    from_state: WorkflowState
    to_state: WorkflowState
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class InvalidTransitionError(Exception):
    """Raised when a state transition violates the FSM graph."""


class NoHandlerRegisteredError(Exception):
    """Raised when a state has no registered handler."""
