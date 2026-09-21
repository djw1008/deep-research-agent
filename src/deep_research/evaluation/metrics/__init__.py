"""Evaluation metrics.

RuleBasedMetrics: 7 fast, API-free metrics for batch evaluation / CI / ablation.
Stats: statistical significance tests (bootstrap CI, Cohen's d, t-test).
"""

from .rule_based import RuleBasedMetrics
from .stats import (
    bootstrap_ci_paired,
    bootstrap_ci_two_sample,
    cohens_d,
    paired_t_test,
)

__all__ = [
    "RuleBasedMetrics",
    "bootstrap_ci_paired",
    "bootstrap_ci_two_sample",
    "cohens_d",
    "paired_t_test",
]
