"""Deterministic parsing and normalization for researcher source citations."""

from __future__ import annotations

import re
from dataclasses import dataclass


_BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)]")
# A downstream agent may cite a dependency as ``task_1:SRC-3``.  The task
# namespace prevents identical local labels from different researchers from
# becoming ambiguous.
SOURCE_LABEL_TOKEN = r"(?:[A-Za-z0-9_.-]+:)?SRC-\d+"
_SOURCE_LABEL_PATTERN = re.compile(rf"\b{SOURCE_LABEL_TOKEN}\b", re.IGNORECASE)


@dataclass(frozen=True)
class CitationParseResult:
    """Normalized text and citation labels validated against known sources."""

    text: str
    cited_labels: set[str]
    invalid_labels: set[str]
    normalized_count: int = 0


def normalize_source_citations(
    text: str,
    available_labels: set[str],
) -> CitationParseResult:
    """Normalize bracketed ``SRC-n`` variants and validate their labels.

    LLMs occasionally emit human-readable variants such as ``[SRC-3 摘要]``.
    The harness accepts those variants, but a label is considered cited only
    when it was actually assigned to a source seen by the current agent.
    """

    available = {label.upper() for label in available_labels}
    cited_labels: set[str] = set()
    invalid_labels: set[str] = set()
    normalized_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal normalized_count

        labels = [
            label.upper()
            for label in _SOURCE_LABEL_PATTERN.findall(match.group(1))
        ]
        if not labels:
            return match.group(0)

        # Preserve first occurrence order while removing repeated labels.
        labels = list(dict.fromkeys(labels))
        for label in labels:
            if label in available:
                cited_labels.add(label)
            else:
                invalid_labels.add(label)

        normalized = "".join(f"[{label}]" for label in labels)
        if normalized != match.group(0):
            normalized_count += 1
        return normalized

    normalized_text = _BRACKET_PATTERN.sub(replace, text or "")
    return CitationParseResult(
        text=normalized_text,
        cited_labels=cited_labels,
        invalid_labels=invalid_labels,
        normalized_count=normalized_count,
    )
