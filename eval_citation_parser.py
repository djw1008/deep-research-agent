#!/usr/bin/env python3
"""引用标记解析器的轻量消融评测。

对比：
1. 基线：只识别严格的 [SRC-n] / [TASK:SRC-n]
2. 项目解析器：normalize_source_citations

无需 LLM 或 API Key。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deep_research.core.citations import (  # noqa: E402
    CitationParseResult,
    SOURCE_LABEL_TOKEN,
    normalize_source_citations,
)


@dataclass(frozen=True)
class CitationCase:
    name: str
    text: str
    available_labels: set[str]
    expected_text: str
    expected_cited: set[str]
    expected_invalid: set[str]


CASES = [
    CitationCase("canonical", "Fact [SRC-1].", {"SRC-1"}, "Fact [SRC-1].", {"SRC-1"}, set()),
    CitationCase("lowercase", "Fact [src-1].", {"SRC-1"}, "Fact [SRC-1].", {"SRC-1"}, set()),
    CitationCase("annotation", "Fact [SRC-1 摘要].", {"SRC-1"}, "Fact [SRC-1].", {"SRC-1"}, set()),
    CitationCase("chinese_prefix", "Fact [来源：SRC-2].", {"SRC-2"}, "Fact [SRC-2].", {"SRC-2"}, set()),
    CitationCase(
        "multiple",
        "Combined [SRC-1, SRC-2].",
        {"SRC-1", "SRC-2"},
        "Combined [SRC-1][SRC-2].",
        {"SRC-1", "SRC-2"},
        set(),
    ),
    CitationCase(
        "namespaced",
        "Inherited [task_1:SRC-3 摘要].",
        {"TASK_1:SRC-3"},
        "Inherited [TASK_1:SRC-3].",
        {"TASK_1:SRC-3"},
        set(),
    ),
    CitationCase("invalid", "Unknown [SRC-99].", {"SRC-1"}, "Unknown [SRC-99].", set(), {"SRC-99"}),
    CitationCase(
        "invalid_annotation",
        "Unknown [SRC-99 说明].",
        {"SRC-1"},
        "Unknown [SRC-99].",
        set(),
        {"SRC-99"},
    ),
    CitationCase("duplicate", "Fact [SRC-1, SRC-1].", {"SRC-1"}, "Fact [SRC-1].", {"SRC-1"}, set()),
    CitationCase(
        "mixed_valid_invalid",
        "Mixed [SRC-1, SRC-99].",
        {"SRC-1"},
        "Mixed [SRC-1][SRC-99].",
        {"SRC-1"},
        {"SRC-99"},
    ),
    CitationCase("numeric_bracket", "Array [0] stays.", set(), "Array [0] stays.", set(), set()),
    CitationCase("plain_bracket", "Note [摘要] stays.", set(), "Note [摘要] stays.", set(), set()),
]


_BASELINE_PATTERN = re.compile(rf"\[({SOURCE_LABEL_TOKEN})\]")


def baseline_parse(text: str, available_labels: set[str]) -> CitationParseResult:
    """基线：仅识别格式完全正确的大写引用，不做文本修复。"""
    available = {label.upper() for label in available_labels}
    labels = set(_BASELINE_PATTERN.findall(text or ""))
    cited = {label for label in labels if label in available}
    invalid = labels - available
    return CitationParseResult(text or "", cited, invalid, 0)


def evaluate_parser(parser) -> tuple[dict, list[dict]]:
    details = []
    counters = {"text": 0, "cited": 0, "invalid": 0, "overall": 0}
    for case in CASES:
        result = parser(case.text, case.available_labels)
        checks = {
            "text": result.text == case.expected_text,
            "cited": result.cited_labels == case.expected_cited,
            "invalid": result.invalid_labels == case.expected_invalid,
        }
        checks["overall"] = all(checks.values())
        for key, passed in checks.items():
            counters[key] += int(passed)
        details.append({
            "name": case.name,
            "input": case.text,
            "output": result.text,
            "cited_labels": sorted(result.cited_labels),
            "invalid_labels": sorted(result.invalid_labels),
            "checks": checks,
        })

    count = len(CASES)
    metrics = {
        "num_cases": count,
        "text_normalization_accuracy": counters["text"] / count,
        "valid_label_accuracy": counters["cited"] / count,
        "invalid_label_accuracy": counters["invalid"] / count,
        "overall_case_accuracy": counters["overall"] / count,
    }
    return metrics, details


def run_evaluation() -> dict:
    baseline_metrics, baseline_details = evaluate_parser(baseline_parse)
    parser_metrics, parser_details = evaluate_parser(normalize_source_citations)
    return {
        "dataset": [
            {
                **asdict(case),
                "available_labels": sorted(case.available_labels),
                "expected_cited": sorted(case.expected_cited),
                "expected_invalid": sorted(case.expected_invalid),
            }
            for case in CASES
        ],
        "baseline": {"metrics": baseline_metrics, "details": baseline_details},
        "citation_parser": {"metrics": parser_metrics, "details": parser_details},
        "improvement": {
            key: round(parser_metrics[key] - baseline_metrics[key], 4)
            for key in parser_metrics
            if key != "num_cases"
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评测引用标记解析器")
    parser.add_argument("-o", "--output", default="outputs")
    args = parser.parse_args()

    result = run_evaluation()
    baseline = result["baseline"]["metrics"]
    current = result["citation_parser"]["metrics"]

    print("引用标记解析器评测")
    print(f"样本数: {current['num_cases']}")
    print(f"{'Metric':<32} {'Baseline':>10} {'Parser':>10} {'Delta':>10}")
    for key in (
        "text_normalization_accuracy",
        "valid_label_accuracy",
        "invalid_label_accuracy",
        "overall_case_accuracy",
    ):
        print(
            f"{key:<32} {baseline[key]:>10.2%} {current[key]:>10.2%} "
            f"{current[key] - baseline[key]:>+10.2%}"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"citation_parser_eval_{timestamp}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
