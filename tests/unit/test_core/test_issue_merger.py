"""Tests for IssueMerger (VerdictEngine) behavior."""

import pytest

from deep_research.core.issue_merger import IssueMerger
from deep_research.core.schema import AttackDimension, FixType, Issue, Severity


def _issue(
    dimension: AttackDimension = AttackDimension.FACTUAL,
    severity: Severity = Severity.MAJOR,
    location: str = "第1段",
    description: str = "desc",
    fix_type: FixType = FixType.IN_PLACE,
    evidence: str = "evidence",
) -> Issue:
    return Issue(
        dimension=dimension,
        severity=severity,
        location=location,
        description=description,
        fix_type=fix_type,
        evidence=evidence,
    )


@pytest.mark.asyncio
async def test_empty_merge_returns_empty():
    assert await IssueMerger.merge_issues([]) == []


@pytest.mark.asyncio
async def test_duplicate_issues_are_merged():
    issues = [
        _issue(description="the model hallucinates a source"),
        _issue(description="the model hallucinates a source"),
    ]
    merged = await IssueMerger.merge_issues(issues)
    assert len(merged) == 1
    assert merged[0].description == "the model hallucinates a source"


@pytest.mark.asyncio
async def test_merge_keeps_most_severe_and_most_conservative_fix():
    issues = [
        _issue(severity=Severity.MINOR, fix_type=FixType.IN_PLACE),
        _issue(severity=Severity.CRITICAL, fix_type=FixType.REMOVAL),
        _issue(severity=Severity.MAJOR, fix_type=FixType.SEARCH),
    ]
    merged = await IssueMerger.merge_issues(issues)
    assert len(merged) == 1
    assert merged[0].severity == Severity.CRITICAL
    assert merged[0].fix_type == FixType.REMOVAL


@pytest.mark.asyncio
async def test_merge_prefers_base_dimension():
    issues = [
        _issue(dimension=AttackDimension.COVERAGE),
        _issue(dimension=AttackDimension.HALLUCINATION),
        _issue(dimension=AttackDimension.LOGIC),
    ]
    merged = await IssueMerger.merge_issues(issues)
    assert merged[0].dimension == AttackDimension.HALLUCINATION


@pytest.mark.asyncio
async def test_merge_evidences_are_concatenated():
    issues = [
        _issue(evidence="evidence one"),
        _issue(evidence="evidence two"),
        _issue(evidence="evidence one"),  # duplicate ignored
    ]
    merged = await IssueMerger.merge_issues(issues)
    assert "evidence one" in merged[0].evidence
    assert "evidence two" in merged[0].evidence


@pytest.mark.asyncio
async def test_similar_descriptions_with_different_locations_are_not_merged():
    issues = [
        _issue(location="第1段", description="unsupported claim"),
        _issue(location="第2段", description="unsupported claim"),
    ]
    merged = await IssueMerger.merge_issues(issues)
    assert len(merged) == 2


def test_jaccard_similarity_groups_descriptions():
    # Identical descriptions should match.
    assert IssueMerger._jaccard_similarity("foo bar", "foo bar") == 1.0
    # Half overlap: {foo, bar} vs {foo, baz} -> intersection 1, union 3.
    assert IssueMerger._jaccard_similarity("foo bar", "foo baz") == pytest.approx(1 / 3)
    # Dissimilar should fall below threshold.
    assert IssueMerger._jaccard_similarity("foo bar", "baz qux") == 0.0


@pytest.mark.asyncio
async def test_conflict_arbitration_removes_coverage_supplement_when_hallucination_removal_exists():
    issues = [
        _issue(
            dimension=AttackDimension.HALLUCINATION,
            fix_type=FixType.REMOVAL,
            location="第1段",
            description="hallucinated claim should be removed",
        ),
        _issue(
            dimension=AttackDimension.COVERAGE,
            fix_type=FixType.SEARCH,
            location="第1段",
            description="coverage gap needs more sources",
        ),
    ]
    merged = await IssueMerger.merge_issues(issues)
    dimensions = {i.dimension for i in merged}
    assert AttackDimension.COVERAGE not in dimensions
    assert len(merged) == 1
    assert merged[0].dimension == AttackDimension.HALLUCINATION


@pytest.mark.asyncio
async def test_sorting_orders_by_dimension_priority_then_severity():
    issues = [
        _issue(
            dimension=AttackDimension.COVERAGE,
            severity=Severity.CRITICAL,
            description="coverage gap",
            location="",
        ),
        _issue(
            dimension=AttackDimension.HALLUCINATION,
            severity=Severity.MINOR,
            description="hallucination issue",
            location="",
        ),
        _issue(
            dimension=AttackDimension.FACTUAL,
            severity=Severity.MAJOR,
            description="factual issue",
            location="",
        ),
    ]
    merged = await IssueMerger.merge_issues(issues)
    assert [i.dimension for i in merged] == [
        AttackDimension.HALLUCINATION,
        AttackDimension.FACTUAL,
        AttackDimension.COVERAGE,
    ]


@pytest.mark.asyncio
async def test_llm_arbitration_filters_false_positives():
    """LLM 仲裁应能剔除明显误报，并保留真实 issue。"""

    class FakeArbiterClient:
        def chat(self, messages):
            from deep_research.models.llm_client import LLMResponse

            return LLMResponse(
                content='[{"dimension": "factual", "severity": "major", "location": "第1段", '
                '"description": "real factual error", "fix_type": "removal", "evidence": "no source"}]'
            )

    issues = [
        _issue(description="real factual error"),
        _issue(description="false positive that does not exist"),
    ]
    merged = await IssueMerger.merge_issues(
        issues,
        llm_client=FakeArbiterClient(),
        query="test query",
        report_content="report content with real factual error",
    )
    assert len(merged) == 1
    assert merged[0].description == "real factual error"
    assert merged[0].fix_type == FixType.REMOVAL
