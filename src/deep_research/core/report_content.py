"""Helpers for keeping generated report bodies separate from managed metadata."""

from __future__ import annotations

import re
from typing import Any


_REFERENCE_HEADING = re.compile(
    r"(?im)^\s{0,3}#{1,6}\s*(?:引用来源|参考来源|参考文献|参考链接|"
    r"references|bibliography|sources)\s*$"
)


def strip_reference_sections(content: str) -> str:
    """Remove an LLM-generated trailing bibliography from a report body.

    Source metadata is maintained in ``ResearchReport.sources`` and rendered by
    the output layer. Keeping a second free-form bibliography in ``content``
    causes duplicate lists and allows dangling source numbers.
    """
    match = _REFERENCE_HEADING.search(content or "")
    if match is None:
        return content
    return content[: match.start()].rstrip()


_CITATION = re.compile(r"\[(\d+)\]")


def prepare_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate sources by URL and assign stable sequential citation numbers."""
    prepared: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for source in sources or []:
        url = str(source.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        item = dict(source)
        item["url"] = url
        item["citation_id"] = len(prepared) + 1
        prepared.append(item)
    return prepared


def invalid_citation_ids(content: str, sources: list[dict[str, Any]]) -> set[int]:
    """Return numeric citations in content that do not exist in the registry."""
    valid_ids = {int(source["citation_id"]) for source in sources if source.get("citation_id")}
    used_ids = citation_ids(content)
    return used_ids - valid_ids


def citation_ids(content: str) -> set[int]:
    """Return all numeric citation identifiers used in report content."""
    return {int(match) for match in _CITATION.findall(content or "")}


def remap_citation_ids(content: str, mapping: dict[int, int]) -> str:
    """Rewrite selected numeric citation identifiers without touching others."""
    if not mapping:
        return content
    return _CITATION.sub(
        lambda match: f"[{mapping.get(int(match.group(1)), int(match.group(1)))}]",
        content,
    )


def remove_invalid_citations(content: str, sources: list[dict[str, Any]]) -> str:
    """Remove dangling numeric citations while preserving valid registered ones."""
    invalid = invalid_citation_ids(content, sources)
    if not invalid:
        return content
    return _CITATION.sub(
        lambda match: "" if int(match.group(1)) in invalid else match.group(0),
        content,
    )
