"""Helpers for keeping generated report bodies separate from managed metadata."""

from __future__ import annotations

import re


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
