from deep_research.core.report_content import (
    invalid_citation_ids,
    prepare_sources,
    remove_invalid_citations,
    strip_reference_sections,
)


def test_prepare_sources_deduplicates_and_numbers_urls():
    sources = prepare_sources([
        {"title": "A", "url": "https://example.com/a"},
        {"title": "A duplicate", "url": "https://example.com/a"},
        {"title": "empty", "url": ""},
        {"title": "B", "url": "https://example.com/b"},
    ])

    assert [(source["citation_id"], source["title"]) for source in sources] == [
        (1, "A"),
        (2, "B"),
    ]


def test_invalid_citations_are_detected_and_removed():
    sources = prepare_sources([{"title": "A", "url": "https://example.com/a"}])
    content = "支持的结论 [1]，悬空引用 [9]。"

    assert invalid_citation_ids(content, sources) == {9}
    assert remove_invalid_citations(content, sources) == "支持的结论 [1]，悬空引用 。"


def test_reference_section_is_removed_but_inline_citations_remain():
    content = "正文结论 [1]。\n\n## 参考文献\n1. Example"

    assert strip_reference_sections(content) == "正文结论 [1]。"
