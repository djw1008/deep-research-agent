from deep_research.core.report_content import (
    citation_ids,
    compact_cited_sources,
    invalid_citation_ids,
    prepare_sources,
    remap_citation_ids,
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


def test_citation_ids_can_be_extracted_and_remapped():
    content = "已有 [1]，候选 [4]，重复候选 [4]。"

    assert citation_ids(content) == {1, 4}
    assert remap_citation_ids(content, {4: 2}) == "已有 [1]，候选 [2]，重复候选 [2]。"


def test_compact_cited_sources_removes_unused_and_renumbers_by_appearance():
    sources = prepare_sources([
        {"title": "A", "url": "https://example.com/a"},
        {"title": "B", "url": "https://example.com/b"},
        {"title": "C", "url": "https://example.com/c"},
    ])

    content, compacted = compact_cited_sources(
        "先引用 C [3]，再引用 A [1]，无效引用 [9]。", sources
    )

    assert content == "先引用 C [1]，再引用 A [2]，无效引用 。"
    assert [(source["citation_id"], source["title"]) for source in compacted] == [
        (1, "C"),
        (2, "A"),
    ]
