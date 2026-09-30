from deep_research.core.schema import ResearchReport
from run import parse_report_markdown, save_report


def test_save_report_renders_one_numbered_reference_list(tmp_path):
    report = ResearchReport(
        query="test",
        content="正文 [1]，悬空引用 [9]。\n\n## 引用来源\n9. fake",
        sources=[
            {"title": "Source", "url": "https://example.com/source"},
            {"title": "Duplicate", "url": "https://example.com/source"},
        ],
        confidence=0.8,
    )

    path = save_report("test", report, {"system": {"work_dir": str(tmp_path)}})
    saved = path.read_text(encoding="utf-8")

    assert saved.count("## 参考链接") == 1
    assert "## 引用来源" not in saved
    assert "悬空引用" in saved and "[9]" not in saved
    assert "- [1] [Source](https://example.com/source)" in saved
    assert "Duplicate" not in saved

    restored = parse_report_markdown(path)
    assert restored.content == "正文 [1]，悬空引用 。"
    assert restored.sources[0]["citation_id"] == 1
