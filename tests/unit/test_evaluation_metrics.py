from deep_research.evaluation.metrics.rule_based import RuleBasedMetrics


def _sources(*citation_ids: int) -> list[dict]:
    return [
        {"citation_id": citation_id, "url": f"https://example.com/{citation_id}"}
        for citation_id in citation_ids
    ]


def test_source_adequacy_rewards_valid_distributed_citations():
    report = "\n".join(
        [
            "这是第一个详细正文段落，其中包含需要来源支持的事实性陈述，并且使用了已注册的引用编号[1]。",
            "这是第二个详细正文段落，同样包含需要来源支持的内容，并且引用另一个已注册的来源[2]。",
        ]
    )
    score = RuleBasedMetrics.source_adequacy(report, _sources(1, 2))
    assert score > 0.8


def test_source_adequacy_penalizes_dangling_citation_ids():
    paragraph = "这是一个足够长的正文段落，用于检查正文中不存在于来源元数据的悬空引用编号不会被当成有效引用[99]。"
    score = RuleBasedMetrics.source_adequacy(paragraph, _sources(1))
    assert score == 0.0


def test_source_adequacy_ignores_program_appended_reference_section():
    report = "\n".join(
        [
            "这是一个没有内联引用的详细正文段落，虽然内容足够长，但末尾链接不能代替正文中对具体论断的引用。",
            "## 参考链接",
            "- [1] https://example.com/1",
        ]
    )
    score = RuleBasedMetrics.source_adequacy(report, _sources(1))
    assert score == 0.0


def test_source_adequacy_uses_only_sources_actually_cited():
    paragraph = "这是一个足够长的正文段落，其中只实际引用了第一个来源，其余来源即使存在于元数据中也不应提高来源密度[1]。"
    sparse = RuleBasedMetrics.source_adequacy(paragraph, _sources(1))
    padded = RuleBasedMetrics.source_adequacy(paragraph, _sources(1, 2, 3, 4, 5))
    assert sparse == padded
