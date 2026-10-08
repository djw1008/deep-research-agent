from deep_research.core.citations import normalize_source_citations


def test_normalizes_readable_and_case_insensitive_source_labels() -> None:
    result = normalize_source_citations(
        "First [src-1 摘要], second [来源：SRC-2].",
        {"SRC-1", "SRC-2"},
    )

    assert result.text == "First [SRC-1], second [SRC-2]."
    assert result.cited_labels == {"SRC-1", "SRC-2"}
    assert result.invalid_labels == set()
    assert result.normalized_count == 2


def test_normalizes_multiple_labels_and_rejects_unknown_labels() -> None:
    result = normalize_source_citations(
        "Combined evidence [SRC-1, SRC-2] but not [SRC-999 摘要].",
        {"SRC-1", "SRC-2"},
    )

    assert result.text == "Combined evidence [SRC-1][SRC-2] but not [SRC-999]."
    assert result.cited_labels == {"SRC-1", "SRC-2"}
    assert result.invalid_labels == {"SRC-999"}


def test_leaves_non_citation_brackets_unchanged() -> None:
    result = normalize_source_citations("Array value [0] and note [摘要].", set())

    assert result.text == "Array value [0] and note [摘要]."
    assert result.normalized_count == 0


def test_normalizes_namespaced_dependency_label() -> None:
    result = normalize_source_citations(
        "Inherited evidence [task_1:SRC-3 摘要].",
        {"TASK_1:SRC-3"},
    )

    assert result.text == "Inherited evidence [TASK_1:SRC-3]."
    assert result.cited_labels == {"TASK_1:SRC-3"}
