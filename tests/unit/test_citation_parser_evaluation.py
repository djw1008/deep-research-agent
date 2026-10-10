from eval_citation_parser import run_evaluation


def test_citation_parser_evaluation_shows_improvement():
    result = run_evaluation()
    baseline = result["baseline"]["metrics"]
    parser = result["citation_parser"]["metrics"]

    assert parser["overall_case_accuracy"] == 1.0
    assert parser["overall_case_accuracy"] > baseline["overall_case_accuracy"]
    assert parser["text_normalization_accuracy"] > baseline["text_normalization_accuracy"]
    assert parser["valid_label_accuracy"] > baseline["valid_label_accuracy"]
    assert parser["invalid_label_accuracy"] > baseline["invalid_label_accuracy"]
