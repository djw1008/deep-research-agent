from deep_research.compressor.context_compressor import ContextCompressor


def test_l2_tokenizer_preserves_versions_and_decimal_numbers():
    compressor = ContextCompressor(l2_min_para_length=1)
    text = (
        "Qwen2.5-72B 的 MMLU 从 84.2 提升到 86.1。"
        "Qwen2.5-Coder 的 LiveCodeBench 得分为 55.5。"
    )

    sentences = compressor._l2_tokenize_sentences(text)

    assert len(sentences) == 2
    assert "Qwen2.5-72B" in sentences[0]
    assert "84.2" in sentences[0]
    assert "86.1" in sentences[0]
    assert "Qwen2.5-Coder" in sentences[1]
    assert "55.5" in sentences[1]


def test_l2_tokenizer_keeps_newline_table_as_one_semantic_unit():
    compressor = ContextCompressor(l2_min_para_length=1)
    text = (
        "模型基础信息。\n"
        "模型\n参数量\n层数\nQwen2.5-72B\n72.7B\n80\n128K\n8K\nQwen\n"
        "模型表现。"
    )

    sentences = compressor._l2_tokenize_sentences(text)

    assert "Qwen2.5-72B\n72.7B\n80\n128K\n8K\nQwen" in sentences[1]
    assert not any(sentence in {"5-72B", "7B", "80"} for sentence in sentences)
