from research_agent.rag.hybrid import BM25Index, reciprocal_rank_fusion, tokenize_for_search


def test_mixed_language_tokenizer_keeps_identifiers_and_chinese_bigrams():
    tokens = tokenize_for_search("LangGraph 支持父子索引 parent_id")
    assert "langgraph" in tokens
    assert "parent_id" in tokens
    assert "父子" in tokens
    assert "索引" in tokens


def test_bm25_recovers_exact_technical_term():
    index = BM25Index(
        ["A", "B"],
        ["普通的向量语义检索说明", "使用 reciprocal rank fusion 进行 RRF 融合"],
    )
    hits = index.search("RRF", top_k=2)
    assert hits[0].item_id == "B"
    assert hits[0].score > 0


def test_rrf_combines_dense_and_lexical_ranks_without_raw_score_comparison():
    fused = reciprocal_rank_fusion(
        {"dense": ["A", "B"], "bm25": ["B", "C"]},
        rank_constant=10,
    )
    assert fused["B"] > fused["A"]
    assert fused["B"] > fused["C"]

