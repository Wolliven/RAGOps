from backend.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_promotes_results_found_by_both_methods():
    semantic_results = [
        {
            "chunk_id": "A",
            "text": "A",
            "semantic_score": 0.9,
        },
        {
            "chunk_id": "B",
            "text": "B",
            "semantic_score": 0.8,
        },
    ]

    bm25_results = [
        {
            "chunk_id": "B",
            "text": "B",
            "bm25_score": 10.0,
        },
        {
            "chunk_id": "C",
            "text": "C",
            "bm25_score": 8.0,
        },
    ]

    results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        top_k=3,
    )

    assert results[0]["chunk_id"] == "B"
    assert "semantic_score" in results[0]
    assert "bm25_score" in results[0]


def test_rrf_respects_top_k():
    semantic_results = [
        {"chunk_id": "A", "text": "A"},
        {"chunk_id": "B", "text": "B"},
    ]

    bm25_results = [
        {"chunk_id": "C", "text": "C", "bm25_score": 1.0},
        {"chunk_id": "D", "text": "D", "bm25_score": 0.5},
    ]

    results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        top_k=2,
    )

    assert len(results) == 2