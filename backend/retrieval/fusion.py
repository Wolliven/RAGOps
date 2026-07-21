def reciprocal_rank_fusion(semantic_results: list[dict], bm25_results: list[dict], top_k: int = 5, rrf_k: int = 60) -> list[dict]:
    """
    Combines semantic and BM25 rankings using Reciprocal Rank Fusion.
    """
    fused_results = {}

    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = result["chunk_id"]

        fused_results[chunk_id] = result.copy()
        fused_results[chunk_id]["rrf_score"] = 1 / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in fused_results:
            fused_results[chunk_id] = result.copy()
            fused_results[chunk_id]["rrf_score"] = 0
        else:
            fused_results[chunk_id]["bm25_score"] = result["bm25_score"]

        fused_results[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    ranked_results = sorted(
        fused_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True
    )

    return ranked_results[:top_k]