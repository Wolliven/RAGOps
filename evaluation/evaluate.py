"""Run retrieval evaluation against the ground-truth query set."""

import json
from pathlib import Path

from backend.services.search_service import (
    semantic_search,
    bm25_search,
    hybrid_search,
)


QUERIES_PATH = Path(__file__).parent / "queries.json"
TOP_K = 5


def load_queries() -> list[dict]:
    """Load the manually labelled evaluation queries."""

    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_first_relevant_rank(
    results: list[dict],
    relevant_chunk_ids: list[str],
) -> int | None:
    """Return the rank of the first relevant result."""

    relevant_ids = set(relevant_chunk_ids)

    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in relevant_ids:
            return rank

    return None

def calculate_hit_at_k(
    ranks: list[int | None],
    k: int,
) -> float:
    """Calculate the proportion of queries with a relevant result in top-k."""

    hits = sum(
        1
        for rank in ranks
        if rank is not None and rank <= k
    )

    return hits / len(ranks)


def calculate_mrr(
    ranks: list[int | None],
) -> float:
    """Calculate mean reciprocal rank."""

    reciprocal_ranks = [
        1 / rank if rank is not None else 0
        for rank in ranks
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def main() -> None:
    queries = load_queries()

    method_ranks = {
        "Semantic": [],
        "BM25": [],
        "Hybrid": [],
    }

    evaluation_document_ids = sorted({
        query["document_id"]
        for query in queries
    })

    print(f"Loaded {len(queries)} evaluation queries.")
    print(
        "Evaluation documents:",
        ", ".join(evaluation_document_ids),
    )
    print()

    for query_data in queries:
        query = query_data["query"]
        relevant_chunk_ids = query_data["relevant_chunk_ids"]

        semantic_results = semantic_search(
            query=query,
            top_k=TOP_K,
            document_ids=evaluation_document_ids,
        )["results"]

        bm25_results = bm25_search(
            query=query,
            top_k=TOP_K,
            document_ids=evaluation_document_ids,
        )["results"]

        hybrid_results = hybrid_search(
            query=query,
            top_k=TOP_K,
            document_ids=evaluation_document_ids,
        )["results"]

        semantic_rank = find_first_relevant_rank(
            semantic_results,
            relevant_chunk_ids,
        )

        bm25_rank = find_first_relevant_rank(
            bm25_results,
            relevant_chunk_ids,
        )

        hybrid_rank = find_first_relevant_rank(
            hybrid_results,
            relevant_chunk_ids,
        )

        method_ranks["Semantic"].append(semantic_rank)
        method_ranks["BM25"].append(bm25_rank)
        method_ranks["Hybrid"].append(hybrid_rank)

        print(
            f'{query_data["id"]} '
            f'[{query_data["category"]}] '
            f'{query}'
        )

        print(
            f"  Semantic: {semantic_rank or '-'} | "
            f"BM25: {bm25_rank or '-'} | "
            f"Hybrid: {hybrid_rank or '-'}"
        )

        print()
        
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    for method_name, ranks in method_ranks.items():
        hit_at_1 = calculate_hit_at_k(ranks, 1)
        hit_at_3 = calculate_hit_at_k(ranks, 3)
        hit_at_5 = calculate_hit_at_k(ranks, 5)
        mrr = calculate_mrr(ranks)

        print()
        print(method_name)
        print(f"  Hit@1: {hit_at_1:.3f}")
        print(f"  Hit@3: {hit_at_3:.3f}")
        print(f"  Hit@5: {hit_at_5:.3f}")
        print(f"  MRR@5: {mrr:.3f}")


if __name__ == "__main__":
    main()