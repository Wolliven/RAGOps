"""Service for running RAGOps search workflows."""

from backend.retrieval.bm25 import (
    build_bm25_index,
    search_bm25,
)
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.semantic import search_chunks
from backend.services.embedding_service import get_embedding_model
from backend.storage.file_store import load_all_embedded_chunks


class NoIndexedChunksError(LookupError):
    """Raised when a search is attempted without indexed documents."""


def bm25_search(
    query: str,
    top_k: int,
) -> dict:
    """Run a BM25-only search."""

    chunks = _load_indexed_chunks()
    normalized_top_k = _normalize_top_k(top_k)

    retriever = build_bm25_index(chunks)

    results = search_bm25(
        query=query,
        retriever=retriever,
        top_k=normalized_top_k,
    )

    return {
        "query": query,
        "results": results,
    }


def compare_search_methods(
    query: str,
    top_k: int,
) -> dict:
    """Return semantic, BM25, and hybrid results for comparison."""

    chunks = _load_indexed_chunks()
    normalized_top_k = _normalize_top_k(top_k)

    semantic_results = search_chunks(
        query=query,
        embedded_chunks=chunks,
        model=get_embedding_model(),
        top_k=normalized_top_k,
    )

    bm25_retriever = build_bm25_index(chunks)

    bm25_results = search_bm25(
        query=query,
        retriever=bm25_retriever,
        top_k=normalized_top_k,
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        top_k=top_k,
    )

    return {
        "query": query,
        "semantic_results": semantic_results,
        "bm25_results": bm25_results,
        "hybrid_results": fused_results,
    }


def hybrid_search(
    query: str,
    top_k: int,
) -> dict:
    """Run semantic and BM25 retrieval, then fuse the rankings."""

    chunks = _load_indexed_chunks()
    normalized_top_k = _normalize_top_k(top_k)

    semantic_results = search_chunks(
        query=query,
        embedded_chunks=chunks,
        model=get_embedding_model(),
        top_k=20,
    )

    bm25_retriever = build_bm25_index(chunks)

    bm25_results = search_bm25(
        query=query,
        retriever=bm25_retriever,
        top_k=20,
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        top_k=normalized_top_k,
    )

    return {
        "query": query,
        "results": fused_results,
    }


def _load_indexed_chunks() -> list[dict]:
    """Load indexed chunks or raise a domain-specific error."""

    chunks = load_all_embedded_chunks()

    if not chunks:
        raise NoIndexedChunksError

    return chunks


def _normalize_top_k(top_k: int) -> int:
    """Limit the requested result count to the supported range."""

    return min(max(top_k, 1), 20)