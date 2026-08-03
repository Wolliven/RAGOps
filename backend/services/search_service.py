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

class EmptyDocumentSelectionError(ValueError):
    """Raised when the document selection is empty."""

class NoMatchingDocumentsError(LookupError):
    """Raised when none of the selected documents are indexed."""


def bm25_search(
    query: str,
    top_k: int,
    document_ids: list[str] | None = None,
) -> dict:
    """Run a BM25-only search."""

    chunks = chunks = _load_indexed_chunks(document_ids)
    effective_top_k = _limit_top_k_to_corpus(
        top_k=top_k,
        corpus_size=len(chunks),
    )

    retriever = build_bm25_index(chunks)

    results = search_bm25(
        query=query,
        retriever=retriever,
        top_k=effective_top_k,
    )

    return {
        "query": query,
        "results": results,
    }


def compare_search_methods(
    query: str,
    top_k: int,
    document_ids: list[str] | None = None,
) -> dict:
    """Return semantic, BM25, and hybrid results for comparison."""

    chunks = _load_indexed_chunks(document_ids)
    effective_top_k = _limit_top_k_to_corpus(
        top_k=top_k,
        corpus_size=len(chunks),
    )

    semantic_results = search_chunks(
        query=query,
        embedded_chunks=chunks,
        model=get_embedding_model(),
        top_k=effective_top_k,
    )

    bm25_retriever = build_bm25_index(chunks)

    bm25_results = search_bm25(
        query=query,
        retriever=bm25_retriever,
        top_k=effective_top_k,
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        top_k=effective_top_k,
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
    document_ids: list[str] | None = None,
) -> dict:
    """Run semantic and BM25 retrieval, then fuse the rankings."""

    chunks = _load_indexed_chunks(document_ids)
    candidate_top_k = _limit_top_k_to_corpus(
        top_k=20,
        corpus_size=len(chunks),
    )

    result_top_k = _limit_top_k_to_corpus(
        top_k=top_k,
        corpus_size=len(chunks),
    )

    semantic_results = search_chunks(
        query=query,
        embedded_chunks=chunks,
        model=get_embedding_model(),
        top_k=candidate_top_k,
    )

    bm25_retriever = build_bm25_index(chunks)

    bm25_results = search_bm25(
        query=query,
        retriever=bm25_retriever,
        top_k=candidate_top_k,
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        top_k=result_top_k,
    )

    return {
        "query": query,
        "results": fused_results,
    }


def _load_indexed_chunks(
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Load indexed chunks and optionally filter them by document."""

    chunks = load_all_embedded_chunks()

    if not chunks:
        raise NoIndexedChunksError

    if document_ids is None:
        return chunks

    if not document_ids:
        raise EmptyDocumentSelectionError

    selected_document_ids = set(document_ids)

    filtered_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("document_id") in selected_document_ids
    ]

    if not filtered_chunks:
        raise NoMatchingDocumentsError

    return filtered_chunks


def _normalize_top_k(top_k: int) -> int:
    """Limit the requested result count to the supported range."""

    return min(max(top_k, 1), 20)

def _limit_top_k_to_corpus(
    top_k: int,
    corpus_size: int,
) -> int:
    """Ensure top_k does not exceed the number of available chunks."""

    return min(
        _normalize_top_k(top_k),
        corpus_size,
    )