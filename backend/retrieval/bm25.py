"""
BM25 lexical retrieval.

This module builds a BM25 index and ranks chunks
using exact term matches and term importance.
"""
import bm25s

def build_bm25_index(chunks: list[dict]):
    """
    Builds a BM25 index from an existing chunk corpus.
    """
    texts = [chunk["text"] for chunk in chunks]

    tokenized_texts = bm25s.tokenize(texts)

    retriever = bm25s.BM25(corpus=chunks)

    retriever.index(tokenized_texts)

    return retriever

def search_bm25(query: str, retriever, top_k: int = 20) -> list[dict]:
    """
    Returns the top-k chunks ranked by lexical relevance to the query.
    """
    tokenized_query = bm25s.tokenize([query])

    results, scores = retriever.retrieve(
        tokenized_query,
        k=top_k
    )

    ranked_chunks = []

    for chunk, score in zip(results[0], scores[0]):
        result = chunk.copy()
        result.pop("embedding", None)
        result["bm25_score"] = float(score)
        ranked_chunks.append(result)

    return ranked_chunks