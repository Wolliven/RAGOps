"""
Semantic retrieval utilities.

This module creates query embeddings, calculates cosine similarity,
and ranks document chunks by semantic relevance.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_scores(query_embedding, chunk_embeddings):
    """
    Calculates cosine similarity between one query vector
    and a collection of chunk vectors.
    """
    query_vector = np.array(query_embedding, dtype=np.float32)
    chunk_matrix = np.array(chunk_embeddings, dtype=np.float32)

    if chunk_matrix.size == 0:
        return np.array([])

    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1)

    denominator = chunk_norms * query_norm
    denominator = np.where(denominator == 0, 1e-10, denominator)

    scores = (chunk_matrix @ query_vector) / denominator

    return scores

def search_chunks(query: str, embedded_chunks: list[dict], model: SentenceTransformer, top_k: int = 3) -> list[dict]:
    """
    Returns the top-k chunks ranked by semantic similarity to the query.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not embedded_chunks:
        return []

    query_embedding = model.encode(query)

    chunk_embeddings = []

    for chunk in embedded_chunks:
        chunk_embeddings.append(chunk["embedding"])

    scores = cosine_scores(query_embedding, chunk_embeddings)

    results = []

    for chunk, score in zip(embedded_chunks, scores):
        result = {
            "source_file": chunk.get("source_file"),
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "chunk_index": chunk.get("chunk_index"),
            "semantic_score": float(score),
            "text": chunk.get("text", "")
        }

        results.append(result)

    results.sort(key=lambda item: item["semantic_score"], reverse=True)

    return results[:top_k]

