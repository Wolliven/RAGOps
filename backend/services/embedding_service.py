"""Service for loading the embedding model and embedding chunks."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from backend.core.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_chunks(chunk_data: list[dict]) -> list[dict]:
    """Generate an embedding for every chunk."""

    texts = []

    for chunk in chunk_data:
        texts.append(chunk["text"])

    model = get_embedding_model()
    embeddings = model.encode(texts)

    embedded_chunks = []

    for chunk, embedding in zip(chunk_data, embeddings):
        embedded_chunk = chunk.copy()
        embedded_chunk["embedding"] = embedding.tolist()
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks