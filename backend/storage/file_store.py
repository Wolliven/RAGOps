"""Functions for reading RAGOps data from disk."""

import json

from backend.core.config import EMBEDDINGS_DIR


def load_all_embedded_chunks() -> list[dict]:
    """Load embedded chunks from every saved JSON file."""

    all_chunks = []

    for embeddings_path in EMBEDDINGS_DIR.glob("*.json"):
        json_text = embeddings_path.read_text(encoding="utf-8")
        chunks = json.loads(json_text)
        all_chunks.extend(chunks)

    return all_chunks