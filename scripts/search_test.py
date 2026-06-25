from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path("data/embeddings/README.json")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_embedded_chunks(path: Path) -> list[dict]:
    json_text = path.read_text(encoding="utf-8")
    return json.loads(json_text)


def cosine_scores(query_embedding, chunk_embeddings):
    query_vector = np.array(query_embedding, dtype=np.float32)
    chunk_matrix = np.array(chunk_embeddings, dtype=np.float32)

    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1)

    scores = (chunk_matrix @ query_vector) / (chunk_norms * query_norm)

    return scores


def search_chunks(query: str, embedded_chunks: list[dict], top_k: int = 3) -> list[dict]:
    query_embedding = embedding_model.encode(query)

    chunk_embeddings = []

    for chunk in embedded_chunks:
        chunk_embeddings.append(chunk["embedding"])

    scores = cosine_scores(query_embedding, chunk_embeddings)

    results = []

    for chunk, score in zip(embedded_chunks, scores):
        result = chunk.copy()
        result["score"] = float(score)
        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    embedded_chunks = load_embedded_chunks(EMBEDDINGS_PATH)

    query = "What is the tech stack of this project?"

    results = search_chunks(query, embedded_chunks, top_k=3)

    print("Query:", query)
    print()

    for result in results:
        print("Chunk ID:", result["chunk_id"])
        print("Score:", result["score"])
        print("Text preview:")
        print(result["text"][:300])
        print("-" * 80)