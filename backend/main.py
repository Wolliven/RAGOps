from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import json
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer

UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def read_root():
    return {"message": "RAGOps backend is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_file(file_path)
    processed_path = PROCESSED_DIR / f"{file_path.stem}.txt"
    processed_path.write_text(text, encoding="utf-8")
    chunks = chunk_text(text)

    chunk_data = create_chunk_metadata(chunks)

    chunks_path = CHUNKS_DIR / f"{file_path.stem}.json"

    chunks_path.write_text(
        json.dumps(chunk_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    embedded_chunks = embed_chunks(chunk_data)

    embeddings_path = EMBEDDINGS_DIR / f"{file_path.stem}.json"
    embeddings_path.write_text(
        json.dumps(embedded_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "saved_path": str(file_path),
        "processed_path": str(processed_path),
        "characters": len(text),
        "embeddings_path": str(embeddings_path),
        "embedding_dimensions": len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0
    }

@app.post("/search")
def search(request: SearchRequest):
    embedded_chunks = load_all_embedded_chunks()

    if not embedded_chunks:
        raise HTTPException(
            status_code=404,
            detail="No embeddings found. Upload and process a document first."
        )

    top_k = min(request.top_k, 10)

    results = search_chunks(
        query=request.query,
        embedded_chunks=embedded_chunks,
        top_k=top_k
    )

    return {
        "query": request.query,
        "top_k": top_k,
        "results": results
    }

def extract_text_from_file(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8")
    
    raise ValueError("Unsupported file type")

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk Size must be a positive integer")
    chunks = []
    for start in range(0, len(text), chunk_size):
        chunks.append(text[start:start + chunk_size])

    return chunks

def create_chunk_metadata(chunks: list[str]) -> list[dict]:
    chunk_data = []
    for index, chunk in enumerate(chunks):
        chunk_data.append({
            "chunk_id": index,
            "text": chunk,
            "characters": len(chunk)
        })

    return chunk_data

def embed_chunks(chunk_data: list[dict]) -> list[dict]:
    texts = []

    for chunk in chunk_data:
        texts.append(chunk["text"])

    embeddings = embedding_model.encode(texts)

    embedded_chunks = []

    for chunk, embedding in zip(chunk_data, embeddings):
        embedded_chunk = chunk.copy()
        embedded_chunk["embedding"] = embedding.tolist()
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks

def load_all_embedded_chunks() -> list[dict]:
    all_chunks = []

    for embeddings_path in EMBEDDINGS_DIR.glob("*.json"):
        json_text = embeddings_path.read_text(encoding="utf-8")
        chunks = json.loads(json_text)

        for chunk in chunks:
            chunk_copy = chunk.copy()
            chunk_copy["source_file"] = embeddings_path.stem
            all_chunks.append(chunk_copy)

    return all_chunks

def cosine_scores(query_embedding, chunk_embeddings):
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

def search_chunks(query: str, embedded_chunks: list[dict], top_k: int = 3) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not embedded_chunks:
        return []

    query_embedding = embedding_model.encode(query)

    chunk_embeddings = []

    for chunk in embedded_chunks:
        chunk_embeddings.append(chunk["embedding"])

    scores = cosine_scores(query_embedding, chunk_embeddings)

    results = []

    for chunk, score in zip(embedded_chunks, scores):
        result = {
            "source_file": chunk.get("source_file"),
            "chunk_id": chunk.get("chunk_id"),
            "score": float(score),
            "text": chunk.get("text", "")
        }

        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]

