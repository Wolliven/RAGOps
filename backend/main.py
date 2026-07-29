"""
FastAPI entry point for RAGOps.

Handles document uploads, text extraction, chunk creation,
embedding generation, and search API endpoints.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import json
from sentence_transformers import SentenceTransformer
from functools import lru_cache

from backend.retrieval.semantic import search_chunks
from backend.retrieval.bm25 import build_bm25_index, search_bm25
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.core.config import (
    UPLOAD_DIR,
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL_NAME,
    create_data_directories,
)
from backend.schemas.search import SearchRequest
from backend.processing.extractors import extract_text_from_file
from backend.processing.chunking import (
    chunk_text,
    create_chunk_metadata,
)

create_data_directories()

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "RAGOps backend is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/search/bm25")
def bm25_search_endpoint(request: SearchRequest):
    chunks = load_all_embedded_chunks()

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found."
        )

    top_k = min(max(request.top_k, 1), 20)

    retriever = build_bm25_index(chunks)

    results = search_bm25(
        query=request.query,
        retriever=retriever,
        top_k=top_k
    )

    return {
        "query": request.query,
        "results": results
    }

@app.post("/search/compare")
def compare_search_endpoint(request: SearchRequest):
    chunks = load_all_embedded_chunks()

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found."
        )

    top_k = min(max(request.top_k, 1), 20)

    semantic_results = search_chunks(
        query=request.query,
        embedded_chunks=chunks,
        model=get_embedding_model(),
        top_k=top_k
    )

    bm25_retriever = build_bm25_index(chunks)

    bm25_results = search_bm25(
        query=request.query,
        retriever=bm25_retriever,
        top_k=top_k
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results= semantic_results,
        bm25_results= bm25_results,
        top_k=request.top_k
        )

    return {
        "query": request.query,
        "semantic_results": semantic_results,
        "bm25_results": bm25_results,
        "hybrid_results": fused_results
}   



@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_file(file_path)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file."
        )
    processed_path = PROCESSED_DIR / f"{file_path.stem}.txt"
    processed_path.write_text(text, encoding="utf-8")
    chunks = chunk_text(text)
    document_id = file_path.stem
    chunk_data = create_chunk_metadata(chunks=chunks, document_id=document_id, source_file=file.filename)

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

    top_k = min(max(request.top_k, 1), 20)

    semantic_results = search_chunks(
        query=request.query,
        embedded_chunks=embedded_chunks,
        model=get_embedding_model(),
        top_k=20
    )

    bm25_retriever = build_bm25_index(embedded_chunks)

    bm25_results = search_bm25(
        query=request.query,
        retriever=bm25_retriever,
        top_k=20
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results= semantic_results,
        bm25_results= bm25_results,
        top_k=top_k
        )

    return {
        "query": request.query,
        "results": fused_results
    }

def embed_chunks(chunk_data: list[dict]) -> list[dict]:
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

def load_all_embedded_chunks() -> list[dict]:
    all_chunks = []

    for embeddings_path in EMBEDDINGS_DIR.glob("*.json"):
        json_text = embeddings_path.read_text(encoding="utf-8")
        chunks = json.loads(json_text)
        all_chunks.extend(chunks)

    return all_chunks