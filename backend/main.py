"""
FastAPI entry point for RAGOps.

Handles document uploads, text extraction, chunk creation,
embedding generation, and search API endpoints.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException

from backend.retrieval.semantic import search_chunks
from backend.retrieval.bm25 import build_bm25_index, search_bm25
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.core.config import create_data_directories
from backend.schemas.search import SearchRequest
from backend.processing.extractors import extract_text_from_file
from backend.processing.chunking import (
    chunk_text,
    create_chunk_metadata,
)
from backend.services.embedding_service import (
    embed_chunks,
    get_embedding_model,
)
from backend.storage.file_store import (
    save_uploaded_file,
    save_processed_text,
    save_chunks,
    save_embeddings,
    load_all_embedded_chunks,
)
create_data_directories()


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
    file_path = save_uploaded_file(
        filename=file.filename,
        file_object=file.file,
    )

    text = extract_text_from_file(file_path)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file.",
        )

    document_id = file_path.stem

    processed_path = save_processed_text(
        document_id=document_id,
        text=text,
    )

    chunks = chunk_text(text)

    chunk_data = create_chunk_metadata(
        chunks=chunks,
        document_id=document_id,
        source_file=file.filename,
    )

    save_chunks(
        document_id=document_id,
        chunks=chunk_data,
    )

    embedded_chunks = embed_chunks(chunk_data)

    embeddings_path = save_embeddings(
        document_id=document_id,
        embedded_chunks=embedded_chunks,
    )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "saved_path": str(file_path),
        "processed_path": str(processed_path),
        "characters": len(text),
        "embeddings_path": str(embeddings_path),
        "embedding_dimensions": (
            len(embedded_chunks[0]["embedding"])
            if embedded_chunks
            else 0
        ),
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