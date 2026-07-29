"""
FastAPI entry point for RAGOps.

Handles document uploads, text extraction, chunk creation,
embedding generation, and search API endpoints.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException

from backend.core.config import create_data_directories
from backend.schemas.search import SearchRequest
from backend.services.ingestion_service import (
    ingest_document,
    EmptyDocumentError,
)
from backend.services.search_service import (
    bm25_search,
    compare_search_methods,
    hybrid_search,
    NoIndexedChunksError,
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
    try:
        return bm25_search(
            query=request.query,
            top_k=request.top_k,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found.",
        ) from error

@app.post("/search/compare")
def compare_search_endpoint(request: SearchRequest):
    try:
        return compare_search_methods(
            query=request.query,
            top_k=request.top_k,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found.",
        ) from error  



@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        return ingest_document(
            filename=file.filename,
            file_object=file.file,
        )
    except EmptyDocumentError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

@app.post("/search")
def search(request: SearchRequest):
    try:
        return hybrid_search(
            query=request.query,
            top_k=request.top_k,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail=(
                "No embeddings found. "
                "Upload and process a document first."
            ),
        ) from error