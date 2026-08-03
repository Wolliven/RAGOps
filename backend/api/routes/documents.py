"""Endpoints for uploading and processing documents."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.ingestion_service import (
    EmptyDocumentError,
    ingest_document,
)
from backend.storage.file_store import list_document_metadata


router = APIRouter(tags=["documents"])

@router.get("/documents")
def list_documents():
    documents = list_document_metadata()

    return {
        "documents": documents,
        "count": len(documents),
    }

@router.post("/upload")
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