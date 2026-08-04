"""Endpoints for uploading and processing documents."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.ingestion_service import (
    EmptyDocumentError,
    ingest_document,
)
from backend.storage.file_store import list_document_metadata
from backend.services.document_service import (
    DocumentNotFoundError,
    delete_indexed_document,
)


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

@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    try:
        return delete_indexed_document(document_id)

    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Indexed document not found.",
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error