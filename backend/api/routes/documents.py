"""Endpoints for uploading and processing documents."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.ingestion_service import (
    EmptyDocumentError,
    ingest_document,
)


router = APIRouter(tags=["documents"])


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