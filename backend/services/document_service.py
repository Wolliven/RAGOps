"""Service for managing indexed documents."""

from backend.storage.file_store import (
    delete_document_data,
    get_original_file_path
)
from pathlib import Path

class DocumentNotFoundError(LookupError):
    """Raised when an indexed document does not exist."""

def delete_indexed_document(document_id: str) -> dict:
    """Delete an indexed document and all its stored data."""

    deleted_document = delete_document_data(document_id)

    if deleted_document is None:
        raise DocumentNotFoundError

    metadata = deleted_document["metadata"]

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "filename": metadata["filename"],
        "deleted_files": len(deleted_document["deleted_files"]),
    }

def get_document_source_path(
    document_id: str,
) -> Path:
    """Return the original source file for an indexed document."""

    file_path = get_original_file_path(document_id)

    if file_path is None:
        raise DocumentNotFoundError

    return file_path