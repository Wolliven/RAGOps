"""Service for managing indexed documents."""

from backend.storage.file_store import delete_document_data


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