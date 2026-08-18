"""Service for managing indexed documents."""

from backend.storage.file_store import (
    delete_document_data,
    load_document_chunks,
    load_processed_text,
)

class DocumentNotFoundError(LookupError):
    """Raised when an indexed document does not exist."""

class DocumentContentNotFoundError(LookupError):
    """Raised when stored document content cannot be loaded."""


class ChunkNotFoundError(LookupError):
    """Raised when a requested document chunk does not exist."""


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

def get_document_view(
    document_id: str,
    chunk_index: int | None = None,
) -> dict:
    """Load processed document text and an optional target chunk."""

    text = load_processed_text(document_id)

    if text is None:
        raise DocumentContentNotFoundError

    target_chunk = None

    if chunk_index is not None:
        chunks = load_document_chunks(document_id)

        if chunks is None:
            raise DocumentContentNotFoundError

        target_chunk = next(
            (
                chunk
                for chunk in chunks
                if chunk.get("chunk_index") == chunk_index
            ),
            None,
        )

        if target_chunk is None:
            raise ChunkNotFoundError

    return {
        "document_id": document_id,
        "text": text,
        "target_chunk": target_chunk,
    }