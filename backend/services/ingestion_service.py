"""Service for processing uploaded documents."""

from typing import BinaryIO
from datetime import datetime, timezone

from backend.processing.chunking import (
    chunk_text,
    create_chunk_metadata,
)
from backend.processing.extractors import extract_text_from_file
from backend.services.embedding_service import embed_chunks
from backend.storage.file_store import (
    save_uploaded_file,
    save_processed_text,
    save_chunks,
    save_embeddings,
    save_document_metadata,
)


class EmptyDocumentError(ValueError):
    """Raised when no usable text can be extracted from a document."""


def ingest_document(
    filename: str,
    file_object: BinaryIO,
) -> dict:
    """Process and index an uploaded document."""

    file_path = save_uploaded_file(
        filename=filename,
        file_object=file_object,
    )

    text = extract_text_from_file(file_path)

    if not text.strip():
        raise EmptyDocumentError(
            "No text could be extracted from this file."
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
        source_file=filename,
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

    embedding_dimensions = (
        len(embedded_chunks[0]["embedding"])
        if embedded_chunks
        else 0
    )

    metadata = {
        "document_id": document_id,
        "filename": filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_type": file_path.suffix.lower(),
        "technical": {
            "characters": len(text),
            "chunk_count": len(chunk_data),
            "embedding_dimensions": embedding_dimensions,
        },
    }

    save_document_metadata(
        document_id=document_id,
        metadata=metadata,
    )

    return {
        "message": "File uploaded successfully",
        "filename": filename,
        "saved_path": str(file_path),
        "processed_path": str(processed_path),
        "characters": len(text),
        "embeddings_path": str(embeddings_path),
        "embedding_dimensions": embedding_dimensions,
    }