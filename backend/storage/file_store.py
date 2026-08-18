"""Functions for reading and writing RAGOps data."""

import json
import shutil
from pathlib import Path
from typing import BinaryIO

from backend.core.config import (
    UPLOAD_DIR,
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    DOCUMENTS_DIR,
)


def save_uploaded_file(
    filename: str,
    file_object: BinaryIO,
) -> Path:
    """Save an uploaded file to the uploads directory."""

    file_path = UPLOAD_DIR / filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file_object, buffer)

    return file_path


def save_processed_text(
    document_id: str,
    text: str,
) -> Path:
    """Save extracted document text."""

    processed_path = PROCESSED_DIR / f"{document_id}.txt"
    processed_path.write_text(text, encoding="utf-8")

    return processed_path


def save_chunks(
    document_id: str,
    chunks: list[dict],
) -> Path:
    """Save document chunks as JSON."""

    return _save_json(
        directory=CHUNKS_DIR,
        document_id=document_id,
        data=chunks,
    )


def save_embeddings(
    document_id: str,
    embedded_chunks: list[dict],
) -> Path:
    """Save embedded chunks as JSON."""

    return _save_json(
        directory=EMBEDDINGS_DIR,
        document_id=document_id,
        data=embedded_chunks,
    )


def save_document_metadata(
    document_id: str,
    metadata: dict,
) -> Path:
    """Save metadata describing an indexed document."""

    return _save_json(
        directory=DOCUMENTS_DIR,
        document_id=document_id,
        data=metadata,
    )


def list_document_metadata() -> list[dict]:
    """Load metadata for all indexed documents."""

    documents = []

    for metadata_path in DOCUMENTS_DIR.glob("*.json"):
        json_text = metadata_path.read_text(encoding="utf-8")
        metadata = json.loads(json_text)
        documents.append(metadata)

    return sorted(
        documents,
        key=lambda document: document.get("uploaded_at", ""),
        reverse=True,
    )

def delete_document_data(document_id: str) -> dict | None:
    """Delete all files associated with an indexed document."""

    _validate_document_id(document_id)

    metadata_path = DOCUMENTS_DIR / f"{document_id}.json"

    if not metadata_path.exists():
        return None

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    filename = Path(metadata["filename"]).name

    document_paths = [
        UPLOAD_DIR / filename,
        PROCESSED_DIR / f"{document_id}.txt",
        CHUNKS_DIR / f"{document_id}.json",
        EMBEDDINGS_DIR / f"{document_id}.json",
    ]

    deleted_files = []

    for file_path in document_paths:
        if file_path.exists():
            file_path.unlink()
            deleted_files.append(str(file_path))

    # Delete metadata last.
    metadata_path.unlink()
    deleted_files.append(str(metadata_path))

    return {
        "metadata": metadata,
        "deleted_files": deleted_files,
    }


def load_all_embedded_chunks() -> list[dict]:
    """Load embedded chunks from every saved JSON file."""

    all_chunks = []

    for embeddings_path in EMBEDDINGS_DIR.glob("*.json"):
        json_text = embeddings_path.read_text(encoding="utf-8")
        chunks = json.loads(json_text)
        all_chunks.extend(chunks)

    return all_chunks


def _save_json(
    directory: Path,
    document_id: str,
    data: dict | list[dict],
) -> Path:
    """Save data as a formatted JSON file."""

    file_path = directory / f"{document_id}.json"

    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return file_path

def _validate_document_id(document_id: str) -> None:
    """Reject document IDs that could escape the data directories."""

    if (
        not document_id
        or document_id in {".", ".."}
        or "/" in document_id
        or "\\" in document_id
    ):
        raise ValueError("Invalid document ID.")

def load_processed_text(document_id: str) -> str | None:
    """Load the processed text for an indexed document."""

    _validate_document_id(document_id)

    file_path = PROCESSED_DIR / f"{document_id}.txt"

    if not file_path.exists():
        return None

    return file_path.read_text(encoding="utf-8")


def load_document_chunks(document_id: str) -> list[dict] | None:
    """Load the stored chunks for an indexed document."""

    _validate_document_id(document_id)

    file_path = CHUNKS_DIR / f"{document_id}.json"

    if not file_path.exists():
        return None

    return json.loads(
        file_path.read_text(encoding="utf-8")
    )