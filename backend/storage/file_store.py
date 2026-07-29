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
    data: list[dict],
) -> Path:
    """Save data as a formatted JSON file."""

    file_path = directory / f"{document_id}.json"

    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return file_path