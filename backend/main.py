"""
FastAPI entry point for RAGOps.

Handles document uploads, text extraction, chunk creation,
embedding generation, and search API endpoints.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import json
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from pypdf import PdfReader
import re
from backend.retrieval.semantic import search_chunks
from backend.retrieval.bm25 import build_bm25_index, search_bm25
from backend.retrieval.fusion import reciprocal_rank_fusion

UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def read_root():
    return {"message": "RAGOps backend is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/search/bm25")
def bm25_search_endpoint(request: SearchRequest):
    chunks = load_all_embedded_chunks()

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found."
        )

    top_k = min(max(request.top_k, 1), 20)

    retriever = build_bm25_index(chunks)

    results = search_bm25(
        query=request.query,
        retriever=retriever,
        top_k=top_k
    )

    return {
        "query": request.query,
        "results": results
    }

@app.post("/search/compare")
def compare_search_endpoint(request: SearchRequest):
    chunks = load_all_embedded_chunks()

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found."
        )

    top_k = min(max(request.top_k, 1), 20)

    semantic_results = search_chunks(
        query=request.query,
        embedded_chunks=chunks,
        model=get_embedding_model(),
        top_k=top_k
    )

    bm25_retriever = build_bm25_index(chunks)

    bm25_results = search_bm25(
        query=request.query,
        retriever=bm25_retriever,
        top_k=top_k
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results= semantic_results,
        bm25_results= bm25_results,
        top_k=request.top_k
        )

    return {
        "query": request.query,
        "semantic_results": semantic_results,
        "bm25_results": bm25_results,
        "hybrid_results": fused_results
}   



@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_file(file_path)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file."
        )
    processed_path = PROCESSED_DIR / f"{file_path.stem}.txt"
    processed_path.write_text(text, encoding="utf-8")
    chunks = chunk_text(text)
    document_id = file_path.stem
    chunk_data = create_chunk_metadata(chunks=chunks, document_id=document_id, source_file=file.filename)

    chunks_path = CHUNKS_DIR / f"{file_path.stem}.json"

    chunks_path.write_text(
        json.dumps(chunk_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    embedded_chunks = embed_chunks(chunk_data)

    embeddings_path = EMBEDDINGS_DIR / f"{file_path.stem}.json"
    embeddings_path.write_text(
        json.dumps(embedded_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "saved_path": str(file_path),
        "processed_path": str(processed_path),
        "characters": len(text),
        "embeddings_path": str(embeddings_path),
        "embedding_dimensions": len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0
    }

@app.post("/search")
def search(request: SearchRequest):
    embedded_chunks = load_all_embedded_chunks()

    if not embedded_chunks:
        raise HTTPException(
            status_code=404,
            detail="No embeddings found. Upload and process a document first."
        )

    top_k = min(max(request.top_k, 1), 20)

    semantic_results = search_chunks(
        query=request.query,
        embedded_chunks=embedded_chunks,
        model=get_embedding_model(),
        top_k=20
    )

    bm25_retriever = build_bm25_index(embedded_chunks)

    bm25_results = search_bm25(
        query=request.query,
        retriever=bm25_retriever,
        top_k=20
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results= semantic_results,
        bm25_results= bm25_results,
        top_k=request.top_k
        )

    return {
        "query": request.query,
        "results": fused_results
    }


def extract_text_from_file(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8")

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


def extract_text_from_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))

    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            cleaned_page_text = clean_pdf_text(page_text)
            pages_text.append(f"\n\n--- Page {page_number} ---\n\n{cleaned_page_text}")

    return "\n".join(pages_text).strip()


def clean_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split by hyphenation at line endings
    text = re.sub(r"-\s*\n\s*", "", text)

    # Remove empty lines and strip each line
    lines = []

    for line in text.splitlines():
        clean_line = line.strip()

        if clean_line:
            lines.append(clean_line)

    paragraphs = []
    current_paragraph = []

    for line in lines:
        current_paragraph.append(line)

        if line.endswith((".", "!", "?", "…", ".”", "»", "\"")):
            paragraph = " ".join(current_paragraph)
            paragraphs.append(paragraph)
            current_paragraph = []

    if current_paragraph:
        paragraph = " ".join(current_paragraph)
        paragraphs.append(paragraph)

    cleaned_text = "\n\n".join(paragraphs)

    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    return cleaned_text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        max_end = min(start + chunk_size, len(text))
        end = max_end

        if max_end < len(text):
            last_space = text.rfind(" ", start, max_end)

            if last_space != -1 and last_space > start:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

def create_chunk_metadata(chunks: list[str], document_id: str, source_file: str) -> list[dict]:
    chunk_data = []
    for index, chunk in enumerate(chunks):
        chunk_data.append({
            "chunk_id": f"{document_id}:{index}",
            "document_id": document_id,
            "source_file": source_file,
            "chunk_index": index,
            "text": chunk,
            "characters": len(chunk)
        })

    return chunk_data

def embed_chunks(chunk_data: list[dict]) -> list[dict]:
    texts = []

    for chunk in chunk_data:
        texts.append(chunk["text"])

    model = get_embedding_model()
    embeddings = model.encode(texts)

    embedded_chunks = []

    for chunk, embedding in zip(chunk_data, embeddings):
        embedded_chunk = chunk.copy()
        embedded_chunk["embedding"] = embedding.tolist()
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks

def load_all_embedded_chunks() -> list[dict]:
    all_chunks = []

    for embeddings_path in EMBEDDINGS_DIR.glob("*.json"):
        json_text = embeddings_path.read_text(encoding="utf-8")
        chunks = json.loads(json_text)
        all_chunks.extend(chunks)

    return all_chunks