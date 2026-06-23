from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import json

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "RAGOps backend is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_file(file_path)
    processed_path = PROCESSED_DIR / f"{file_path.stem}.txt"
    processed_path.write_text(text, encoding="utf-8")
    chunks = chunk_text(text)
    chunk_data = create_chunk_metadata(chunks)

    chunks_path = CHUNKS_DIR / f"{file_path.stem}.json"

    chunks_path.write_text(
        json.dumps(chunk_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "saved_path": str(file_path),
        "processed_path": str(processed_path),
        "characters": len(text)
    }

def extract_text_from_file(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8")
    
    raise ValueError("Unsupported file type")

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk Size must be a positive integer")
    chunks = []
    for start in range(0, len(text), chunk_size):
        chunks.append(text[start:start + chunk_size])

    return chunks

def create_chunk_metadata(chunks: list[str]) -> list[dict]:
    chunk_data = []
    for index, chunk in enumerate(chunks):
        chunk_data.append({
            "chunk_id": index,
            "text": chunk,
            "characters": len(chunk)
        })

    return chunk_data