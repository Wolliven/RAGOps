from pathlib import Path


UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")
DOCUMENTS_DIR = Path("data/documents")

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def create_data_directories() -> None:
    """Create the directories used to store RAGOps data."""

    directories = [
        UPLOAD_DIR,
        PROCESSED_DIR,
        CHUNKS_DIR,
        EMBEDDINGS_DIR,
        DOCUMENTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)