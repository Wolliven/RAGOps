# RAGOps

RAGOps is a small AI document intelligence platform for uploading documents, searching them semantically, generating answers with sources, and evaluating response quality.

The goal of this project is to build a practical RAG-based system step by step, focusing on document ingestion, retrieval, answer generation, citations, evaluation, and logging.

## Planned Features

- Upload documents
- Extract text from TXT, Markdown, PDF, and CSV files
- Split documents into chunks
- Generate embeddings for each chunk
- Search relevant chunks using semantic search
- Generate answers using an LLM
- Show sources used in each answer
- Log questions, answers, latency, and retrieved chunks
- Evaluate answer quality using a small test dataset

## Tech Stack

- Python
- FastAPI
- Streamlit
- SQLite
- sentence-transformers
- FAISS or Qdrant
- Ollama, Gemini, or OpenAI

## Project Status

Initial development stage.

Current goal:

- Set up the basic project structure
- Build document upload
- Extract and save text from uploaded files

## Architecture

User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Document Processor
  ↓
Embedding Model
  ↓
Vector Database
  ↓
Retriever
  ↓
LLM
  ↓
Answer + Sources + Metrics

## License

MIT License