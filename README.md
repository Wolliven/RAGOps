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

## How to run

### Run the backend

Open a terminal in the project root:

````powershell
.venv\Scripts\activate
uvicorn backend.main:app --reload
````

FastAPI runs at:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Run the frontend

Open a second terminal in the folder containing `package.json` and `app.js`.

Install the Node.js dependencies the first time:

```powershell
npm install
```

Start the Express frontend:

```powershell
npm start
```

The web manager opens at:

[http://127.0.0.1:3000/ragops/search](http://127.0.0.1:3000/ragops/search)

## Normal flow

1. Start the FastAPI backend.
2. Start the Express frontend.
3. Open the RAGOps Web Manager.
4. Upload a `.txt`, `.md`, or `.pdf` document.
5. Select the indexed documents to search.
6. Ask a question and review the retrieved chunks.

````

Importante: para ejecutar el frontend **no necesitas activar el entorno virtual de Python**. Son procesos separados:

```text
Terminal 1: Python / FastAPI / port 8000
Terminal 2: Node.js / Express / port 3000
````

La parte antigua:

```powershell
python -m streamlit run frontend/app.py
```

ya debe eliminarse del flujo principal. Puedes mencionar Streamlit únicamente en una sección como `Legacy development interface`, si todavía conservas ese frontend.


## License

MIT License