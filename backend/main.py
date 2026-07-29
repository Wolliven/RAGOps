"""FastAPI application entry point for RAGOps."""

from fastapi import FastAPI

from backend.api.routes import documents, health, search
from backend.core.config import create_data_directories


create_data_directories()

app = FastAPI()

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)