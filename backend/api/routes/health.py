"""Basic application status endpoints."""

from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def read_root():
    return {"message": "RAGOps backend is running"}


@router.get("/health")
def health_check():
    return {"status": "ok"}