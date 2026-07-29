"""Endpoints for document search."""

from fastapi import APIRouter, HTTPException

from backend.schemas.search import SearchRequest
from backend.services.search_service import (
    NoIndexedChunksError,
    bm25_search,
    compare_search_methods,
    hybrid_search,
)


router = APIRouter(tags=["search"])


@router.post("/search/bm25")
def bm25_search_endpoint(request: SearchRequest):
    try:
        return bm25_search(
            query=request.query,
            top_k=request.top_k,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found.",
        ) from error


@router.post("/search/compare")
def compare_search_endpoint(request: SearchRequest):
    try:
        return compare_search_methods(
            query=request.query,
            top_k=request.top_k,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found.",
        ) from error


@router.post("/search")
def search_endpoint(request: SearchRequest):
    try:
        return hybrid_search(
            query=request.query,
            top_k=request.top_k,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail=(
                "No embeddings found. "
                "Upload and process a document first."
            ),
        ) from error