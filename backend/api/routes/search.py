"""Endpoints for document search."""

from fastapi import APIRouter, HTTPException

from backend.schemas.search import SearchRequest
from backend.services.search_service import (
    EmptyDocumentSelectionError,
    NoIndexedChunksError,
    NoMatchingDocumentsError,
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
            document_ids=request.document_ids,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found.",
        ) from error
    except EmptyDocumentSelectionError as error:
        raise HTTPException(
            status_code=400,
            detail="Select at least one document.",
        ) from error
    except NoMatchingDocumentsError as error:
        raise HTTPException(
            status_code=404,
            detail="None of the selected documents are indexed.",
        ) from error


@router.post("/search/compare")
def compare_search_endpoint(request: SearchRequest):
    try:
        return compare_search_methods(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found.",
        ) from error
    except EmptyDocumentSelectionError as error:
        raise HTTPException(
            status_code=400,
            detail="Select at least one document.",
        ) from error
    except NoMatchingDocumentsError as error:
        raise HTTPException(
            status_code=404,
            detail="None of the selected documents are indexed.",
        ) from error


@router.post("/search")
def search_endpoint(request: SearchRequest):
    try:
        return hybrid_search(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except NoIndexedChunksError as error:
        raise HTTPException(
            status_code=404,
            detail=(
                "No embeddings found. "
                "Upload and process a document first."
            ),
        ) from error
    except EmptyDocumentSelectionError as error:
        raise HTTPException(
            status_code=400,
            detail="Select at least one document.",
        ) from error
    except NoMatchingDocumentsError as error:
        raise HTTPException(
            status_code=404,
            detail="None of the selected documents are indexed.",
        ) from error