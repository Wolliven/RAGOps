"""Request and response schemas for search endpoints."""

from pydantic import BaseModel


class SearchRequest(BaseModel):
    """Data required to perform a search."""

    query: str
    top_k: int = 3