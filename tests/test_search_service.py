import pytest

from backend.services import search_service


def test_load_indexed_chunks_filters_documents(monkeypatch):
    chunks = [
        {"chunk_id": "Cat:0", "document_id": "Cat"},
        {"chunk_id": "Cat:1", "document_id": "Cat"},
        {"chunk_id": "Black_hole:0", "document_id": "Black_hole"},
    ]

    monkeypatch.setattr(
        search_service,
        "load_all_embedded_chunks",
        lambda: chunks,
    )

    result = search_service._load_indexed_chunks(["Cat"])

    assert len(result) == 2
    assert all(
        chunk["document_id"] == "Cat"
        for chunk in result
    )


def test_load_indexed_chunks_raises_when_index_is_empty(
    monkeypatch,
):
    monkeypatch.setattr(
        search_service,
        "load_all_embedded_chunks",
        lambda: [],
    )

    with pytest.raises(search_service.NoIndexedChunksError):
        search_service._load_indexed_chunks()


def test_load_indexed_chunks_rejects_empty_selection(
    monkeypatch,
):
    monkeypatch.setattr(
        search_service,
        "load_all_embedded_chunks",
        lambda: [
            {"chunk_id": "Cat:0", "document_id": "Cat"}
        ],
    )

    with pytest.raises(
        search_service.EmptyDocumentSelectionError
    ):
        search_service._load_indexed_chunks([])


def test_load_indexed_chunks_rejects_unknown_documents(
    monkeypatch,
):
    monkeypatch.setattr(
        search_service,
        "load_all_embedded_chunks",
        lambda: [
            {"chunk_id": "Cat:0", "document_id": "Cat"}
        ],
    )

    with pytest.raises(
        search_service.NoMatchingDocumentsError
    ):
        search_service._load_indexed_chunks(["Unknown"])