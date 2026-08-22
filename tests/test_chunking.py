import pytest

from backend.processing.chunking import (
    chunk_text,
    create_chunk_metadata,
)


def test_chunk_text_preserves_source_positions():
    text = "alpha beta gamma delta epsilon"

    chunks = chunk_text(
        text,
        chunk_size=15,
        overlap=5,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert (
            text[chunk["start_char"]:chunk["end_char"]]
            == chunk["text"]
        )


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_chunk_text_rejects_invalid_parameters(
    chunk_size,
    overlap,
):
    with pytest.raises(ValueError):
        chunk_text(
            "example text",
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_create_chunk_metadata():
    chunks = [
        {
            "text": "hello",
            "start_char": 0,
            "end_char": 5,
        },
        {
            "text": "world",
            "start_char": 6,
            "end_char": 11,
        },
    ]

    result = create_chunk_metadata(
        chunks=chunks,
        document_id="Test",
        source_file="test.txt",
    )

    assert result[0]["chunk_id"] == "Test:0"
    assert result[1]["chunk_id"] == "Test:1"
    assert result[0]["document_id"] == "Test"
    assert result[0]["characters"] == 5
    assert result[1]["start_char"] == 6