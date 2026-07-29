"""Functions for splitting documents into searchable chunks."""


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        max_end = min(start + chunk_size, len(text))
        end = max_end

        if max_end < len(text):
            last_space = text.rfind(" ", start, max_end)

            if last_space != -1 and last_space > start:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_chunk_metadata(
    chunks: list[str],
    document_id: str,
    source_file: str,
) -> list[dict]:
    """Attach document information to each chunk."""

    chunk_data = []

    for index, chunk in enumerate(chunks):
        chunk_data.append({
            "chunk_id": f"{document_id}:{index}",
            "document_id": document_id,
            "source_file": source_file,
            "chunk_index": index,
            "text": chunk,
            "characters": len(chunk),
        })

    return chunk_data