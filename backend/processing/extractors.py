"""Utilities for extracting and cleaning text from documents."""

from pathlib import Path
import re

from pypdf import PdfReader


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from a supported document."""

    extension = file_path.suffix.lower()

    if extension in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8")

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract and clean text from every page of a PDF."""

    reader = PdfReader(str(file_path))

    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            cleaned_page_text = clean_pdf_text(page_text)
            pages_text.append(
                f"\n\n--- Page {page_number} ---\n\n{cleaned_page_text}"
            )

    return "\n".join(pages_text).strip()


def clean_pdf_text(text: str) -> str:
    """Clean common formatting problems in extracted PDF text."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split by hyphenation at line endings
    text = re.sub(r"-\s*\n\s*", "", text)

    # Remove empty lines and strip each line
    lines = []

    for line in text.splitlines():
        clean_line = line.strip()

        if clean_line:
            lines.append(clean_line)

    paragraphs = []
    current_paragraph = []

    for line in lines:
        current_paragraph.append(line)

        if line.endswith((".", "!", "?", "…", ".”", "»", "\"")):
            paragraph = " ".join(current_paragraph)
            paragraphs.append(paragraph)
            current_paragraph = []

    if current_paragraph:
        paragraph = " ".join(current_paragraph)
        paragraphs.append(paragraph)

    cleaned_text = "\n\n".join(paragraphs)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    return cleaned_text.strip()