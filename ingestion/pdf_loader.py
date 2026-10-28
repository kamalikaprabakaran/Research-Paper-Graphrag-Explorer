"""PDF ingestion module for extracting text and metadata from research papers.

Uses PyMuPDF (fitz) to open PDF files and extract text page by page.

Limitations:
    Scanned image-only PDFs will return empty text for each page.
    OCR is outside the scope of Phase 1. Pages with no extractable text
    are returned with an empty string rather than raising an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pymupdf
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PageData(BaseModel):
    """Text content of a single PDF page."""

    page_number: int = Field(..., description="1-based page index")
    text: str = Field(..., description="Extracted text, whitespace-stripped")


class DocumentMetadata(BaseModel):
    """Optional metadata embedded in the PDF file."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None


class PDFDocument(BaseModel):
    """Structured result of a PDF ingestion operation."""

    filename: str
    page_count: int
    metadata: DocumentMetadata
    pages: list[PageData]
    total_characters: int = Field(
        ..., description="Sum of character counts across all pages"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str | Path) -> PDFDocument:
    """Open a PDF file and return its text and metadata as a ``PDFDocument``.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        A ``PDFDocument`` containing page-level text and document metadata.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        IsADirectoryError: If *file_path* points to a directory.
        ValueError: If the file does not have a ``.pdf`` extension or
            cannot be parsed as a valid PDF.
    """
    path = Path(file_path)

    # --- path validation ---------------------------------------------------
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. Only .pdf files are accepted."
        )

    # --- PDF parsing -------------------------------------------------------
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(f"Unable to open PDF: {exc}") from exc

    try:
        # Extract metadata (PyMuPDF returns a dict with string values)
        raw_meta = doc.metadata or {}
        metadata = DocumentMetadata(
            title=raw_meta.get("title") or None,
            author=raw_meta.get("author") or None,
            subject=raw_meta.get("subject") or None,
            creator=raw_meta.get("creator") or None,
        )

        # Extract pages
        pages: list[PageData] = []
        for idx in range(len(doc)):
            page = doc[idx]
            text = (page.get_text() or "").strip()
            pages.append(PageData(page_number=idx + 1, text=text))

        total_characters = sum(len(p.text) for p in pages)

        return PDFDocument(
            filename=path.name,
            page_count=len(doc),
            metadata=metadata,
            pages=pages,
            total_characters=total_characters,
        )
    finally:
        doc.close()
