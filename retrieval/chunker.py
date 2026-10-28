"""Text chunking module for the Research Paper Connection Explorer.

Converts a :class:`~ingestion.pdf_loader.PDFDocument` into a flat list of
:class:`TextChunk` objects suitable for downstream embedding and retrieval.

Design principles
-----------------
* Character-based sliding window – no external NLP library required.
* Pages are **never** merged; every chunk inherits the page number of the
  source page, preserving citation accuracy.
* Blank / empty pages produce zero chunks rather than meaningless empty chunks.
* Configurable *chunk_size* and *overlap*; both are validated eagerly.
* The sliding window always advances by at least one character, making
  infinite loops impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ingestion.pdf_loader import PDFDocument, PageData


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextChunk:
    """A single text chunk extracted from one page of a PDF document.

    Attributes:
        doc_id:      The filename of the source document.
        page_number: 1-based page number within the source document.
        chunk_index: 0-based index of this chunk within its source page.
        text:        The chunk text (never empty for a valid chunk).
    """

    doc_id: str
    page_number: int
    chunk_index: int
    text: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    document: PDFDocument,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[TextChunk]:
    """Split a :class:`PDFDocument` into overlapping text chunks.

    Each chunk is scoped to a single page; text from different pages is
    never combined.  Empty pages and documents with no text produce an
    empty result list.

    Args:
        document:   The :class:`PDFDocument` returned by ``load_pdf()``.
        chunk_size: Maximum number of characters per chunk.  Must be ≥ 1.
        overlap:    Number of characters shared between consecutive chunks
                    on the same page.  Must be ≥ 0 and < *chunk_size*.

    Returns:
        A flat list of :class:`TextChunk` objects in page-then-chunk order.

    Raises:
        TypeError:  If *document* is not a :class:`PDFDocument`.
        ValueError: If *chunk_size* or *overlap* are invalid.
    """
    if not isinstance(document, PDFDocument):
        raise TypeError(
            f"Expected a PDFDocument, got {type(document).__name__!r}."
        )

    _validate_config(chunk_size, overlap)

    chunks: list[TextChunk] = []
    for page in document.pages:
        page_chunks = _chunk_page(
            page=page,
            doc_id=document.filename,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(page_chunks)

    return chunks


def chunk_pages(
    pages: Sequence[PageData],
    doc_id: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[TextChunk]:
    """Lower-level helper: chunk an arbitrary sequence of :class:`PageData`.

    Useful for unit-testing without constructing a full :class:`PDFDocument`.

    Args:
        pages:      Sequence of :class:`PageData` objects.
        doc_id:     Identifier to embed in every returned :class:`TextChunk`.
        chunk_size: Maximum characters per chunk.  Must be ≥ 1.
        overlap:    Shared characters between consecutive chunks.
                    Must be ≥ 0 and < *chunk_size*.

    Returns:
        A flat list of :class:`TextChunk` objects.

    Raises:
        ValueError: If *chunk_size* or *overlap* are invalid.
    """
    _validate_config(chunk_size, overlap)

    chunks: list[TextChunk] = []
    for page in pages:
        chunks.extend(_chunk_page(page=page, doc_id=doc_id, chunk_size=chunk_size, overlap=overlap))
    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_config(chunk_size: int, overlap: int) -> None:
    """Raise :class:`ValueError` for invalid chunking parameters."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError(
            f"chunk_size must be an int, got {type(chunk_size).__name__!r}."
        )
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise ValueError(
            f"overlap must be an int, got {type(overlap).__name__!r}."
        )
    if chunk_size < 1:
        raise ValueError(
            f"chunk_size must be ≥ 1, got {chunk_size}."
        )
    if overlap < 0:
        raise ValueError(
            f"overlap must be ≥ 0, got {overlap}."
        )
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})."
        )


def _chunk_page(
    page: PageData,
    doc_id: str,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    """Produce chunks for a single page using a sliding window.

    The window always advances by at least ``chunk_size - overlap`` characters
    (≥ 1), so the loop is guaranteed to terminate.

    Args:
        page:       The source :class:`PageData`.
        doc_id:     Document identifier forwarded to :class:`TextChunk`.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        A (possibly empty) list of :class:`TextChunk` objects for this page.
    """
    text = page.text
    if not text:
        return []

    step = chunk_size - overlap  # guaranteed ≥ 1 by _validate_config
    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(
            TextChunk(
                doc_id=doc_id,
                page_number=page.page_number,
                chunk_index=chunk_index,
                text=chunk_text,
            )
        )
        start += step
        chunk_index += 1

    return chunks
