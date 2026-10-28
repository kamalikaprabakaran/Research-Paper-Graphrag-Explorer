"""Tests for the PDF ingestion module.

All test PDFs are generated programmatically with PyMuPDF.
No external files, APIs, or credentials are required.
"""

from pathlib import Path

import pymupdf
import pytest

from ingestion.pdf_loader import load_pdf


# ---------------------------------------------------------------------------
# Helpers – programmatic PDF generation
# ---------------------------------------------------------------------------

def _create_pdf(path: Path, pages: list[str], metadata: dict | None = None) -> Path:
    """Create a minimal PDF with the given page texts and optional metadata."""
    doc = pymupdf.open()
    if metadata:
        doc.set_metadata(metadata)
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_valid_multipage_pdf(tmp_path: Path):
    """A 3-page PDF returns the correct text for every page."""
    texts = ["First page content", "Second page content", "Third page content"]
    pdf_path = _create_pdf(tmp_path / "multi.pdf", texts)

    result = load_pdf(pdf_path)

    assert result.page_count == 3
    assert len(result.pages) == 3
    for idx, expected in enumerate(texts):
        assert expected in result.pages[idx].text


def test_page_numbering_is_one_based(tmp_path: Path):
    """Page numbers start at 1, not 0."""
    pdf_path = _create_pdf(tmp_path / "nums.pdf", ["a", "b"])

    result = load_pdf(pdf_path)

    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2


def test_page_count_and_total_characters(tmp_path: Path):
    """page_count matches number of pages; total_characters is consistent."""
    texts = ["Hello", "World"]
    pdf_path = _create_pdf(tmp_path / "count.pdf", texts)

    result = load_pdf(pdf_path)

    assert result.page_count == 2
    assert result.total_characters == sum(len(p.text) for p in result.pages)
    # Each page should contain at least the inserted text
    assert result.total_characters >= sum(len(t) for t in texts)


def test_metadata_extraction(tmp_path: Path):
    """Document metadata is extracted when present."""
    meta = {
        "title": "Test Paper",
        "author": "Jane Doe",
        "subject": "Unit Testing",
        "creator": "PyMuPDF",
    }
    pdf_path = _create_pdf(tmp_path / "meta.pdf", ["content"], metadata=meta)

    result = load_pdf(pdf_path)

    assert result.metadata.title == "Test Paper"
    assert result.metadata.author == "Jane Doe"
    assert result.metadata.subject == "Unit Testing"
    assert result.metadata.creator == "PyMuPDF"


def test_page_with_no_text(tmp_path: Path):
    """A page with no text returns an empty string without crashing."""
    doc = pymupdf.open()
    doc.new_page()  # blank page – no text inserted
    doc.new_page()
    doc[1].insert_text((72, 72), "has text")
    pdf_path = tmp_path / "blank.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = load_pdf(pdf_path)

    assert result.pages[0].text == ""
    assert "has text" in result.pages[1].text


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------

def test_nonexistent_file():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent/paper.pdf")


def test_directory_instead_of_file(tmp_path: Path):
    """Passing a directory raises IsADirectoryError."""
    with pytest.raises(IsADirectoryError):
        load_pdf(tmp_path)


def test_non_pdf_file(tmp_path: Path):
    """A file without a .pdf extension raises ValueError."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("not a pdf")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_pdf(txt_file)


def test_corrupted_pdf(tmp_path: Path):
    """A file with .pdf extension but invalid content raises ValueError."""
    bad_pdf = tmp_path / "broken.pdf"
    bad_pdf.write_bytes(b"this is not a real pdf file")

    with pytest.raises(ValueError, match="Unable to open PDF"):
        load_pdf(bad_pdf)


def test_single_blank_page_pdf(tmp_path: Path):
    """A PDF with one blank page returns page_count=1 and empty text.

    PyMuPDF cannot save a document with zero pages, so a single blank page
    is the closest testable edge case for a near-empty PDF.
    """
    doc = pymupdf.open()
    doc.new_page()  # one blank page, no text
    pdf_path = tmp_path / "blank_only.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = load_pdf(pdf_path)

    assert result.page_count == 1
    assert len(result.pages) == 1
    assert result.pages[0].text == ""
    assert result.total_characters == 0
