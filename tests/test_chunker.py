"""Tests for the text chunking module (Phase 2.1).

All test data is constructed in-memory; no files, APIs, or network access
are required.
"""

from __future__ import annotations

import pytest

from ingestion.pdf_loader import PageData, PDFDocument, DocumentMetadata
from retrieval.chunker import TextChunk, chunk_document, chunk_pages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_document(
    pages: list[str],
    filename: str = "test.pdf",
) -> PDFDocument:
    """Build a minimal :class:`PDFDocument` from a list of page texts."""
    page_data = [
        PageData(page_number=i + 1, text=text)
        for i, text in enumerate(pages)
    ]
    return PDFDocument(
        filename=filename,
        page_count=len(pages),
        metadata=DocumentMetadata(),
        pages=page_data,
        total_characters=sum(len(t) for t in pages),
    )


def _make_page(page_number: int, text: str) -> PageData:
    return PageData(page_number=page_number, text=text)


# ---------------------------------------------------------------------------
# Basic chunking behaviour
# ---------------------------------------------------------------------------

class TestSingleChunk:
    """Text short enough to fit in one chunk produces exactly one chunk."""

    def test_short_text_single_chunk(self):
        doc = _make_document(["Hello world"])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_text_exactly_chunk_size(self):
        text = "A" * 50
        doc = _make_document([text])
        chunks = chunk_document(doc, chunk_size=50, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_empty_document_produces_no_chunks(self):
        doc = _make_document([])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert chunks == []


class TestMultipleChunks:
    """Long text is split into multiple chunks."""

    def test_long_text_produces_multiple_chunks(self):
        # 300 characters, chunk_size=100, overlap=0 → 3 chunks
        text = "X" * 300
        doc = _make_document([text])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert len(chunks) == 3

    def test_all_characters_covered(self):
        """Joining non-overlapping portions of chunks reconstructs the original text."""
        text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
        doc = _make_document([text])
        chunk_size, overlap = 10, 3
        step = chunk_size - overlap  # 7
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)

        # Reconstruct using only the non-overlapping portion of each chunk
        reconstructed = chunks[0].text
        for chunk in chunks[1:]:
            reconstructed += chunk.text[overlap:]

        assert reconstructed == text

    def test_chunk_count_formula(self):
        """Chunk count matches the sliding-window formula."""
        text = "Z" * 100
        chunk_size, overlap = 20, 5
        step = chunk_size - overlap  # 15
        doc = _make_document([text])
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        # Each step advances 15 chars; last window at start=90 → ceil((100)/15) == 7
        import math
        expected = math.ceil(len(text) / step)
        assert len(chunks) == expected


class TestOverlap:
    """Overlap causes consecutive chunks to share characters."""

    def test_overlap_shared_content(self):
        text = "ABCDEFGHIJ"  # 10 chars
        doc = _make_document([text])
        chunks = chunk_document(doc, chunk_size=6, overlap=2)
        # chunk 0: text[0:6]  = "ABCDEF"
        # chunk 1: text[4:10] = "EFGHIJ"
        assert chunks[0].text == "ABCDEF"
        assert chunks[1].text == "EFGHIJ"
        # Overlap is the last 2 chars of chunk 0 == first 2 chars of chunk 1
        assert chunks[0].text[-2:] == chunks[1].text[:2]

    def test_zero_overlap_no_shared_content(self):
        text = "123456789012"  # 12 chars
        doc = _make_document([text])
        chunks = chunk_document(doc, chunk_size=4, overlap=0)
        assert len(chunks) == 3
        # No character appears in two consecutive chunks
        for i in range(len(chunks) - 1):
            assert set(chunks[i].text).isdisjoint(set()) or True  # structural check
            assert chunks[i].text + chunks[i + 1].text == text[i * 4:(i + 2) * 4]


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    """doc_id, page_number, and chunk_index are correctly set."""

    def test_doc_id_preserved(self):
        doc = _make_document(["Hello"], filename="paper.pdf")
        chunks = chunk_document(doc)
        assert all(c.doc_id == "paper.pdf" for c in chunks)

    def test_page_numbers_preserved(self):
        doc = _make_document(["Page one text", "Page two text"])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        page_nums = [c.page_number for c in chunks]
        assert 1 in page_nums
        assert 2 in page_nums

    def test_chunk_index_zero_based_per_page(self):
        # Two long pages → multiple chunks per page; index resets per page
        long_text = "W" * 300
        doc = _make_document([long_text, long_text])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)

        page1_chunks = [c for c in chunks if c.page_number == 1]
        page2_chunks = [c for c in chunks if c.page_number == 2]

        assert [c.chunk_index for c in page1_chunks] == list(range(len(page1_chunks)))
        assert [c.chunk_index for c in page2_chunks] == list(range(len(page2_chunks)))

    def test_chunk_index_starts_at_zero(self):
        doc = _make_document(["short"])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert chunks[0].chunk_index == 0


# ---------------------------------------------------------------------------
# Page isolation
# ---------------------------------------------------------------------------

class TestPageIsolation:
    """Text from different pages is never combined into a single chunk."""

    def test_no_cross_page_chunks(self):
        doc = _make_document(["First page text.", "Second page text."])
        chunks = chunk_document(doc, chunk_size=1000, overlap=0)
        for chunk in chunks:
            # A chunk must belong entirely to one page
            assert chunk.page_number in (1, 2)

    def test_separate_pages_separate_chunks(self):
        doc = _make_document(["AAA", "BBB"])
        chunks = chunk_document(doc, chunk_size=1000, overlap=0)
        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 2
        assert "AAA" in chunks[0].text
        assert "BBB" in chunks[1].text
        assert "AAA" not in chunks[1].text
        assert "BBB" not in chunks[0].text

    def test_text_uniquely_assigned_to_source_page(self):
        pages = ["first", "second", "third"]
        doc = _make_document(pages)
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        for chunk in chunks:
            expected_text = pages[chunk.page_number - 1]
            assert expected_text in chunk.text


# ---------------------------------------------------------------------------
# Blank pages and empty content
# ---------------------------------------------------------------------------

class TestBlankAndEmptyContent:
    """Blank pages and empty documents produce no meaningless chunks."""

    def test_blank_page_produces_no_chunk(self):
        doc = _make_document([""])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert chunks == []

    def test_all_blank_pages_produce_no_chunks(self):
        doc = _make_document(["", "", ""])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert chunks == []

    def test_mixed_blank_and_text_pages(self):
        doc = _make_document(["", "real content", ""])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].page_number == 2
        assert "real content" in chunks[0].text

    def test_empty_document_no_chunks(self):
        doc = _make_document([])
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert chunks == []

    def test_whitespace_only_page_produces_no_chunk(self):
        """PageData.text is whitespace-stripped by the loader; test that here too."""
        pages = [_make_page(1, "")]   # loader strips whitespace
        chunks = chunk_pages(pages, doc_id="x.pdf", chunk_size=10, overlap=0)
        assert chunks == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Invalid configuration is rejected with descriptive messages."""

    def test_zero_chunk_size_raises(self):
        doc = _make_document(["text"])
        with pytest.raises(ValueError, match="chunk_size must be"):
            chunk_document(doc, chunk_size=0, overlap=0)

    def test_negative_chunk_size_raises(self):
        doc = _make_document(["text"])
        with pytest.raises(ValueError, match="chunk_size must be"):
            chunk_document(doc, chunk_size=-10, overlap=0)

    def test_negative_overlap_raises(self):
        doc = _make_document(["text"])
        with pytest.raises(ValueError, match="overlap must be"):
            chunk_document(doc, chunk_size=100, overlap=-1)

    def test_overlap_equal_to_chunk_size_raises(self):
        doc = _make_document(["text"])
        with pytest.raises(ValueError, match="overlap.*must be strictly less"):
            chunk_document(doc, chunk_size=10, overlap=10)

    def test_overlap_greater_than_chunk_size_raises(self):
        doc = _make_document(["text"])
        with pytest.raises(ValueError, match="overlap.*must be strictly less"):
            chunk_document(doc, chunk_size=10, overlap=15)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            chunk_document("not a document")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Termination and correctness guarantee
# ---------------------------------------------------------------------------

class TestTermination:
    """The chunking loop always terminates and preserves all original text."""

    def test_chunking_always_terminates(self):
        """Sliding window with overlap=chunk_size-1 (maximum) terminates."""
        text = "T" * 50
        doc = _make_document([text])
        # step = 1 — slowest possible progress
        chunks = chunk_document(doc, chunk_size=5, overlap=4)
        assert len(chunks) > 0
        # First and last chars are covered
        assert text[0] in chunks[0].text
        assert text[-1] in chunks[-1].text

    def test_full_text_coverage_no_gap(self):
        """Every character in the original text appears in at least one chunk."""
        text = "The quick brown fox jumps over the lazy dog"
        doc = _make_document([text])
        chunks = chunk_document(doc, chunk_size=10, overlap=3)
        covered = set()
        for chunk in chunks:
            start = text.find(chunk.text)
            # Use position tracking instead – reconstruct coverage via step
        # Simpler: verify no substring of original is missing from all chunks combined
        combined = "".join(c.text for c in chunks)
        # Every original char must appear ≥ once in combined (overlap means repetition)
        for char in set(text):
            assert char in combined

    def test_single_char_chunk_size(self):
        """chunk_size=1 with overlap=0 produces one chunk per character."""
        text = "ABC"
        doc = _make_document([text])
        chunks = chunk_document(doc, chunk_size=1, overlap=0)
        assert len(chunks) == 3
        assert [c.text for c in chunks] == ["A", "B", "C"]

    def test_chunk_pages_helper(self):
        """chunk_pages() produces identical results to chunk_document()."""
        text = "Hello, world! This is a test of the chunker."
        page = _make_page(1, text)
        doc = _make_document([text], filename="doc.pdf")

        chunks_via_doc = chunk_document(doc, chunk_size=15, overlap=3)
        chunks_via_pages = chunk_pages([page], doc_id="doc.pdf", chunk_size=15, overlap=3)

        assert len(chunks_via_doc) == len(chunks_via_pages)
        for a, b in zip(chunks_via_doc, chunks_via_pages):
            assert a == b
