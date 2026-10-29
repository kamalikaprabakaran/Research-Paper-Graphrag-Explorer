import builtins
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from ingestion.pdf_loader import PDFDocument, PageData, DocumentMetadata
from retrieval.chunker import TextChunk
from retrieval.embedder import EmbeddingResult
from retrieval.vector_store import SearchResult, VectorStore
from generation.answer_generator import AnswerResult, Citation
from pipeline.rag_pipeline import RAGPipeline, NoPDFsProvidedError, EmptyDocumentError

# Helper to create a simple PDFDocument with given text per page
def make_pdf_doc(filename: str, pages_text: List[str]) -> PDFDocument:
    pages = [PageData(page_number=i + 1, text=text) for i, text in enumerate(pages_text)]
    metadata = DocumentMetadata()
    return PDFDocument(
        filename=filename,
        page_count=len(pages),
        metadata=metadata,
        pages=pages,
        total_characters=sum(len(p.text) for p in pages),
    )

# Dummy chunks corresponding to the PDF pages
def dummy_chunks(pdf: PDFDocument) -> List[TextChunk]:
    # Use the real chunker with small size to guarantee one chunk per page
    from retrieval.chunker import chunk_document
    return chunk_document(pdf, chunk_size=1000, overlap=0)

# Fake embedding result – vectors are deterministic simple arrays
def fake_embedding_result(chunks: List[TextChunk]) -> EmbeddingResult:
    import numpy as np
    vectors = np.arange(len(chunks) * 3, dtype=np.float32).reshape(len(chunks), 3)
    # L2‑normalize rows
    norms = (vectors ** 2).sum(axis=1, keepdims=True) ** 0.5
    vectors = vectors / norms
    return EmbeddingResult(chunks=chunks, vectors=vectors)

# Fake answer generator – returns a deterministic AnswerResult
class DummyAnswerGenerator:
    def __init__(self, store: VectorStore, embedder, settings=None, client_factory=None):
        self.store = store
        self.embedder = embedder

    def answer(self, question: str, top_k: int = 5) -> AnswerResult:
        # Assume retrieval returns the first chunk as evidence
        if not self.store.chunks:
            raise RuntimeError("Store is empty")
        sr = SearchResult.from_chunk(self.store.chunks[0], score=1.0)
        citation = Citation.from_search_result(sr)
        return AnswerResult(
            question=question,
            answer="Dummy answer for testing.",
            citations=[citation],
            insufficient_evidence=False,
        )

@pytest.fixture
def pipeline_factory():
    def _make(pdfs: List[PDFDocument]):
        # Mock the pdf_loader to return PDFs from the provided list sequentially
        loader_mock = MagicMock(side_effect=pdfs)
        # Use real chunker
        chunker_func = dummy_chunks
        # Mock embedder to return deterministic embeddings
        embedder_mock = MagicMock()
        embedder_mock.embed_chunks.side_effect = lambda chunks: fake_embedding_result(chunks)
        embedder_mock.embed_query.return_value = __import__("numpy").array([0.1, 0.2, 0.3], dtype=__import__("numpy").float32)
        # Use real VectorStore
        store = VectorStore()
        # Inject dummy answer generator
        answer_gen = DummyAnswerGenerator(store, embedder_mock)
        return RAGPipeline(
            pdf_loader=loader_mock,
            chunker=chunker_func,
            embedder=embedder_mock,
            vector_store=store,
            answer_generator=answer_gen,
        )
    return _make

def test_successful_index_single_pdf(pipeline_factory):
    pdf = make_pdf_doc("test1.pdf", ["Page one text.", "Page two text."])
    rag = pipeline_factory([pdf])
    rag.index_pdfs(["/fake/path/test1.pdf"])
    # After indexing, store should contain vectors for all chunks
    assert rag.store.size == len(rag.store.chunks) > 0
    # Verify that embedder was called exactly once for the chunks
    assert rag.embedder.embed_chunks.called

def test_successful_index_multiple_pdfs(pipeline_factory):
    pdf1 = make_pdf_doc("doc1.pdf", ["First doc page."])
    pdf2 = make_pdf_doc("doc2.pdf", ["Second doc page."])
    rag = pipeline_factory([pdf1, pdf2])
    rag.index_pdfs(["/fake/a.pdf", "/fake/b.pdf"])
    assert rag.store.size == 2
    filenames = {c.doc_id for c in rag.store.chunks}
    assert filenames == {"doc1.pdf", "doc2.pdf"}

def test_metadata_preservation(pipeline_factory):
    pdf = make_pdf_doc("meta.pdf", ["Only page."])
    rag = pipeline_factory([pdf])
    rag.index_pdfs(["/path/meta.pdf"])
    chunk = rag.store.chunks[0]
    assert chunk.doc_id == "meta.pdf"
    assert chunk.page_number == 1
    assert chunk.chunk_index == 0

def test_answer_flow(pipeline_factory):
    pdf = make_pdf_doc("answer.pdf", ["Content for answer."])
    rag = pipeline_factory([pdf])
    rag.index_pdfs(["/path/answer.pdf"])
    result = rag.answer_question("What is this?")
    assert isinstance(result, AnswerResult)
    assert result.answer.startswith("Dummy answer")
    assert len(result.citations) == 1
    cit = result.citations[0]
    assert cit.doc_id == "answer.pdf"

def test_no_pdfs_provided_error(pipeline_factory):
    rag = pipeline_factory([])
    with pytest.raises(NoPDFsProvidedError):
        rag.index_pdfs([])

def test_empty_document_error(pipeline_factory):
    # PDF with empty page text leads to no chunks
    pdf = make_pdf_doc("empty.pdf", ["   "])
    rag = pipeline_factory([pdf])
    with pytest.raises(EmptyDocumentError):
        rag.index_pdfs(["/path/empty.pdf"])
