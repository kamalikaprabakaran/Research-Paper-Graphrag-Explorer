import logging
from pathlib import Path
from typing import Callable, Sequence, TYPE_CHECKING, Any

from ingestion.pdf_loader import PDFDocument, load_pdf
from retrieval.chunker import TextChunk, chunk_document
from retrieval.embedder import ChunkEmbedder, EmbeddingResult
from retrieval.vector_store import VectorStore

if TYPE_CHECKING:
    from generation.answer_generator import AnswerGenerator, AnswerResult

logger = logging.getLogger(__name__)

class RAGPipelineError(Exception):
    """Base class for pipeline errors."""

class EmptyDocumentError(RAGPipelineError):
    """Raised when a PDF yields no extractable text or chunks."""

class NoPDFsProvidedError(RAGPipelineError):
    """Raised when `index_pdfs` receives an empty iterable."""

class RAGPipeline:
    """High‑level orchestrator for end‑to‑end Retrieval‑Augmented Generation.

    All heavy components are injectable to facilitate unit testing.
    """

    def __init__(
        self,
        *,
        pdf_loader: Callable[[str | Path], PDFDocument] = load_pdf,
        chunker: Callable[[PDFDocument], list[TextChunk]] = chunk_document,
        embedder: ChunkEmbedder | None = None,
        vector_store: VectorStore | None = None,
        answer_generator: Any = None,
        settings=None,
        client_factory: Callable | None = None,
    ) -> None:
        self._pdf_loader = pdf_loader
        self._chunker = chunker
        self._embedder = embedder or ChunkEmbedder()
        self._store = vector_store or VectorStore()
        self._settings = settings
        self._client_factory = client_factory
        if answer_generator is not None:
            self._answer_generator = answer_generator
        else:
            # Lazy import to avoid importing heavy dependencies unless needed
            from generation.answer_generator import AnswerGenerator, AnswerResult
            self._answer_generator = AnswerGenerator(
                vector_store=self._store,
                embedder=self._embedder,
                settings=self._settings,
                client_factory=self._client_factory,
            )
        logger.debug("RAGPipeline initialised; embedder loaded=%s", getattr(self._embedder, "is_loaded", False))

    def index_pdfs(self, pdf_paths: Sequence[str | Path]) -> None:
        """Load PDFs, chunk, embed and add to the vector store.

        Raises
        ------
        NoPDFsProvidedError
            If *pdf_paths* is empty.
        EmptyDocumentError
            If a PDF yields no chunks.
        """
        if not pdf_paths:
            raise NoPDFsProvidedError("No PDF file paths were supplied for indexing.")

        for path in pdf_paths:
            logger.debug("Indexing PDF: %s", path)
            pdf_doc: PDFDocument = self._pdf_loader(path)
            chunks: list[TextChunk] = self._chunker(pdf_doc)
            # Remove chunks that contain only whitespace – they are not useful for retrieval
            chunks = [c for c in chunks if c.text.strip()]
            if not chunks:
                raise EmptyDocumentError(
                    f"PDF '{pdf_doc.filename}' produced no usable text chunks to index."
                )
            # Use the injected embedder directly – this respects lazy loading.
            embedding_result: EmbeddingResult = self._embedder.embed_chunks(chunks)
            self._store.add(embedding_result)

        logger.info(
            "Indexed %d PDF(s); store now contains %d vectors.",
            len(pdf_paths),
            self._store.size,
        )

    def answer_question(self, question: str, top_k: int = 5) -> "AnswerResult":
        """Generate a grounded answer for *question* using the stored index."""
        return self._answer_generator.answer(question, top_k=top_k)

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def embedder(self) -> ChunkEmbedder:
        return self._embedder

    @property
    def answer_generator(self) -> Any:
        return self._answer_generator
