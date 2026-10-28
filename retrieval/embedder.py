"""Local text-embedding module for the Research Paper Connection Explorer.

Converts :class:`~retrieval.chunker.TextChunk` objects into dense numerical
vectors using a locally-cached Sentence Transformers model.

Model
-----
Default: ``sentence-transformers/all-MiniLM-L6-v2``
  * Output dimension: 384
  * Licence: Apache-2.0
  * The model is downloaded once from Hugging Face Hub and cached locally;
    subsequent runs are fully offline.

Normalization
-------------
All embedding vectors are **L2-normalized** before being returned.
Normalization means that cosine similarity between two vectors equals their
dot product, which is what FAISS's ``IndexFlatIP`` (inner-product index)
computes.  It also prevents high-character-count chunks from dominating purely
because their unnormalized magnitude is larger.

Lazy loading
------------
The underlying :class:`sentence_transformers.SentenceTransformer` model is
loaded on the *first* call to :meth:`ChunkEmbedder.embed_chunks` or
:meth:`ChunkEmbedder.embed_query`.  Constructing a :class:`ChunkEmbedder`
is therefore always instant and never triggers a network request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from retrieval.chunker import TextChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
"""Hugging Face model ID used when no custom model name is supplied."""

EMBEDDING_DIM: int = 384
"""Expected output dimension of the default model."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """Paired chunk metadata and embedding matrix.

    Attributes:
        chunks:  The original :class:`TextChunk` objects **in input order**.
                 Use these for source citations or metadata display.
        vectors: Float32 NumPy array of shape ``(N, D)`` where *N* is the
                 number of chunks and *D* is the embedding dimension (384 for
                 the default model).  Each row is L2-normalized.

    Note:
        ``len(chunks) == vectors.shape[0]`` is always true.
        For an empty input, ``vectors`` has shape ``(0, 0)`` and ``chunks``
        is an empty list.
    """

    chunks: list[TextChunk] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))

    def __post_init__(self) -> None:
        if not isinstance(self.vectors, np.ndarray):
            raise TypeError("vectors must be a NumPy ndarray.")
        if self.vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D, got shape {self.vectors.shape}."
            )
        n_chunks = len(self.chunks)
        n_vecs = self.vectors.shape[0]
        if n_chunks != n_vecs:
            raise ValueError(
                f"chunks length ({n_chunks}) does not match vectors row count ({n_vecs})."
            )


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------

class ChunkEmbedder:
    """Wraps a :class:`sentence_transformers.SentenceTransformer` model.

    The model is loaded **lazily**: the first call to :meth:`embed_chunks` or
    :meth:`embed_query` triggers the load; subsequent calls reuse the same
    in-memory model.

    Args:
        model_name: Hugging Face model ID or local path.  Defaults to
                    :data:`DEFAULT_MODEL`.

    Example::

        embedder = ChunkEmbedder()
        result = embedder.embed_chunks(chunks)
        query_vec = embedder.embed_query("attention mechanism")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        self._model_name: str = model_name
        self._model = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[TextChunk]) -> EmbeddingResult:
        """Embed a list of :class:`TextChunk` objects.

        Args:
            chunks: List of :class:`TextChunk` instances.  May be empty.

        Returns:
            An :class:`EmbeddingResult` whose ``chunks`` and ``vectors``
            fields are aligned by index.  Chunk order is preserved exactly.

        Raises:
            TypeError: If *chunks* is not a list, or contains non-TextChunk items.
        """
        _validate_chunk_list(chunks)

        if not chunks:
            # Fast path: skip model loading entirely for empty input.
            return EmbeddingResult(
                chunks=[],
                vectors=np.empty((0, 0), dtype=np.float32),
            )

        texts = [chunk.text for chunk in chunks]
        model = self._load_model()
        raw: np.ndarray = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = raw.astype(np.float32)
        logger.debug(
            "Embedded %d chunks → shape %s using model '%s'.",
            len(chunks),
            vectors.shape,
            self._model_name,
        )
        return EmbeddingResult(chunks=list(chunks), vectors=vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: The search query to embed.

        Returns:
            A 1-D NumPy array of shape ``(D,)`` and dtype ``float32``,
            L2-normalized.

        Raises:
            TypeError: If *query* is not a string.
            ValueError: If *query* is empty.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a str, got {type(query).__name__!r}."
            )
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only.")

        model = self._load_model()
        raw: np.ndarray = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return raw.astype(np.float32)

    @property
    def model_name(self) -> str:
        """The Hugging Face model ID this embedder will use."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """``True`` if the underlying model has been loaded into memory."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Return the cached model, loading it on first call.

        Returns:
            A :class:`sentence_transformers.SentenceTransformer` instance.
        """
        if self._model is None:
            logger.info("Loading SentenceTransformer model '%s' …", self._model_name)
            from sentence_transformers import SentenceTransformer  # local import keeps tests fast

            self._model = SentenceTransformer(self._model_name)
            logger.info("Model loaded successfully.")
        return self._model


# ---------------------------------------------------------------------------
# Module-level convenience functions (singleton embedder)
# ---------------------------------------------------------------------------

_default_embedder: ChunkEmbedder | None = None


def _get_default_embedder() -> ChunkEmbedder:
    """Return (or create) the module-level singleton :class:`ChunkEmbedder`."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = ChunkEmbedder()
    return _default_embedder


def embed_chunks(chunks: list[TextChunk]) -> EmbeddingResult:
    """Module-level convenience wrapper around :meth:`ChunkEmbedder.embed_chunks`.

    Uses a lazily-initialized singleton embedder with the default model.

    Args:
        chunks: List of :class:`TextChunk` objects to embed.

    Returns:
        An :class:`EmbeddingResult` with aligned chunks and vectors.
    """
    return _get_default_embedder().embed_chunks(chunks)


def embed_query(query: str) -> np.ndarray:
    """Module-level convenience wrapper around :meth:`ChunkEmbedder.embed_query`.

    Uses the same singleton embedder as :func:`embed_chunks`.

    Args:
        query: Search query string.

    Returns:
        1-D ``float32`` NumPy array of shape ``(D,)``, L2-normalized.
    """
    return _get_default_embedder().embed_query(query)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_chunk_list(chunks: object) -> None:
    """Raise :class:`TypeError` if *chunks* is not a list of :class:`TextChunk`.

    Args:
        chunks: Object to validate.

    Raises:
        TypeError: If *chunks* is not a list, or any item is not a TextChunk.
    """
    if not isinstance(chunks, list):
        raise TypeError(
            f"chunks must be a list, got {type(chunks).__name__!r}."
        )
    for i, item in enumerate(chunks):
        if not isinstance(item, TextChunk):
            raise TypeError(
                f"All items in chunks must be TextChunk instances; "
                f"item at index {i} is {type(item).__name__!r}."
            )
