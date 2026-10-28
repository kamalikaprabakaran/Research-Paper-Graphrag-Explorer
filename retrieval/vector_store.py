"""FAISS-based vector search module for the Research Paper Connection Explorer.

Stores dense embeddings produced by :mod:`retrieval.embedder` in a FAISS
``IndexFlatIP`` index and returns ranked :class:`SearchResult` objects that
include the original chunk metadata.

Index choice
------------
``faiss.IndexFlatIP`` performs **exact** inner-product (dot-product) search.
Because every embedding returned by :class:`~retrieval.embedder.ChunkEmbedder`
is **L2-normalized**, inner product equals cosine similarity:

    cosine_similarity(a, b) = dot(a, b)     when ‖a‖ = ‖b‖ = 1

This avoids the approximation error of ``IndexIVF*`` or ``IndexHNSW`` families
and is fast enough for typical research-paper collections (thousands of chunks).

Typical usage::

    from retrieval.vector_store import VectorStore

    store = VectorStore()
    store.add(embedding_result)               # index chunks from one paper
    store.add(another_embedding_result)       # incrementally index more

    results = store.search(query_vector, top_k=5)
    for r in results:
        print(r.score, r.doc_id, r.page_number, r.text[:80])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import faiss
import numpy as np

from retrieval.chunker import TextChunk
from retrieval.embedder import EmbeddingResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model – search result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchResult:
    """A single ranked search result returned by :meth:`VectorStore.search`.

    Attributes:
        score:        Cosine-similarity score in ``[-1, 1]``; higher is better.
                      Equal to the inner product of two L2-normalized vectors.
        text:         The chunk text that matched the query.
        doc_id:       Filename or identifier of the source document.
        page_number:  1-based page number within the source document.
        chunk_index:  0-based chunk index within its source page.
    """

    score: float
    text: str
    doc_id: str
    page_number: int
    chunk_index: int

    @classmethod
    def from_chunk(cls, chunk: TextChunk, score: float) -> "SearchResult":
        """Construct a :class:`SearchResult` from a :class:`TextChunk` and a score."""
        return cls(
            score=score,
            text=chunk.text,
            doc_id=chunk.doc_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
        )


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """FAISS ``IndexFlatIP`` wrapper that keeps chunk metadata aligned with vectors.

    The store supports:

    * **Incremental addition** – call :meth:`add` multiple times to index
      chunks from several papers without rebuilding.
    * **Exact top-k search** – call :meth:`search` with a 1-D query vector.
    * **Clear / rebuild** – call :meth:`clear` to reset, then :meth:`add` again.

    The dimension *D* is inferred from the first :class:`EmbeddingResult`
    passed to :meth:`add` and is then fixed for the lifetime of the store
    (or until :meth:`clear` is called).

    Args:
        dim: Optional embedding dimension.  If provided the FAISS index is
             pre-allocated immediately; otherwise it is created lazily on the
             first :meth:`add` call.

    Example::

        store = VectorStore()
        store.add(embedding_result)
        results = store.search(query_vec, top_k=5)
    """

    def __init__(self, dim: int | None = None) -> None:
        if dim is not None:
            _validate_dim(dim)
        self._dim: int | None = dim
        self._index: faiss.IndexFlatIP | None = (
            faiss.IndexFlatIP(dim) if dim is not None else None
        )
        self._chunks: list[TextChunk] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int | None:
        """Embedding dimension, or ``None`` if no vectors have been added yet."""
        return self._dim

    @property
    def size(self) -> int:
        """Number of vectors currently indexed."""
        return self._index.ntotal if self._index is not None else 0

    @property
    def chunks(self) -> list[TextChunk]:
        """Read-only view of the indexed :class:`TextChunk` objects, in insertion order."""
        return list(self._chunks)

    def add(self, result: EmbeddingResult) -> None:
        """Add embeddings from an :class:`EmbeddingResult` to the index.

        Chunks and their vectors are appended to any previously indexed data.
        The chunk list and the FAISS index are always kept in sync.

        Args:
            result: An :class:`EmbeddingResult` produced by
                    :meth:`~retrieval.embedder.ChunkEmbedder.embed_chunks`.
                    May be an empty result (no-op).

        Raises:
            TypeError:  If *result* is not an :class:`EmbeddingResult`.
            ValueError: If *result* vectors have a different dimension than
                        previously indexed vectors, or if vectors are not 2-D
                        ``float32``.
        """
        _validate_embedding_result(result)

        if not result.chunks:
            # Empty result – nothing to add.
            return

        vectors = _ensure_float32_c_contiguous(result.vectors)
        n, d = vectors.shape

        # Initialise the index on the first non-empty add.
        if self._index is None:
            self._dim = d
            self._index = faiss.IndexFlatIP(d)
            logger.debug("Initialised FAISS IndexFlatIP with dim=%d.", d)
        elif d != self._dim:
            raise ValueError(
                f"Vector dimension mismatch: the index holds dim={self._dim}, "
                f"but the new vectors have dim={d}."
            )

        self._index.add(vectors)
        self._chunks.extend(result.chunks)
        logger.debug("Added %d vectors; total indexed: %d.", n, self._index.ntotal)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the top-k most similar chunks for a query vector.

        Args:
            query_vector: A 1-D ``float32`` NumPy array of shape ``(D,)``.
                          Must be L2-normalized (as returned by
                          :meth:`~retrieval.embedder.ChunkEmbedder.embed_query`).
            top_k:        Number of results to return.  Must be ≥ 1.
                          If *top_k* exceeds the number of indexed chunks,
                          all indexed chunks are returned (no error).

        Returns:
            A list of :class:`SearchResult` objects sorted by descending
            similarity score.  Returns an empty list if the index is empty.

        Raises:
            TypeError:  If *query_vector* is not a NumPy array.
            ValueError: If *query_vector* is not 1-D, has the wrong dimension,
                        is not ``float32``, or if *top_k* is not a positive int.
        """
        _validate_top_k(top_k)
        _validate_query_vector(query_vector, expected_dim=self._dim)

        if self._index is None or self._index.ntotal == 0:
            return []

        # Cap top_k to the number of indexed vectors.
        k = min(top_k, self._index.ntotal)

        # FAISS expects a 2-D query matrix: shape (1, D).
        q = _ensure_float32_c_contiguous(query_vector.reshape(1, -1))
        scores_2d, indices_2d = self._index.search(q, k)

        scores: np.ndarray = scores_2d[0]     # shape (k,)
        indices: np.ndarray = indices_2d[0]   # shape (k,)

        results: list[SearchResult] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                # FAISS uses -1 for "not found" slots (can occur with IVF; rare
                # with IndexFlatIP but handled defensively).
                continue
            results.append(
                SearchResult.from_chunk(self._chunks[int(idx)], float(score))
            )

        return results

    def clear(self) -> None:
        """Remove all indexed vectors and metadata, resetting the store.

        The embedding dimension is also reset so a subsequent :meth:`add`
        can use a different dimension.
        """
        self._index = None
        self._dim = None
        self._chunks = []
        logger.debug("VectorStore cleared.")

    def rebuild(self, result: EmbeddingResult) -> None:
        """Clear the existing index and rebuild from *result*.

        Equivalent to calling :meth:`clear` followed by :meth:`add`.

        Args:
            result: The :class:`EmbeddingResult` to index from scratch.
        """
        self.clear()
        self.add(result)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_dim(dim: int) -> None:
    """Raise :class:`ValueError` if *dim* is not a positive integer."""
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError(f"dim must be a positive int, got {dim!r}.")


def _validate_top_k(top_k: int) -> None:
    """Raise :class:`ValueError` if *top_k* is not a positive integer."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(
            f"top_k must be a positive int ≥ 1, got {top_k!r}."
        )


def _validate_query_vector(
    query_vector: object,
    expected_dim: int | None,
) -> None:
    """Raise :class:`TypeError` or :class:`ValueError` for invalid query vectors."""
    if not isinstance(query_vector, np.ndarray):
        raise TypeError(
            f"query_vector must be a NumPy ndarray, got {type(query_vector).__name__!r}."
        )
    if query_vector.ndim != 1:
        raise ValueError(
            f"query_vector must be 1-D, got shape {query_vector.shape}."
        )
    if query_vector.dtype != np.float32:
        raise ValueError(
            f"query_vector must be float32, got dtype {query_vector.dtype}."
        )
    if expected_dim is not None and query_vector.shape[0] != expected_dim:
        raise ValueError(
            f"query_vector dimension {query_vector.shape[0]} does not match "
            f"the index dimension {expected_dim}."
        )


def _validate_embedding_result(result: object) -> None:
    """Raise :class:`TypeError` if *result* is not an :class:`EmbeddingResult`."""
    if not isinstance(result, EmbeddingResult):
        raise TypeError(
            f"result must be an EmbeddingResult, got {type(result).__name__!r}."
        )


def _ensure_float32_c_contiguous(arr: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 copy of *arr* (required by FAISS)."""
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr
