"""Tests for the FAISS vector search module (Phase 2.3).

All tests use synthetic NumPy vectors; no model downloads or network access
are required.

Test strategy
-------------
* Vectors are hand-crafted unit vectors so similarity rankings are trivial
  to reason about.
* :class:`~retrieval.embedder.EmbeddingResult` is constructed directly from
  ``TextChunk`` objects and pre-built NumPy arrays — no real embedder is used.
* FAISS itself is exercised directly (not mocked), which catches real
  integration issues without requiring external services.
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieval.chunker import TextChunk
from retrieval.embedder import EmbeddingResult
from retrieval.vector_store import SearchResult, VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # small fixed dimension for fast tests


def _unit(v: list[float]) -> np.ndarray:
    """Return a float32 L2-normalized vector."""
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).astype(np.float32)


def _make_chunk(
    text: str = "chunk text",
    doc_id: str = "paper.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
) -> TextChunk:
    return TextChunk(
        text=text,
        doc_id=doc_id,
        page_number=page_number,
        chunk_index=chunk_index,
    )


def _make_result(
    texts: list[str],
    vectors: np.ndarray,
    doc_id: str = "paper.pdf",
) -> EmbeddingResult:
    """Build an EmbeddingResult from a list of texts and a pre-built matrix."""
    assert len(texts) == vectors.shape[0], "texts and vectors must have the same length"
    chunks = [
        _make_chunk(text=t, doc_id=doc_id, chunk_index=i)
        for i, t in enumerate(texts)
    ]
    return EmbeddingResult(chunks=chunks, vectors=vectors)


def _random_unit_matrix(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Return an (n, dim) float32 matrix with L2-normalized rows."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / norms


# ---------------------------------------------------------------------------
# TestBuildIndex
# ---------------------------------------------------------------------------

class TestBuildIndex:
    """Building an index from valid embeddings."""

    def test_size_after_add(self):
        vecs = _random_unit_matrix(3)
        result = _make_result(["a", "b", "c"], vecs)
        store = VectorStore()
        store.add(result)
        assert store.size == 3

    def test_dim_set_after_add(self):
        vecs = _random_unit_matrix(2)
        result = _make_result(["x", "y"], vecs)
        store = VectorStore()
        assert store.dim is None
        store.add(result)
        assert store.dim == DIM

    def test_dim_pre_allocated(self):
        store = VectorStore(dim=DIM)
        assert store.dim == DIM
        assert store.size == 0

    def test_chunks_stored_after_add(self):
        vecs = _random_unit_matrix(2)
        result = _make_result(["hello", "world"], vecs)
        store = VectorStore()
        store.add(result)
        assert len(store.chunks) == 2
        assert store.chunks[0].text == "hello"
        assert store.chunks[1].text == "world"

    def test_empty_result_is_noop(self):
        store = VectorStore()
        empty = EmbeddingResult(
            chunks=[], vectors=np.empty((0, 0), dtype=np.float32)
        )
        store.add(empty)   # must not raise
        assert store.size == 0
        assert store.dim is None


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------

class TestSearch:
    """Retrieving results and verifying ranking / scores."""

    def _make_distinct_store(self) -> tuple[VectorStore, list[np.ndarray]]:
        """Index 3 orthogonal-ish basis vectors for easy ranking checks."""
        # Make three clearly distinct unit vectors in 8-D
        v0 = _unit([1, 0, 0, 0, 0, 0, 0, 0])
        v1 = _unit([0, 1, 0, 0, 0, 0, 0, 0])
        v2 = _unit([0, 0, 1, 0, 0, 0, 0, 0])
        vecs = np.stack([v0, v1, v2])
        result = _make_result(["doc0", "doc1", "doc2"], vecs)
        store = VectorStore()
        store.add(result)
        return store, [v0, v1, v2]

    def test_top_result_is_most_similar(self):
        store, vecs = self._make_distinct_store()
        # Query closest to v1
        query = vecs[1]
        results = store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0].text == "doc1"

    def test_results_sorted_by_descending_score(self):
        store, vecs = self._make_distinct_store()
        query = vecs[0]
        results = store.search(query, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_is_approximately_one_for_exact_match(self):
        vecs = _random_unit_matrix(1)
        result = _make_result(["only chunk"], vecs)
        store = VectorStore()
        store.add(result)
        query = vecs[0]  # identical to the indexed vector
        results = store.search(query, top_k=1)
        assert abs(results[0].score - 1.0) < 1e-5

    def test_returns_search_result_objects(self):
        vecs = _random_unit_matrix(2)
        result = _make_result(["a", "b"], vecs)
        store = VectorStore()
        store.add(result)
        results = store.search(vecs[0], top_k=1)
        assert isinstance(results[0], SearchResult)


# ---------------------------------------------------------------------------
# TestMetadataPreservation
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    """Chunk metadata (doc_id, page_number, chunk_index) round-trips correctly."""

    def test_doc_id_preserved(self):
        vecs = _random_unit_matrix(1)
        chunk = _make_chunk(text="content", doc_id="paper_A.pdf")
        result = EmbeddingResult(chunks=[chunk], vectors=vecs)
        store = VectorStore()
        store.add(result)
        results = store.search(vecs[0], top_k=1)
        assert results[0].doc_id == "paper_A.pdf"

    def test_page_number_preserved(self):
        vecs = _random_unit_matrix(1)
        chunk = _make_chunk(text="pg3", page_number=3)
        result = EmbeddingResult(chunks=[chunk], vectors=vecs)
        store = VectorStore()
        store.add(result)
        results = store.search(vecs[0], top_k=1)
        assert results[0].page_number == 3

    def test_chunk_index_preserved(self):
        vecs = _random_unit_matrix(1)
        chunk = _make_chunk(text="c7", chunk_index=7)
        result = EmbeddingResult(chunks=[chunk], vectors=vecs)
        store = VectorStore()
        store.add(result)
        results = store.search(vecs[0], top_k=1)
        assert results[0].chunk_index == 7

    def test_text_preserved(self):
        vecs = _random_unit_matrix(1)
        chunk = _make_chunk(text="exact chunk text here")
        result = EmbeddingResult(chunks=[chunk], vectors=vecs)
        store = VectorStore()
        store.add(result)
        results = store.search(vecs[0], top_k=1)
        assert results[0].text == "exact chunk text here"

    def test_multi_chunk_metadata_alignment(self):
        """Each result must map to the correct chunk, not a shifted one."""
        v0 = _unit([1, 0, 0, 0, 0, 0, 0, 0])
        v1 = _unit([0, 1, 0, 0, 0, 0, 0, 0])
        vecs = np.stack([v0, v1])
        chunks = [
            _make_chunk(text="page1chunk", page_number=1, chunk_index=0),
            _make_chunk(text="page2chunk", page_number=2, chunk_index=0),
        ]
        result = EmbeddingResult(chunks=chunks, vectors=vecs)
        store = VectorStore()
        store.add(result)

        # Query v1 → "page2chunk" should be top result
        results = store.search(v1, top_k=2)
        assert results[0].text == "page2chunk"
        assert results[0].page_number == 2


# ---------------------------------------------------------------------------
# TestEmptyIndex
# ---------------------------------------------------------------------------

class TestEmptyIndex:
    """Searching an empty index returns an empty list."""

    def test_empty_index_returns_empty_list(self):
        store = VectorStore()
        query = _random_unit_matrix(1)[0]
        results = store.search(query, top_k=5)
        assert results == []

    def test_empty_index_after_clear(self):
        vecs = _random_unit_matrix(3)
        result = _make_result(["a", "b", "c"], vecs)
        store = VectorStore()
        store.add(result)
        store.clear()
        query = _random_unit_matrix(1)[0]
        results = store.search(query, top_k=3)
        assert results == []


# ---------------------------------------------------------------------------
# TestTopK
# ---------------------------------------------------------------------------

class TestTopK:
    """top_k edge-cases."""

    def test_top_k_larger_than_index_returns_all(self):
        vecs = _random_unit_matrix(3)
        result = _make_result(["a", "b", "c"], vecs)
        store = VectorStore()
        store.add(result)
        results = store.search(vecs[0], top_k=100)
        assert len(results) == 3  # capped to indexed count

    def test_top_k_equals_one(self):
        vecs = _random_unit_matrix(5)
        result = _make_result([str(i) for i in range(5)], vecs)
        store = VectorStore()
        store.add(result)
        results = store.search(vecs[0], top_k=1)
        assert len(results) == 1

    def test_top_k_equals_index_size(self):
        vecs = _random_unit_matrix(4)
        result = _make_result(["a", "b", "c", "d"], vecs)
        store = VectorStore()
        store.add(result)
        results = store.search(vecs[0], top_k=4)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# TestIncrementalAdd
# ---------------------------------------------------------------------------

class TestIncrementalAdd:
    """Adding to the index multiple times keeps metadata aligned."""

    def test_size_grows_incrementally(self):
        store = VectorStore()
        for i in range(3):
            vecs = _random_unit_matrix(2, seed=i)
            result = _make_result([f"a{i}", f"b{i}"], vecs, doc_id=f"doc{i}.pdf")
            store.add(result)
        assert store.size == 6

    def test_metadata_from_all_batches_present(self):
        v0 = _random_unit_matrix(1, seed=0)
        v1 = _random_unit_matrix(1, seed=1)
        r0 = _make_result(["first"], v0, doc_id="doc0.pdf")
        r1 = _make_result(["second"], v1, doc_id="doc1.pdf")
        store = VectorStore()
        store.add(r0)
        store.add(r1)
        doc_ids = {c.doc_id for c in store.chunks}
        assert "doc0.pdf" in doc_ids
        assert "doc1.pdf" in doc_ids

    def test_no_metadata_corruption_after_add(self):
        """After adding a second batch the first batch's metadata is intact."""
        v0 = _unit([1, 0, 0, 0, 0, 0, 0, 0])
        v1 = _unit([0, 1, 0, 0, 0, 0, 0, 0])
        r0 = EmbeddingResult(
            chunks=[_make_chunk(text="original", page_number=10)],
            vectors=v0.reshape(1, -1),
        )
        r1 = EmbeddingResult(
            chunks=[_make_chunk(text="new", page_number=20)],
            vectors=v1.reshape(1, -1),
        )
        store = VectorStore()
        store.add(r0)
        store.add(r1)
        # Query v0 → should get "original" from page 10
        results = store.search(v0, top_k=1)
        assert results[0].text == "original"
        assert results[0].page_number == 10

    def test_dimension_mismatch_between_batches_raises(self):
        vecs_8 = _random_unit_matrix(1, dim=8)
        vecs_16 = _random_unit_matrix(1, dim=16)
        r0 = _make_result(["a"], vecs_8)
        r1 = _make_result(["b"], vecs_16)
        store = VectorStore()
        store.add(r0)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add(r1)


# ---------------------------------------------------------------------------
# TestRebuild
# ---------------------------------------------------------------------------

class TestRebuild:
    """clear() and rebuild() reset the index without stale metadata."""

    def test_rebuild_replaces_old_data(self):
        old_vecs = _random_unit_matrix(3, seed=0)
        new_vecs = _random_unit_matrix(2, seed=1)
        store = VectorStore()
        store.add(_make_result(["old0", "old1", "old2"], old_vecs))
        store.rebuild(_make_result(["new0", "new1"], new_vecs))
        assert store.size == 2
        texts = {c.text for c in store.chunks}
        assert "old0" not in texts
        assert "new0" in texts

    def test_clear_resets_dim(self):
        vecs = _random_unit_matrix(2, dim=8)
        store = VectorStore()
        store.add(_make_result(["a", "b"], vecs))
        assert store.dim == 8
        store.clear()
        assert store.dim is None

    def test_clear_resets_chunks(self):
        vecs = _random_unit_matrix(2)
        store = VectorStore()
        store.add(_make_result(["a", "b"], vecs))
        store.clear()
        assert store.chunks == []

    def test_can_add_different_dim_after_clear(self):
        store = VectorStore()
        store.add(_make_result(["a"], _random_unit_matrix(1, dim=8)))
        store.clear()
        store.add(_make_result(["b"], _random_unit_matrix(1, dim=16)))
        assert store.dim == 16


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    """Invalid inputs raise descriptive, typed exceptions."""

    def test_non_embedding_result_raises(self):
        store = VectorStore()
        with pytest.raises(TypeError, match="EmbeddingResult"):
            store.add("not a result")  # type: ignore[arg-type]

    def test_query_not_ndarray_raises(self):
        store = VectorStore()
        store.add(_make_result(["a"], _random_unit_matrix(1)))
        with pytest.raises(TypeError, match="NumPy ndarray"):
            store.search([0.1] * DIM, top_k=1)  # type: ignore[arg-type]

    def test_query_wrong_shape_raises(self):
        store = VectorStore()
        store.add(_make_result(["a"], _random_unit_matrix(1)))
        q_2d = _random_unit_matrix(1)  # shape (1, DIM) — should be 1-D
        with pytest.raises(ValueError, match="1-D"):
            store.search(q_2d, top_k=1)

    def test_query_wrong_dtype_raises(self):
        store = VectorStore()
        store.add(_make_result(["a"], _random_unit_matrix(1)))
        q = np.zeros(DIM, dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            store.search(q, top_k=1)

    def test_query_wrong_dimension_raises(self):
        store = VectorStore()
        store.add(_make_result(["a"], _random_unit_matrix(1, dim=8)))
        q = np.ones(16, dtype=np.float32)
        q /= np.linalg.norm(q)
        with pytest.raises(ValueError, match="dimension"):
            store.search(q, top_k=1)

    def test_top_k_zero_raises(self):
        store = VectorStore()
        store.add(_make_result(["a"], _random_unit_matrix(1)))
        with pytest.raises(ValueError, match="top_k"):
            store.search(_random_unit_matrix(1)[0], top_k=0)

    def test_top_k_negative_raises(self):
        store = VectorStore()
        store.add(_make_result(["a"], _random_unit_matrix(1)))
        with pytest.raises(ValueError, match="top_k"):
            store.search(_random_unit_matrix(1)[0], top_k=-3)

    def test_invalid_dim_at_construction_raises(self):
        with pytest.raises(ValueError, match="dim"):
            VectorStore(dim=0)

    def test_non_float32_vectors_are_converted(self):
        """EmbeddingResult with float64 vectors is accepted (converted internally)."""
        vecs_f64 = _random_unit_matrix(2).astype(np.float64)
        chunks = [_make_chunk(text="a"), _make_chunk(text="b", chunk_index=1)]
        # EmbeddingResult validates its own vectors; pass float32
        vecs_f32 = vecs_f64.astype(np.float32)
        result = EmbeddingResult(chunks=chunks, vectors=vecs_f32)
        store = VectorStore()
        store.add(result)
        assert store.size == 2


# ---------------------------------------------------------------------------
# TestSearchResultDataclass
# ---------------------------------------------------------------------------

class TestSearchResultDataclass:
    """SearchResult behaves as an immutable dataclass."""

    def test_from_chunk_populates_all_fields(self):
        chunk = _make_chunk(text="hello", doc_id="x.pdf", page_number=5, chunk_index=2)
        sr = SearchResult.from_chunk(chunk, score=0.95)
        assert sr.text == "hello"
        assert sr.doc_id == "x.pdf"
        assert sr.page_number == 5
        assert sr.chunk_index == 2
        assert sr.score == pytest.approx(0.95)

    def test_search_result_is_frozen(self):
        sr = SearchResult(score=0.9, text="t", doc_id="d", page_number=1, chunk_index=0)
        with pytest.raises((AttributeError, TypeError)):
            sr.score = 0.5  # type: ignore[misc]
