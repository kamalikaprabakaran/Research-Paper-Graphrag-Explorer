"""Tests for the local text-embedding module (Phase 2.2).

All tests run **offline**: the real SentenceTransformer model is replaced by a
lightweight stub via ``unittest.mock``.  No network access, downloads, or API
keys are required.

Mock strategy
-------------
``sentence_transformers.SentenceTransformer`` is patched at the point of
import inside ``retrieval.embedder`` (``retrieval.embedder.SentenceTransformer``
would not work because the class is imported inside ``_load_model``).
Instead we patch the module the embedder imports *from*:
``sentence_transformers.SentenceTransformer``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from retrieval.chunker import TextChunk
from retrieval.embedder import (
    ChunkEmbedder,
    EmbeddingResult,
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    embed_chunks,
    embed_query,
    _validate_chunk_list,
)


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_DIM = EMBEDDING_DIM  # 384


def _make_chunk(
    text: str = "sample text",
    doc_id: str = "paper.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
) -> TextChunk:
    return TextChunk(
        doc_id=doc_id,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
    )


def _fake_normalized_matrix(n: int, dim: int = _DIM) -> np.ndarray:
    """Return a deterministic (n, dim) float32 array with L2-normalized rows."""
    rng = np.random.default_rng(42)
    mat = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms


def _mock_model_factory(n_chunks: int, dim: int = _DIM):
    """Return a mock SentenceTransformer whose encode() returns normalized vectors."""
    mock_model = MagicMock()
    # encode() is called with a list of texts; return a pre-built matrix
    mock_model.encode.return_value = _fake_normalized_matrix(n_chunks, dim)
    return mock_model


# Context manager shorthand: patch the SentenceTransformer class inside the
# sentence_transformers module, which is what ChunkEmbedder._load_model()
# imports from.
PATCH_TARGET = "sentence_transformers.SentenceTransformer"


# ---------------------------------------------------------------------------
# TestEmptyInput
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Empty chunk list returns a correctly shaped result without loading the model."""

    def test_empty_list_returns_empty_result(self):
        embedder = ChunkEmbedder()
        result = embedder.embed_chunks([])
        assert isinstance(result, EmbeddingResult)
        assert result.chunks == []
        assert result.vectors.shape == (0, 0)
        assert result.vectors.dtype == np.float32

    def test_empty_list_does_not_load_model(self):
        embedder = ChunkEmbedder()
        assert not embedder.is_loaded
        with patch(PATCH_TARGET) as mock_cls:
            embedder.embed_chunks([])
            mock_cls.assert_not_called()
        assert not embedder.is_loaded

    def test_module_level_embed_chunks_empty(self):
        """Module-level embed_chunks() fast-paths correctly for empty input."""
        # Use a fresh singleton to avoid state from other tests
        with patch("retrieval.embedder._default_embedder", None):
            with patch(PATCH_TARGET) as mock_cls:
                result = embed_chunks([])
                mock_cls.assert_not_called()
        assert result.chunks == []
        assert result.vectors.shape == (0, 0)


# ---------------------------------------------------------------------------
# TestEmbedChunks
# ---------------------------------------------------------------------------

class TestEmbedChunks:
    """Embedding a list of chunks produces aligned, correctly-shaped results."""

    def test_vector_count_matches_chunk_count(self):
        chunks = [_make_chunk(f"text {i}", chunk_index=i) for i in range(5)]
        mock_model = _mock_model_factory(5)
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_chunks(chunks)
        assert result.vectors.shape[0] == 5
        assert len(result.chunks) == 5

    def test_vector_dimension(self):
        chunks = [_make_chunk("hello")]
        mock_model = _mock_model_factory(1)
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_chunks(chunks)
        assert result.vectors.shape == (1, _DIM)

    def test_vector_dtype_is_float32(self):
        chunks = [_make_chunk("hello")]
        mock_model = _mock_model_factory(1)
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_chunks(chunks)
        assert result.vectors.dtype == np.float32

    def test_chunk_order_preserved(self):
        texts = [f"chunk {i}" for i in range(4)]
        chunks = [_make_chunk(t, chunk_index=i) for i, t in enumerate(texts)]
        mock_model = _mock_model_factory(4)
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_chunks(chunks)
        assert [c.text for c in result.chunks] == texts

    def test_chunk_metadata_alignment(self):
        chunks = [
            _make_chunk("page 1 text", doc_id="a.pdf", page_number=1, chunk_index=0),
            _make_chunk("page 2 text", doc_id="a.pdf", page_number=2, chunk_index=0),
        ]
        mock_model = _mock_model_factory(2)
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_chunks(chunks)
        assert result.chunks[0].page_number == 1
        assert result.chunks[1].page_number == 2
        assert result.vectors.shape == (2, _DIM)

    def test_encode_called_with_correct_texts(self):
        chunks = [_make_chunk("first"), _make_chunk("second")]
        mock_model = _mock_model_factory(2)
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            embedder.embed_chunks(chunks)
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        encoded_texts = call_args[0][0]
        assert encoded_texts == ["first", "second"]


# ---------------------------------------------------------------------------
# TestNormalization
# ---------------------------------------------------------------------------

class TestNormalization:
    """Returned vectors are L2-normalized (each row has norm ≈ 1.0)."""

    def test_chunk_vectors_are_normalized(self):
        chunks = [_make_chunk(f"text {i}") for i in range(3)]
        # Give the mock model unnormalized vectors; embedder should normalize them
        # Note: our embedder passes normalize_embeddings=True to model.encode(),
        # so the model is expected to return normalized vectors. We return
        # pre-normalized vectors from our mock to simulate that behaviour.
        mock_model = _mock_model_factory(3)
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_chunks(chunks)
        norms = np.linalg.norm(result.vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_query_vector_is_normalized(self):
        query_vec = _fake_normalized_matrix(1)[0]  # shape (384,)
        mock_model = MagicMock()
        mock_model.encode.return_value = query_vec
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_query("what is attention?")
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# TestQueryEmbedding
# ---------------------------------------------------------------------------

class TestQueryEmbedding:
    """embed_query() returns a single 1-D normalized float32 vector."""

    def test_query_shape(self):
        query_vec = _fake_normalized_matrix(1)[0]  # (384,)
        mock_model = MagicMock()
        mock_model.encode.return_value = query_vec
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_query("attention mechanism")
        assert result.ndim == 1
        assert result.shape == (_DIM,)

    def test_query_dtype(self):
        query_vec = _fake_normalized_matrix(1)[0]
        mock_model = MagicMock()
        mock_model.encode.return_value = query_vec
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            result = embedder.embed_query("test")
        assert result.dtype == np.float32

    def test_module_level_embed_query(self):
        """Module-level embed_query() delegates to the singleton embedder."""
        query_vec = _fake_normalized_matrix(1)[0]
        mock_model = MagicMock()
        mock_model.encode.return_value = query_vec
        with patch("retrieval.embedder._default_embedder", None):
            with patch(PATCH_TARGET, return_value=mock_model):
                result = embed_query("transformer architecture")
        assert result.ndim == 1
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    """Invalid inputs raise descriptive exceptions."""

    def test_non_list_chunks_raises_type_error(self):
        embedder = ChunkEmbedder()
        with pytest.raises(TypeError, match="chunks must be a list"):
            embedder.embed_chunks(("a", "b"))  # type: ignore[arg-type]

    def test_non_chunk_item_raises_type_error(self):
        embedder = ChunkEmbedder()
        with pytest.raises(TypeError, match="TextChunk"):
            embedder.embed_chunks(["not a chunk"])  # type: ignore[arg-type]

    def test_non_string_query_raises_type_error(self):
        embedder = ChunkEmbedder()
        with pytest.raises(TypeError, match="query must be a str"):
            embedder.embed_query(42)  # type: ignore[arg-type]

    def test_empty_query_raises_value_error(self):
        embedder = ChunkEmbedder()
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_query("   ")

    def test_whitespace_query_raises_value_error(self):
        embedder = ChunkEmbedder()
        with pytest.raises(ValueError):
            embedder.embed_query("")

    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            ChunkEmbedder(model_name="")

    def test_validate_chunk_list_helper_direct(self):
        """_validate_chunk_list raises for tuples and mixed lists."""
        with pytest.raises(TypeError):
            _validate_chunk_list({"key": "val"})
        with pytest.raises(TypeError):
            _validate_chunk_list([_make_chunk(), "oops"])


# ---------------------------------------------------------------------------
# TestLazyLoading
# ---------------------------------------------------------------------------

class TestLazyLoading:
    """The model is loaded exactly once, on the first embed call."""

    def test_model_not_loaded_on_construction(self):
        embedder = ChunkEmbedder()
        assert not embedder.is_loaded

    def test_model_loaded_after_first_embed_chunks(self):
        chunk = _make_chunk()
        mock_model = _mock_model_factory(1)
        with patch(PATCH_TARGET, return_value=mock_model) as mock_cls:
            embedder = ChunkEmbedder()
            assert not embedder.is_loaded
            embedder.embed_chunks([chunk])
            assert embedder.is_loaded
            # Constructor called exactly once
            mock_cls.assert_called_once_with(DEFAULT_MODEL)

    def test_model_loaded_after_embed_query(self):
        query_vec = _fake_normalized_matrix(1)[0]
        mock_model = MagicMock()
        mock_model.encode.return_value = query_vec
        with patch(PATCH_TARGET, return_value=mock_model):
            embedder = ChunkEmbedder()
            assert not embedder.is_loaded
            embedder.embed_query("graph neural network")
            assert embedder.is_loaded

    def test_model_loaded_only_once_across_multiple_calls(self):
        chunks = [_make_chunk()]
        mock_model = _mock_model_factory(1)
        with patch(PATCH_TARGET, return_value=mock_model) as mock_cls:
            embedder = ChunkEmbedder()
            embedder.embed_chunks(chunks)
            embedder.embed_chunks(chunks)
            embedder.embed_chunks(chunks)
            # SentenceTransformer() constructor called exactly once
            mock_cls.assert_called_once()

    def test_model_name_property(self):
        embedder = ChunkEmbedder("some/model")
        assert embedder.model_name == "some/model"


# ---------------------------------------------------------------------------
# TestEmbeddingResult
# ---------------------------------------------------------------------------

class TestEmbeddingResult:
    """EmbeddingResult validates its own invariants."""

    def test_mismatched_chunks_and_vectors_raises(self):
        chunks = [_make_chunk(), _make_chunk()]
        vectors = np.zeros((3, _DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            EmbeddingResult(chunks=chunks, vectors=vectors)

    def test_non_2d_vectors_raises(self):
        chunks = [_make_chunk()]
        vectors = np.zeros((_DIM,), dtype=np.float32)  # 1-D
        with pytest.raises(ValueError, match="2-D"):
            EmbeddingResult(chunks=chunks, vectors=vectors)

    def test_valid_result_construction(self):
        chunks = [_make_chunk()]
        vectors = np.zeros((1, _DIM), dtype=np.float32)
        result = EmbeddingResult(chunks=chunks, vectors=vectors)
        assert len(result.chunks) == 1
        assert result.vectors.shape == (1, _DIM)
