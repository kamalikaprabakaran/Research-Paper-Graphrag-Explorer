"""Tests for the OpenRouter answer-generation module (Phase 2.4).

All tests run 100% offline.  The embedder, vector store, and OpenRouter client
are replaced by lightweight mocks or stubs so no real API calls or model
downloads occur.

Mock strategy
-------------
* A ``SimpleNamespace``-based ``FakeSettings`` replaces the real Settings.
* A stub ``ChunkEmbedder`` subclass overrides ``embed_query`` without loading
  a model.
* A ``MockVectorStore`` subclass overrides ``search`` to return controlled data.
* The OpenRouter client is injected via the ``client_factory`` parameter so
  tests never import or instantiate a real ``openai.OpenAI``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retrieval.chunker import TextChunk
from retrieval.embedder import ChunkEmbedder
from retrieval.vector_store import SearchResult, VectorStore

from generation.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
    AnswerResult,
    Citation,
    EmptyRetrievalError,
    MissingCredentialsError,
    build_prompt,
    _INSUFFICIENT_EVIDENCE_MARKER,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

DIM = 8


def _unit_vec(values: list[float]) -> np.ndarray:
    a = np.array(values, dtype=np.float32)
    return (a / np.linalg.norm(a)).astype(np.float32)


def _fake_settings(
    api_key: str = "test-key",
    model: str = "openai/gpt-4o-mini",
    base_url: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        OPENROUTER_API_KEY=api_key,
        OPENROUTER_MODEL=model,
        OPENROUTER_BASE_URL=base_url,
    )


def _make_search_result(
    text: str = "evidence text",
    doc_id: str = "paper.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        score=score,
        text=text,
        doc_id=doc_id,
        page_number=page_number,
        chunk_index=chunk_index,
    )


def _fake_client_factory(answer_text: str = "Grounded answer."):
    """Return a client_factory callable that produces a mock OpenAI client."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = answer_text
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )

    def factory(api_key, base_url):
        return mock_client

    factory._mock_client = mock_client  # expose for call-count assertions
    return factory


class FakeEmbedder(ChunkEmbedder):
    """Minimal stub that returns a deterministic query vector without loading a model."""

    def __init__(self, query_vec: np.ndarray | None = None):
        super().__init__(model_name="stub/model")
        self._query_vec = query_vec if query_vec is not None else np.ones(DIM, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:  # type: ignore[override]
        return self._query_vec


class FakeVectorStore(VectorStore):
    """Minimal stub whose search() returns a controlled list."""

    def __init__(self, results: list[SearchResult] | None = None):
        super().__init__(dim=DIM)
        self._results = results if results is not None else []
        # Pre-populate _dim so validation passes when querying
        self._dim = DIM

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:  # type: ignore[override]
        return self._results[:top_k]


def _make_generator(
    results: list[SearchResult] | None = None,
    answer_text: str = "Grounded answer.",
    settings=None,
) -> tuple[AnswerGenerator, MagicMock]:
    """Return a fully-mocked AnswerGenerator and its underlying mock client."""
    factory = _fake_client_factory(answer_text)
    gen = AnswerGenerator(
        vector_store=FakeVectorStore(results or [_make_search_result()]),
        embedder=FakeEmbedder(),
        settings=settings or _fake_settings(),
        client_factory=factory,
    )
    return gen, factory._mock_client


# ---------------------------------------------------------------------------
# TestSuccessfulAnswer
# ---------------------------------------------------------------------------

class TestSuccessfulAnswer:
    """A well-configured generator returns a grounded AnswerResult."""

    def test_returns_answer_result(self):
        gen, _ = _make_generator()
        result = gen.answer("What is attention?")
        assert isinstance(result, AnswerResult)

    def test_answer_text_matches_model_reply(self):
        gen, _ = _make_generator(answer_text="Attention is a mechanism.")
        result = gen.answer("What is attention?")
        assert result.answer == "Attention is a mechanism."

    def test_question_preserved_in_result(self):
        gen, _ = _make_generator()
        result = gen.answer("My test question?")
        assert result.question == "My test question?"

    def test_insufficient_evidence_false_for_normal_answer(self):
        gen, _ = _make_generator(answer_text="Clear grounded answer.")
        result = gen.answer("Question?")
        assert not result.insufficient_evidence

    def test_insufficient_evidence_true_when_model_signals_it(self):
        gen, _ = _make_generator(answer_text=_INSUFFICIENT_EVIDENCE_MARKER)
        result = gen.answer("Unknown question?")
        assert result.insufficient_evidence


# ---------------------------------------------------------------------------
# TestQuestionEmbeddingAndTopK
# ---------------------------------------------------------------------------

class TestQuestionEmbeddingAndTopK:
    """The question is correctly embedded and top_k is propagated."""

    def test_embed_query_called_with_question(self):
        embedder = FakeEmbedder()
        original_embed = embedder.embed_query
        called_with = []

        def tracking_embed(q):
            called_with.append(q)
            return original_embed(q)

        embedder.embed_query = tracking_embed  # type: ignore[method-assign]

        factory = _fake_client_factory()
        gen = AnswerGenerator(
            vector_store=FakeVectorStore([_make_search_result()]),
            embedder=embedder,
            settings=_fake_settings(),
            client_factory=factory,
        )
        gen.answer("What is BERT?")
        assert called_with == ["What is BERT?"]

    def test_top_k_limits_citations(self):
        results_pool = [_make_search_result(text=f"r{i}") for i in range(5)]
        gen, _ = _make_generator(results=results_pool)
        result = gen.answer("Question?", top_k=2)
        # citations must not exceed top_k
        assert len(result.citations) <= 2

    def test_default_top_k_is_five(self):
        results_pool = [_make_search_result(text=f"r{i}") for i in range(10)]
        gen, _ = _make_generator(results=results_pool)
        result = gen.answer("Question?")
        assert len(result.citations) <= 5


# ---------------------------------------------------------------------------
# TestCitations
# ---------------------------------------------------------------------------

class TestCitations:
    """Citations match the retrieved chunks, not any LLM output."""

    def test_citation_count_matches_retrieved(self):
        results = [_make_search_result(text=f"chunk{i}") for i in range(3)]
        gen, _ = _make_generator(results=results)
        result = gen.answer("Question?", top_k=3)
        assert len(result.citations) == 3

    def test_citation_doc_id_matches_chunk(self):
        sr = _make_search_result(doc_id="myfile.pdf")
        gen, _ = _make_generator(results=[sr])
        result = gen.answer("Question?", top_k=1)
        assert result.citations[0].doc_id == "myfile.pdf"

    def test_citation_page_number_matches_chunk(self):
        sr = _make_search_result(page_number=7)
        gen, _ = _make_generator(results=[sr])
        result = gen.answer("Question?", top_k=1)
        assert result.citations[0].page_number == 7

    def test_citation_chunk_index_matches_chunk(self):
        sr = _make_search_result(chunk_index=3)
        gen, _ = _make_generator(results=[sr])
        result = gen.answer("Question?", top_k=1)
        assert result.citations[0].chunk_index == 3

    def test_citation_excerpt_is_from_chunk_text(self):
        sr = _make_search_result(text="The quick brown fox")
        gen, _ = _make_generator(results=[sr])
        result = gen.answer("Question?", top_k=1)
        assert "The quick brown fox" in result.citations[0].excerpt

    def test_no_fabricated_citations(self):
        """Citations come only from SearchResult objects, never from LLM text."""
        sr = _make_search_result(doc_id="real.pdf", page_number=1)
        # LLM claims a different source — citations must still reflect FAISS results
        gen, _ = _make_generator(
            results=[sr],
            answer_text="See fabricated.pdf page 99 for details.",
        )
        result = gen.answer("Question?", top_k=1)
        assert len(result.citations) == 1
        assert result.citations[0].doc_id == "real.pdf"

    def test_citation_order_matches_retrieval_order(self):
        results = [
            _make_search_result(doc_id="a.pdf", score=0.9),
            _make_search_result(doc_id="b.pdf", score=0.7),
        ]
        gen, _ = _make_generator(results=results)
        result = gen.answer("Question?", top_k=2)
        assert result.citations[0].doc_id == "a.pdf"
        assert result.citations[1].doc_id == "b.pdf"


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Empty questions and invalid top_k values raise ValueError."""

    def test_empty_question_raises(self):
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="empty"):
            gen.answer("")

    def test_whitespace_question_raises(self):
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="empty"):
            gen.answer("   ")

    def test_non_string_question_raises(self):
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="str"):
            gen.answer(42)  # type: ignore[arg-type]

    def test_zero_top_k_raises(self):
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="top_k"):
            gen.answer("Question?", top_k=0)

    def test_negative_top_k_raises(self):
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="top_k"):
            gen.answer("Question?", top_k=-1)

    def test_float_top_k_raises(self):
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="top_k"):
            gen.answer("Question?", top_k=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestEmptyRetrieval
# ---------------------------------------------------------------------------

class TestEmptyRetrieval:
    """An empty FAISS result raises EmptyRetrievalError — no silent substitution."""

    def test_empty_retrieval_raises(self):
        factory = _fake_client_factory()
        gen = AnswerGenerator(
            vector_store=FakeVectorStore(results=[]),
            embedder=FakeEmbedder(),
            settings=_fake_settings(),
            client_factory=factory,
        )
        with pytest.raises(EmptyRetrievalError):
            gen.answer("Unknown question?")

    def test_api_not_called_when_retrieval_empty(self):
        factory = _fake_client_factory()
        gen = AnswerGenerator(
            vector_store=FakeVectorStore(results=[]),
            embedder=FakeEmbedder(),
            settings=_fake_settings(),
            client_factory=factory,
        )
        with pytest.raises(EmptyRetrievalError):
            gen.answer("Question?")
        # The mock client must never have been called
        factory._mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# TestMissingCredentials
# ---------------------------------------------------------------------------

class TestMissingCredentials:
    """Missing API key or model name raises MissingCredentialsError."""

    def test_missing_api_key_raises(self):
        gen, _ = _make_generator(settings=_fake_settings(api_key=""))
        with pytest.raises(MissingCredentialsError, match="OPENROUTER_API_KEY"):
            gen.answer("Question?")

    def test_none_api_key_raises(self):
        gen, _ = _make_generator(settings=_fake_settings(api_key=None))
        with pytest.raises(MissingCredentialsError, match="OPENROUTER_API_KEY"):
            gen.answer("Question?")

    def test_missing_model_raises(self):
        gen, _ = _make_generator(settings=_fake_settings(model=""))
        with pytest.raises(MissingCredentialsError, match="OPENROUTER_MODEL"):
            gen.answer("Question?")

    def test_is_subclass_of_answer_generation_error(self):
        assert issubclass(MissingCredentialsError, AnswerGenerationError)


# ---------------------------------------------------------------------------
# TestAPIErrors
# ---------------------------------------------------------------------------

class TestAPIErrors:
    """Network/API failures raise AnswerGenerationError."""

    def test_api_exception_raises_answer_generation_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("503 Service Unavailable")

        def factory(api_key, base_url):
            return mock_client

        gen = AnswerGenerator(
            vector_store=FakeVectorStore([_make_search_result()]),
            embedder=FakeEmbedder(),
            settings=_fake_settings(),
            client_factory=factory,
        )
        with pytest.raises(AnswerGenerationError, match="OpenRouter API call failed"):
            gen.answer("Question?")

    def test_authentication_error_wrapped(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

        def factory(api_key, base_url):
            return mock_client

        gen = AnswerGenerator(
            vector_store=FakeVectorStore([_make_search_result()]),
            embedder=FakeEmbedder(),
            settings=_fake_settings(),
            client_factory=factory,
        )
        with pytest.raises(AnswerGenerationError):
            gen.answer("Question?")


# ---------------------------------------------------------------------------
# TestMalformedResponses
# ---------------------------------------------------------------------------

class TestMalformedResponses:
    """Empty or structurally broken model responses raise AnswerGenerationError."""

    def _generator_with_response(self, choices):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=choices)

        def factory(api_key, base_url):
            return mock_client

        return AnswerGenerator(
            vector_store=FakeVectorStore([_make_search_result()]),
            embedder=FakeEmbedder(),
            settings=_fake_settings(),
            client_factory=factory,
        )

    def test_empty_content_raises(self):
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        gen = self._generator_with_response([mock_choice])
        with pytest.raises(AnswerGenerationError, match="empty"):
            gen.answer("Question?")

    def test_whitespace_content_raises(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "   "
        gen = self._generator_with_response([mock_choice])
        with pytest.raises(AnswerGenerationError, match="empty"):
            gen.answer("Question?")

    def test_no_choices_raises(self):
        gen = self._generator_with_response([])
        with pytest.raises(AnswerGenerationError):
            gen.answer("Question?")


# ---------------------------------------------------------------------------
# TestBuildPrompt (unit tests for the pure helper)
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """build_prompt() constructs a correct messages list."""

    def test_returns_two_messages(self):
        sr = _make_search_result()
        msgs = build_prompt("My question?", [sr])
        assert len(msgs) == 2

    def test_system_message_first(self):
        msgs = build_prompt("Q?", [_make_search_result()])
        assert msgs[0]["role"] == "system"

    def test_user_message_second(self):
        msgs = build_prompt("Q?", [_make_search_result()])
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Q?"

    def test_evidence_text_in_system_message(self):
        sr = _make_search_result(text="Critical finding here", doc_id="doc.pdf", page_number=3)
        msgs = build_prompt("Q?", [sr])
        assert "Critical finding here" in msgs[0]["content"]
        assert "doc.pdf" in msgs[0]["content"]
        assert "page 3" in msgs[0]["content"]

    def test_insufficient_evidence_phrase_in_system(self):
        msgs = build_prompt("Q?", [_make_search_result()])
        assert _INSUFFICIENT_EVIDENCE_MARKER in msgs[0]["content"]

    def test_grounding_instruction_present(self):
        msgs = build_prompt("Q?", [_make_search_result()])
        assert "ONLY" in msgs[0]["content"]

    def test_multiple_results_numbered(self):
        results = [_make_search_result(text=f"passage {i}") for i in range(3)]
        msgs = build_prompt("Q?", results)
        content = msgs[0]["content"]
        assert "[1]" in content
        assert "[2]" in content
        assert "[3]" in content


# ---------------------------------------------------------------------------
# TestConstructorValidation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    """AnswerGenerator raises TypeError for wrong argument types."""

    def test_non_vector_store_raises(self):
        with pytest.raises(TypeError, match="VectorStore"):
            AnswerGenerator(
                vector_store="not a store",  # type: ignore[arg-type]
                embedder=FakeEmbedder(),
                settings=_fake_settings(),
            )

    def test_non_embedder_raises(self):
        with pytest.raises(TypeError, match="ChunkEmbedder"):
            AnswerGenerator(
                vector_store=FakeVectorStore(),
                embedder="not an embedder",  # type: ignore[arg-type]
                settings=_fake_settings(),
            )


# ---------------------------------------------------------------------------
# TestCitationDataclass
# ---------------------------------------------------------------------------

class TestCitationDataclass:
    """Citation.from_search_result maps fields correctly."""

    def test_all_fields_mapped(self):
        sr = _make_search_result(
            text="Long text " * 30,
            doc_id="cited.pdf",
            page_number=5,
            chunk_index=2,
        )
        cit = Citation.from_search_result(sr)
        assert cit.doc_id == "cited.pdf"
        assert cit.page_number == 5
        assert cit.chunk_index == 2

    def test_excerpt_truncated_to_200_chars(self):
        long_text = "A" * 500
        sr = _make_search_result(text=long_text)
        cit = Citation.from_search_result(sr)
        assert len(cit.excerpt) == 200

    def test_short_text_not_padded(self):
        sr = _make_search_result(text="Short")
        cit = Citation.from_search_result(sr)
        assert cit.excerpt == "Short"
