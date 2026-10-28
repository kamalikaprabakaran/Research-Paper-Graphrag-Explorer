"""OpenRouter answer-generation module for the Research Paper Connection Explorer.

Retrieves relevant text chunks from the FAISS vector store for a user question,
then sends them as grounded evidence to the OpenRouter API to produce a cited
answer.

Architecture
------------
::

    question (str)
         │
         ▼
    ChunkEmbedder.embed_query()   ← uses existing embedder; never re-loaded here
         │ query_vector (np.ndarray)
         ▼
    VectorStore.search()          ← existing FAISS store; top_k configurable
         │ retrieved: list[SearchResult]
         ▼
    _build_prompt()               ← formats evidence + question into a system+user pair
         │ messages
         ▼
    OpenAI(base_url=OPENROUTER)   ← standard OpenAI-compatible client
         │ completion text
         ▼
    AnswerResult(answer, citations)  ← structured, citable output

OpenRouter configuration
------------------------
The following environment variables are read via the shared ``Settings`` object:

* ``OPENROUTER_API_KEY``  – Required to call the API.
* ``OPENROUTER_MODEL``    – Model ID, e.g. ``openai/gpt-4o-mini``.
* ``OPENROUTER_BASE_URL`` – Base URL for the OpenAI-compatible endpoint
                            (default: ``https://openrouter.ai/api/v1``).

See ``.env.example`` for the expected variable names.

Grounding policy
----------------
The system prompt instructs the model to answer **only** from the supplied
evidence passages and to explicitly say "I cannot determine this from the
provided documents" when the evidence is insufficient.  No fabricated citations
can appear — citations are built exclusively from the ``SearchResult`` objects
returned by FAISS, not from anything the LLM generates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from retrieval.embedder import ChunkEmbedder
from retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
"""Fallback OpenRouter base URL when ``OPENROUTER_BASE_URL`` env var is absent."""

DEFAULT_TOP_K: int = 5
"""Default number of retrieved chunks to use as evidence."""

_INSUFFICIENT_EVIDENCE_MARKER: str = "I cannot determine this from the provided documents"
"""Phrase added to the system prompt instructing the model to signal gaps."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    """A single source citation derived from a retrieved chunk.

    Attributes:
        doc_id:       Filename of the source document.
        page_number:  1-based page number within the document.
        chunk_index:  0-based chunk index within the page.
        excerpt:      A short excerpt of the retrieved text (first 200 chars).
    """

    doc_id: str
    page_number: int
    chunk_index: int
    excerpt: str

    @classmethod
    def from_search_result(cls, sr: SearchResult) -> "Citation":
        """Build a :class:`Citation` from a :class:`~retrieval.vector_store.SearchResult`."""
        return cls(
            doc_id=sr.doc_id,
            page_number=sr.page_number,
            chunk_index=sr.chunk_index,
            excerpt=sr.text[:200],
        )


@dataclass
class AnswerResult:
    """The structured output of an answer-generation call.

    Attributes:
        question:   The original user question (unchanged).
        answer:     The model-generated answer text, grounded in the evidence.
        citations:  One :class:`Citation` per retrieved chunk, in relevance
                    order (most similar first).  Always derived from real FAISS
                    results — never fabricated.
        insufficient_evidence: ``True`` when the model signalled that the
                    retrieved passages cannot answer the question.
    """

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    insufficient_evidence: bool = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AnswerGenerationError(Exception):
    """Raised when answer generation fails for a documented, recoverable reason.

    Attributes:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MissingCredentialsError(AnswerGenerationError):
    """Raised when ``OPENROUTER_API_KEY`` or ``OPENROUTER_MODEL`` is not set."""


class EmptyRetrievalError(AnswerGenerationError):
    """Raised when FAISS returns no results for the query."""


# ---------------------------------------------------------------------------
# Prompt builder (pure function – easy to test in isolation)
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    retrieved: list[SearchResult],
) -> list[dict]:
    """Construct the OpenAI-style ``messages`` list for OpenRouter.

    Formats each ``SearchResult`` as a numbered evidence passage and appends
    a grounding instruction to the system message.

    Args:
        question:  The user's question.
        retrieved: Ranked list of retrieved :class:`SearchResult` objects.

    Returns:
        A two-element list  ``[system_message, user_message]``  ready to pass
        to ``client.chat.completions.create(messages=...)``.
    """
    evidence_blocks: list[str] = []
    for i, sr in enumerate(retrieved, start=1):
        evidence_blocks.append(
            f"[{i}] Source: {sr.doc_id}, page {sr.page_number}, "
            f"chunk {sr.chunk_index}\n{sr.text}"
        )

    evidence_text = "\n\n".join(evidence_blocks)

    system_content = (
        "You are a precise research assistant. "
        "Answer the user's question using ONLY the evidence passages below. "
        "Do not invent facts, add external knowledge, or fabricate citations. "
        f'If the evidence is insufficient, say exactly: "{_INSUFFICIENT_EVIDENCE_MARKER}". '
        "When citing, refer to sources by their bracketed numbers (e.g. [1], [2]).\n\n"
        f"Evidence passages:\n{evidence_text}"
    )

    user_content = question

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Client factory (dependency-injectable for testing)
# ---------------------------------------------------------------------------

def _make_openai_client(api_key: str, base_url: str):
    """Return an ``openai.OpenAI`` client pointed at OpenRouter.

    Kept as a separate function so tests can replace it via dependency
    injection without importing the real ``openai`` library.

    Args:
        api_key:  The OpenRouter API key.
        base_url: The OpenRouter base URL.

    Returns:
        An ``openai.OpenAI`` client instance.
    """
    from openai import OpenAI  # local import keeps the module import-time fast

    return OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """Orchestrates retrieval and OpenRouter generation to answer a question.

    Args:
        vector_store: A populated :class:`~retrieval.vector_store.VectorStore`.
        embedder:     A :class:`~retrieval.embedder.ChunkEmbedder` used to
                      embed the question.
        settings:     A ``Settings`` object with ``OPENROUTER_API_KEY``,
                      ``OPENROUTER_MODEL``, and optionally
                      ``OPENROUTER_BASE_URL``.  If ``None``, the default
                      ``Settings()`` is loaded from environment variables.
        client_factory: Callable that returns an OpenAI-compatible client.
                       Defaults to :func:`_make_openai_client`.  Override in
                       tests to inject a mock client.

    Example::

        generator = AnswerGenerator(store, embedder)
        result = generator.answer("What is the attention mechanism?")
        print(result.answer)
        for c in result.citations:
            print(c.doc_id, c.page_number)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: ChunkEmbedder,
        settings=None,
        client_factory=None,
    ) -> None:
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must be a VectorStore, got {type(vector_store).__name__!r}."
            )
        if not isinstance(embedder, ChunkEmbedder):
            raise TypeError(
                f"embedder must be a ChunkEmbedder, got {type(embedder).__name__!r}."
            )

        self._store = vector_store
        self._embedder = embedder
        self._settings = settings or _load_settings()
        self._client_factory = client_factory or _make_openai_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> AnswerResult:
        """Generate a grounded answer to *question*.

        Pipeline:
        1. Validate inputs.
        2. Validate credentials.
        3. Embed the question.
        4. Retrieve top-k chunks from FAISS.
        5. Build the prompt.
        6. Call OpenRouter.
        7. Return a structured :class:`AnswerResult`.

        Args:
            question: A non-empty user question.
            top_k:    Number of evidence chunks to retrieve.  Must be ≥ 1.

        Returns:
            An :class:`AnswerResult` with grounded ``answer`` text and
            ``citations`` sourced exclusively from FAISS results.

        Raises:
            ValueError:             If *question* is empty/whitespace or
                                    *top_k* is not a positive integer.
            MissingCredentialsError: If API key or model name is not configured.
            EmptyRetrievalError:    If FAISS returns no chunks for the question.
            AnswerGenerationError:  If the OpenRouter call fails or the response
                                    is malformed.
        """
        _validate_question(question)
        _validate_top_k(top_k)
        self._validate_credentials()

        # 3. Embed question
        logger.debug("Embedding question: %r", question[:80])
        query_vector = self._embedder.embed_query(question)

        # 4. Retrieve
        retrieved: list[SearchResult] = self._store.search(query_vector, top_k=top_k)
        if not retrieved:
            raise EmptyRetrievalError(
                "No relevant chunks found in the vector store for this question. "
                "Ensure the store has been populated with document embeddings before querying."
            )

        logger.debug("Retrieved %d chunks from FAISS.", len(retrieved))

        # 5. Build prompt
        messages = build_prompt(question, retrieved)

        # 6. Call OpenRouter
        answer_text = self._call_openrouter(messages)

        # 7. Build citations from real FAISS results only
        citations = [Citation.from_search_result(sr) for sr in retrieved]
        insufficient = _INSUFFICIENT_EVIDENCE_MARKER.lower() in answer_text.lower()

        return AnswerResult(
            question=question,
            answer=answer_text,
            citations=citations,
            insufficient_evidence=insufficient,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_credentials(self) -> None:
        """Raise :class:`MissingCredentialsError` if key or model are absent."""
        if not getattr(self._settings, "OPENROUTER_API_KEY", None):
            raise MissingCredentialsError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        if not getattr(self._settings, "OPENROUTER_MODEL", None):
            raise MissingCredentialsError(
                "OPENROUTER_MODEL is not set. "
                "Set it to a model ID such as 'openai/gpt-4o-mini'."
            )

    def _call_openrouter(self, messages: list[dict]) -> str:
        """Send *messages* to OpenRouter and return the reply text.

        Args:
            messages: OpenAI-style messages list.

        Returns:
            The assistant's reply as a plain string.

        Raises:
            AnswerGenerationError: On API errors or unexpected response shape.
        """
        base_url = (
            getattr(self._settings, "OPENROUTER_BASE_URL", None)
            or OPENROUTER_BASE_URL
        )
        api_key: str = self._settings.OPENROUTER_API_KEY
        model: str = self._settings.OPENROUTER_MODEL

        try:
            client = self._client_factory(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as exc:
            raise AnswerGenerationError(
                f"OpenRouter API call failed: {exc}"
            ) from exc

        # Extract text from the response
        try:
            text: str = completion.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise AnswerGenerationError(
                f"Malformed response from OpenRouter: {exc}"
            ) from exc

        if not text.strip():
            raise AnswerGenerationError(
                "OpenRouter returned an empty response. "
                "Try a different model or check your API quota."
            )

        logger.debug("Received %d-char answer from OpenRouter.", len(text))
        return text.strip()


# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------

def _load_settings():
    """Load the shared :class:`~config.settings.Settings` object.

    Extracted as a function so tests can easily patch it.
    """
    from config.settings import Settings  # local import avoids circular deps

    return Settings()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_question(question: object) -> None:
    """Raise :class:`ValueError` for blank or non-string questions."""
    if not isinstance(question, str):
        raise ValueError(
            f"question must be a str, got {type(question).__name__!r}."
        )
    if not question.strip():
        raise ValueError("question must not be empty or whitespace-only.")


def _validate_top_k(top_k: object) -> None:
    """Raise :class:`ValueError` for non-positive top_k."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(
            f"top_k must be a positive integer ≥ 1, got {top_k!r}."
        )
