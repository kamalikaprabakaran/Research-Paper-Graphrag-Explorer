"""Retrieval package – text chunking, local embeddings, and vector search."""

from retrieval.chunker import TextChunk, chunk_document, chunk_pages
from retrieval.embedder import (
    ChunkEmbedder,
    EmbeddingResult,
    DEFAULT_MODEL,
    embed_chunks,
    embed_query,
)
from retrieval.vector_store import SearchResult, VectorStore

__all__ = [
    # chunker
    "chunk_document",
    "chunk_pages",
    "TextChunk",
    # embedder
    "ChunkEmbedder",
    "EmbeddingResult",
    "DEFAULT_MODEL",
    "embed_chunks",
    "embed_query",
    # vector store
    "VectorStore",
    "SearchResult",
]
