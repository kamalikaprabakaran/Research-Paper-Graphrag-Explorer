"""Retrieval package – text chunking and (future) embedding/search."""

from retrieval.chunker import TextChunk, chunk_document, chunk_pages

__all__ = ["chunk_document", "chunk_pages", "TextChunk"]
