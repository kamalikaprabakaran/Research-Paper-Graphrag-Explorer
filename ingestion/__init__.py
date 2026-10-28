"""Ingestion package – PDF loading and text extraction."""

from ingestion.pdf_loader import DocumentMetadata, PageData, PDFDocument, load_pdf

__all__ = ["load_pdf", "PDFDocument", "PageData", "DocumentMetadata"]
