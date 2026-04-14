"""
Document Loader Services Package

This package provides modular services for loading documents from various sources:
- PDF files
- Text files
- Web pages
"""

from main.services.pdf_loader_service import PDFLoaderService
from main.services.text_loader_service import TextLoaderService
from main.services.web_loader_service import WebLoaderService

__all__ = [
    "PDFLoaderService",
    "TextLoaderService",
    "WebLoaderService",
]

