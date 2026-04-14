"""
Services Package

This package provides modular services for the complete RAG ingestion pipeline:
- Document loaders (PDF, text, HTML, web)
- Document processing and chunking
- Metadata management
- File type detection
- Embedding and vector storage
- Complete injection pipeline orchestration
"""

from main.services.pdf_loader_service import PDFLoaderService
from main.services.text_loader_service import TextLoaderService
from main.services.html_loader_service import HTMLLoaderService
from main.services.web_loader_service import WebLoaderService
from main.services.chunking_service import ChunkingService
from main.services.document_processing_service import FileTypeDetector

# Note: document_processing_service, injection_pipeline_service, and embedding_service
# are available but imported directly to avoid circular dependencies:
# from main.services import document_processing_service
# from main.services import injection_pipeline_service
# from main.services import embedding_service

__all__ = [
    "PDFLoaderService",
    "TextLoaderService",
    "HTMLLoaderService",
    "WebLoaderService",
    "ChunkingService",
    "FileTypeDetector",
]

