"""
Document Processing Service
Database-independent utilities for loading and chunking documents.
This module handles multi-source loading (PDF, text, HTML, web) and splitting into chunks.

Uses modular service classes for document loading:
- PDFLoaderService for PDF files
- TextLoaderService for text files
- HTMLLoaderService for HTML files
- WebLoaderService for web pages
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document

from main.services import (
    PDFLoaderService,
    TextLoaderService,
    HTMLLoaderService,
    WebLoaderService,
    ChunkingService
)


# -----------------------------------------------------------------------------
# File Type Detection
# -----------------------------------------------------------------------------

class FileTypeDetector:
    """Utility class for detecting file types."""
    
    # File type mappings
    PDF_EXTENSIONS = {".pdf"}
    TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
    HTML_EXTENSIONS = {".html", ".htm"}
    SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | TEXT_EXTENSIONS | HTML_EXTENSIONS
    
    @classmethod
    def get_file_type(cls, file_path: Path) -> Optional[str]:
        """
        Get the file type category for a given file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type string ('pdf', 'text', 'html') or None if unsupported
        """
        suffix = file_path.suffix.lower()
        
        if suffix in cls.PDF_EXTENSIONS:
            return "pdf"
        elif suffix in cls.TEXT_EXTENSIONS:
            return "text"
        elif suffix in cls.HTML_EXTENSIONS:
            return "html"
        else:
            return None
    
    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """
        Check if a file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file type is supported, False otherwise
        """
        suffix = file_path.suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def categorize_files(cls, file_paths: List[Path]) -> Dict[str, List[Path]]:
        """
        Categorize multiple files by type.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary with file type as key and list of paths as value
        """
        categorized = {
            "pdf": [],
            "text": [],
            "html": [],
            "unsupported": []
        }
        
        for file_path in file_paths:
            file_type = cls.get_file_type(file_path)
            if file_type:
                categorized[file_type].append(file_path)
            else:
                categorized["unsupported"].append(file_path)
        
        return categorized
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported file extensions (with dots)
        """
        return sorted(list(cls.SUPPORTED_EXTENSIONS))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Load environment variables
try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")


# Processing configuration
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Default docs directory: project_root/ Documents
    DEFAULT_DOCS_DIR = Path(__file__).parent.parent.parent / "Documents"
    DOCS_DIR = Path(os.getenv("DOCS_DIR", str(DEFAULT_DOCS_DIR)))

    WEB_URL = os.getenv("WEB_URL", "https://python.langchain.com/docs/introduction/")

    logger.info(
        f"Document processing config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, docs_dir={DOCS_DIR}"
    )
except Exception as e:
    logger.error(f"Failed to configure document processing: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Helpers: configuration accessors
# -----------------------------------------------------------------------------

def get_chunk_size() -> int:
    return CHUNK_SIZE


def get_chunk_overlap() -> int:
    return CHUNK_OVERLAP




# -----------------------------------------------------------------------------
# Loader Services (Initialized once for reuse)
# -----------------------------------------------------------------------------

# Initialize service instances
_pdf_service = PDFLoaderService()
_text_service = TextLoaderService()
_html_service = HTMLLoaderService()
_web_service = WebLoaderService()
_chunking_service = ChunkingService(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


# -----------------------------------------------------------------------------
# Loader Functions (Wrapper functions for backward compatibility)
# -----------------------------------------------------------------------------

def load_pdf(file_path: Path) -> List[Document]:
    """
    Load a PDF file using PDFLoaderService.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects (one per page)
    """
    return _pdf_service.load_pdfs(file_path)


def load_text_file(file_path: Path) -> List[Document]:
    """
    Load a text file using TextLoaderService.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of Document objects
    """
    return _text_service.load_text_files(file_path)


def load_html_file(file_path: Path) -> List[Document]:
    """
    Load an HTML file using HTMLLoaderService.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        List of Document objects
    """
    return _html_service.load_html_files(file_path)


def load_web_page(url: str) -> List[Document]:
    """
    Load a web page using WebLoaderService.
    
    Args:
        url: URL of the web page
        
    Returns:
        List of Document objects
    """
    return _web_service.load_web_pages(url)




# -----------------------------------------------------------------------------
# Splitting / Processing
# -----------------------------------------------------------------------------

def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split documents into chunks using ChunkingService.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of Document objects (chunks)
    """
    return _chunking_service.chunk_documents(documents, chunk_size, chunk_overlap)


def process_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process documents into chunks with statistics using ChunkingService.
    
    Args:
        documents: List of Document objects to process
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Dictionary with chunks and processing statistics
    """
    return _chunking_service.chunk_documents_with_stats(
        documents,
        chunk_size=chunk_size or get_chunk_size(),
        chunk_overlap=chunk_overlap or get_chunk_overlap()
    )


def load_and_process_multi_source(
    pdf_files: Optional[List[Path]] = None,
    text_files: Optional[List[Path]] = None,
    html_files: Optional[List[Path]] = None,
    web_urls: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load documents from multiple sources and process them into chunks.
    
    Uses service classes for loading:
    - PDFLoaderService for PDF files
    - TextLoaderService for text files
    - HTMLLoaderService for HTML files
    - WebLoaderService for web URLs
    
    Args:
        pdf_files: Optional list of PDF file paths
        text_files: Optional list of text file paths
        html_files: Optional list of HTML file paths
        web_urls: Optional list of web page URLs
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Dictionary with processing results including chunks and statistics
    """
    all_docs: List[Document] = []

    logger.info("=" * 70)
    logger.info("LOADING DOCUMENTS FROM MULTIPLE SOURCES")
    logger.info("=" * 70)

    # Load PDF files using PDFLoaderService
    if pdf_files:
        pdf_docs = _pdf_service.load_pdfs(pdf_files)
        all_docs.extend(pdf_docs)

    # Load text files using TextLoaderService
    if text_files:
        text_docs = _text_service.load_text_files(text_files)
        all_docs.extend(text_docs)

    # Load HTML files using HTMLLoaderService
    if html_files:
        html_docs = _html_service.load_html_files(html_files)
        all_docs.extend(html_docs)

    # Load web pages using WebLoaderService
    if web_urls:
        web_docs = _web_service.load_web_pages(web_urls)
        all_docs.extend(web_docs)

    if not all_docs:
        raise ValueError("No documents were loaded from any source")

    logger.info(f"\nTotal documents loaded: {len(all_docs)}")
    logger.info(f"  Total characters: {sum(len(d.page_content) for d in all_docs):,}")

    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING DOCUMENTS")
    logger.info("=" * 70)

    result = process_documents(all_docs, chunk_size, chunk_overlap)
    result["source_documents"] = len(all_docs)
    return result


