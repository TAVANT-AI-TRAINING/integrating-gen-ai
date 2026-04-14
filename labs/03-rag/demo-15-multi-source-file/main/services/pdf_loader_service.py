"""
PDF Document Loader Service

This service handles loading documents from PDF files using PyPDFLoader.
Provides a clean, reusable interface for PDF document loading operations.
"""

from typing import List, Optional, Union
from pathlib import Path
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PDFLoaderService:
    """Service for loading documents from PDF files."""
    
    def __init__(self, docs_directory: Optional[Path] = None):
        """
        Initialize the PDF loader service.
        
        Args:
            docs_directory: Directory containing PDF files. If None, uses default.
        """
        self.docs_directory = docs_directory
    
    def _load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Internal method to load a single PDF file safely.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects (empty list if loading fails)
        """
        if not pdf_path.exists():
            logger.warning(f"    PDF file not found: {pdf_path}")
            return []
        
        try:
            pdf_loader = PyPDFLoader(str(pdf_path))
            documents = pdf_loader.load()
            logger.info(f"    Loaded {len(documents)} page(s) from PDF")
            logger.info(f"    Source: {pdf_path}")
            return documents
        except Exception as e:
            logger.error(f"    Error loading PDF {pdf_path}: {e}")
            return []
    
    def load_pdfs(
        self,
        source: Union[Path, List[Path]]
    ) -> List[Document]:
        """
        Load PDF file(s) safely. Handles single file or multiple files.
        
        Args:
            source: Single file path or list of file paths.
            
        Returns:
            List of Document objects (empty list if loading fails)
            
        Examples:
            # Load single file
            docs = service.load_pdfs(Path("file.pdf"))
            
            # Load multiple files
            docs = service.load_pdfs([Path("file1.pdf"), Path("file2.pdf")])
        """
        all_documents: List[Document] = []
        
        # Convert single path to list for uniform processing
        paths: List[Path] = source if isinstance(source, list) else [source]
        
        # Process each path
        for pdf_path in paths:
            documents = self._load_single_pdf(pdf_path)
            all_documents.extend(documents)
        
        return all_documents
