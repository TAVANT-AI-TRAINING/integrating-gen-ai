"""
HTML Document Loader Service
This service handles loading documents from HTML files using BSHTMLLoader.
Provides a clean, reusable interface for HTML document loading operations.
"""

from typing import List, Optional, Union
from pathlib import Path
import logging
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HTMLLoaderService:
    """Service for loading documents from HTML files."""
    
    def __init__(self):
        """Initialize the HTML loader service."""
        pass
    
    def _load_single_html_file(self, file_path: Path) -> List[Document]:
        """
        Internal method to load a single HTML file safely.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            List of Document objects (empty list if loading fails)
        """
        if not file_path.exists():
            logger.warning(f"HTML file not found: {file_path}")
            return []
        
        try:
            # ============================================================================
            # LANGCHAIN: HTML LOADER INITIALIZATION AND DOCUMENT EXTRACTION
            # ============================================================================
            # These two lines perform the core LangChain document loading operation:
            # 1. Initialize BSHTMLLoader with file path (LangChain loader)
            # 2. Extract HTML content as Document object(s)
            # ============================================================================
            html_loader = BSHTMLLoader(str(file_path))
            documents = html_loader.load()
            # ============================================================================
            
            logger.info(f"Loaded HTML file: {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading HTML file {file_path}: {e}")
            return []
    
    def load_html_files(
        self,
        source: Union[Path, List[Path]]
    ) -> List[Document]:
        """
        Load HTML file(s) safely. Handles single file or multiple files.
        
        Args:
            source: Single file path or list of file paths.
            
        Returns:
            List of Document objects (empty list if loading fails)
            
        Examples:
            # Load single file
            docs = service.load_html_files(Path("file.html"))
            
            # Load multiple files
            docs = service.load_html_files([Path("file1.html"), Path("file2.html")])
        """
        all_documents: List[Document] = []
        
        # Convert single path to list for uniform processing
        paths: List[Path] = source if isinstance(source, list) else [source]
        
        # Process each path
        for file_path in paths:
            documents = self._load_single_html_file(file_path)
            all_documents.extend(documents)
        
        return all_documents
