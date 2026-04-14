"""
Text Document Loader Service

This service handles loading documents from text files using TextLoader.
Provides a clean, reusable interface for text document loading operations.
"""

from typing import List, Optional, Union
from pathlib import Path
import logging
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextLoaderService:
    """Service for loading documents from text files."""
    
    DEFAULT_ENCODING = "utf-8"
    
    def __init__(self, docs_directory: Optional[Path] = None, encoding: str = DEFAULT_ENCODING):
        """
        Initialize the text loader service.
        
        Args:
            docs_directory: Directory containing text files. If None, uses default.
            encoding: Text file encoding (default: utf-8)
        """
        self.docs_directory = docs_directory
        self.encoding = encoding
    
    def _load_single_text_file(self, file_path: Path, encoding: Optional[str] = None) -> List[Document]:
        """
        Internal method to load a single text file safely.
        
        Args:
            file_path: Path to the text file
            encoding: Text file encoding. Uses instance default if None.
            
        Returns:
            List of Document objects (empty list if loading fails)
        """
        if not file_path.exists():
            logger.warning(f"    Text file not found: {file_path}")
            return []
        
        encoding_to_use = encoding or self.encoding
        
        try:
            txt_loader = TextLoader(str(file_path), encoding=encoding_to_use)
            documents = txt_loader.load()
            logger.info(f"    Loaded: {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"    Error loading text file {file_path}: {e}")
            return []
    
    def load_text_files(
        self,
        source: Union[Path, List[Path]],
        encoding: Optional[str] = None
    ) -> List[Document]:
        """
        Load text file(s) safely. Handles single file or multiple files.
        
        Args:
            source: Single file path or list of file paths.
            encoding: Text file encoding. Uses instance default if None.
            
        Returns:
            List of Document objects (empty list if loading fails)
            
        Examples:
            # Load single file
            docs = service.load_text_files(Path("file.txt"))
            
            # Load multiple files
            docs = service.load_text_files([Path("file1.txt"), Path("file2.txt")])
        """
        all_documents: List[Document] = []
        encoding_to_use = encoding or self.encoding
        
        # Convert single path to list for uniform processing
        paths: List[Path] = source if isinstance(source, list) else [source]
        
        # Process each path
        for file_path in paths:
            documents = self._load_single_text_file(file_path, encoding_to_use)
            all_documents.extend(documents)
        
        return all_documents
