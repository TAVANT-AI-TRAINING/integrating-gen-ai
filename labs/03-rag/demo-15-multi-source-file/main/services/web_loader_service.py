"""
Web Document Loader Service

This service handles loading documents from web pages using WebBaseLoader.
Provides a clean, reusable interface for web document loading operations.
Uses BeautifulSoup filtering to extract only relevant content (article, main, div elements).
"""

from typing import List, Optional, Union
import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

try:
    import bs4
except ImportError:
    bs4 = None

logger = logging.getLogger(__name__)


class WebLoaderService:
    """Service for loading documents from web pages."""
    
    def __init__(self):
        """Initialize the web loader service."""
        pass
    
    def _load_single_web_page(self, url: str) -> List[Document]:
        """
        Internal method to load a single web page safely.
        Uses BeautifulSoup filtering to extract only relevant content elements.
        Falls back to basic loader if filtered content is empty.
        
        Args:
            url: URL of the web page to load
            
        Returns:
            List of Document objects (empty list if loading fails)
        """
        try:
            documents = []
            
            # Try filtered loading first if bs4 is available
            if bs4 is not None:
                try:
                    web_loader = WebBaseLoader(
                        web_paths=[url],
                        bs_kwargs={
                            "parse_only": bs4.SoupStrainer(
                            name=["article", "main", "section"]
                        )
                        }
                    )
                    documents = web_loader.load()
                    
                    # Check if filtered content is empty, fallback to basic loader
                    if documents and len(documents[0].page_content.strip()) == 0:
                        logger.warning("    Filtered content is empty, falling back to basic loader")
                        documents = []
                except Exception as filter_error:
                    logger.warning(f"    Filtered loading failed: {filter_error}, trying basic loader")
                    documents = []
            
            # Use basic loader if filtered loading failed or returned empty content
            if not documents:
                web_loader = WebBaseLoader(web_paths=[url])
                documents = web_loader.load()
            
            logger.info(f"    Loaded {len(documents)} document(s) from web")
            if documents:
                logger.info(f"    Content length: {len(documents[0].page_content):,} characters")
            return documents
        except Exception as e:
            logger.error(f"    Error loading web page {url}: {e}")
            logger.error(f"    This may be due to network issues or site restrictions")
            return []
    
    def load_web_pages(
        self,
        source: Optional[Union[str, List[str]]] = None
    ) -> List[Document]:
        """
        Load web page(s) safely. Handles single URL or multiple URLs.
        
        Args:
            source: Single URL string, or list of URL strings.
            
        Returns:
            List of Document objects (empty list if loading fails)
            
        Examples:
            # Load single URL
            docs = service.load_web_pages("https://example.com")
            
            # Load multiple URLs
            docs = service.load_web_pages(["https://example.com", "https://another.com"])
        """
        all_documents: List[Document] = []
        
        if source is None:
            logger.warning("    No URL provided")
            return []
        
        # Convert single URL to list for uniform processing
        urls: List[str] = source if isinstance(source, list) else [source]
        
        logger.info(f"    Loading {len(urls)} web page(s)...")
        
        for url in urls:
            logger.info(f"    URL: {url}")
            documents = self._load_single_web_page(url)
            all_documents.extend(documents)
        
        return all_documents
