"""
Multi-Source Document Loader: Load Documents from Multiple Sources

This script demonstrates loading documents from:
- PDF files (using PDFLoaderService)
- Text files (using TextLoaderService)
- Web pages (using WebLoaderService)

All documents are standardized into Document objects with preserved metadata.

Usage:
    uv run main.py
"""

import sys
from typing import List
from pathlib import Path
import logging
from langchain_core.documents import Document

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from main.services import PDFLoaderService, TextLoaderService, WebLoaderService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration (project_root is already set above)
DOCS_DIR = project_root / "Documents"
PDF_FILE = DOCS_DIR / "company_policy.pdf"
WEB_URL = "https://www.python.org/"
PREVIEW_LENGTH = 150


def load_all_sources() -> List[Document]:
    """
    Load documents from PDF, text files, and web using service classes.
    
    Returns:
        List of Document objects from all sources
    """
    all_docs: List[Document] = []
    
    logger.info("=" * 70)
    logger.info("LOADING DOCUMENTS FROM MULTIPLE SOURCES")
    logger.info("=" * 70)
    
    # Initialize services
    pdf_service = PDFLoaderService(docs_directory=DOCS_DIR)
    text_service = TextLoaderService(docs_directory=DOCS_DIR)
    web_service = WebLoaderService()
    
    # 1. Load PDF file
    logger.info("\n[1] Loading PDF document...")
    pdf_docs = pdf_service.load_pdfs(PDF_FILE)
    all_docs.extend(pdf_docs)
    
    # 2. Load all .txt files from DOCS_DIR
    logger.info("\n[2] Loading text files...")
    text_files = list(DOCS_DIR.glob("*.txt"))
    if text_files:
        text_docs = text_service.load_text_files(text_files)
        all_docs.extend(text_docs)
    
    # 3. Load web page
    logger.info("\n[3] Loading web page...")
    logger.info(f"    URL: {WEB_URL}")
    web_docs = web_service.load_web_pages(WEB_URL)
    all_docs.extend(web_docs)
    
    return all_docs


def inspect_documents(documents: List[Document]) -> None:
    """
    Inspect and display information about loaded documents.
    
    Args:
        documents: List of Document objects
    """
    logger.info("\n" + "=" * 70)
    logger.info("DOCUMENT INSPECTION")
    logger.info("=" * 70)
    
    if not documents:
        logger.warning("No documents were loaded!")
        return
    
    logger.info(f"\nTotal documents loaded: {len(documents)}")
    
    # Group by source type
    pdf_docs = [d for d in documents if d.metadata['source'].endswith('.pdf')]
    txt_docs = [d for d in documents if d.metadata['source'].endswith('.txt')]
    web_docs = [d for d in documents if 'http' in d.metadata['source']]
    
    logger.info(f"\nDocument breakdown by type:")
    logger.info(f"  PDF documents: {len(pdf_docs)}")
    logger.info(f"  Text documents: {len(txt_docs)}")
    logger.info(f"  Web documents: {len(web_docs)}")
    
    # Show detailed information for each document
    logger.info("\n" + "=" * 70)
    logger.info("DETAILED DOCUMENT INFORMATION")
    logger.info("=" * 70)
    
    for i, doc in enumerate(documents, 1):
        logger.info(f"\n--- Document {i} ---")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        # Show page number if it's a PDF
        if 'page' in doc.metadata:
            logger.info(f"Page: {doc.metadata['page']}")
        
        logger.info(f"Content length: {len(doc.page_content):,} characters")
        logger.info(f"Word count: {len(doc.page_content.split()):,} words")
        
        # Show content preview
        preview_length = min(PREVIEW_LENGTH, len(doc.page_content))
        preview = doc.page_content[:preview_length].replace('\n', ' ')
        logger.info(f"Preview: {preview}...")
    
    # Show metadata structure
    logger.info("\n" + "=" * 70)
    logger.info("METADATA STRUCTURE EXAMPLE")
    logger.info("=" * 70)
    logger.info("\nFirst document metadata:")
    for key, value in documents[0].metadata.items():
        logger.info(f"  {key}: {value}")


def main() -> None:
    """Main execution function."""
    logger.info("LO1 PRACTICE: LOADING DOCUMENTS FROM MULTIPLE SOURCES")
    logger.info("")
    
    # Load documents from all sources using services
    documents = load_all_sources()
    
    # Inspect the loaded documents
    inspect_documents(documents)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("KEY TAKEAWAYS")
    logger.info("=" * 70)
    logger.info("1. PDFLoaderService creates one Document per page")
    logger.info("2. TextLoaderService creates one Document per file")
    logger.info("3. WebLoaderService creates one Document per URL")
    logger.info("4. All services preserve metadata (source, page numbers, etc.)")
    logger.info("5. Document objects have a standardized structure:")
    logger.info("   - page_content: The actual text content")
    logger.info("   - metadata: Dictionary with source information")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
