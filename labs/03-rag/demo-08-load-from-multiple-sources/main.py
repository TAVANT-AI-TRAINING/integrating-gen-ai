"""
Demo 08: Load Documents from Multiple Sources

This demo shows how to load documents from different sources using LangChain loaders:
- PDF files (one Document per page)
- Text files (one Document per file)
- Web pages (one Document per URL)

All loaders produce standardized Document objects with page_content and metadata.

Usage:
    uv run python main.py
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document


# Configuration
DOCS_DIR = Path("Documents")
PDF_FILE = DOCS_DIR / "company_policy.pdf"
WEB_URL = "https://www.python.org/"


def load_pdf_documents() -> List[Document]:
    """
    Load PDF file using PyPDFLoader.
    
    Creates one Document per page.
    Metadata includes: source, page number
    
    Returns:
        List of Document objects (one per page)
    """
    print("\n[1] Loading PDF document...")
    print(f"    File: {PDF_FILE}")
    
    if not PDF_FILE.exists():
        print(f"    ⚠️  PDF file not found: {PDF_FILE}")
        return []
    
    try:
        loader = PyPDFLoader(str(PDF_FILE))
        documents = loader.load()
        print(f"    ✓ Loaded {len(documents)} page(s)")
        return documents
    except Exception as e:
        print(f"    ✗ Error loading PDF: {e}")
        return []


def load_text_documents() -> List[Document]:
    """
    Load all .txt files from Documents directory.
    
    Creates one Document per file.
    Metadata includes: source
    
    Returns:
        List of Document objects (one per file)
    """
    print("\n[2] Loading text files...")
    print(f"    Directory: {DOCS_DIR}")
    
    all_docs = []
    txt_files = list(DOCS_DIR.glob("*.txt"))
    
    if not txt_files:
        print("    ⚠️  No .txt files found")
        return []
    
    print(f"    Found {len(txt_files)} text file(s)")
    
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            documents = loader.load()
            all_docs.extend(documents)
            print(f"    ✓ Loaded: {txt_file.name}")
        except Exception as e:
            print(f"    ✗ Error loading {txt_file.name}: {e}")
    
    return all_docs


def load_web_documents() -> List[Document]:
    """
    Load web page using WebBaseLoader.
    
    Creates one Document per URL.
    Metadata includes: source, title, language
    
    Returns:
        List of Document objects (one per URL)
    """
    print("\n[3] Loading web page...")
    print(f"    URL: {WEB_URL}")
    
    try:
        loader = WebBaseLoader(WEB_URL)
        documents = loader.load()
        print(f"    ✓ Loaded {len(documents)} document(s)")
        if documents:
            print(f"    ✓ Content length: {len(documents[0].page_content):,} characters")
        return documents
    except Exception as e:
        print(f"    ✗ Error loading web page: {e}")
        return []


def inspect_documents(documents: List[Document]) -> None:
    """
    Display information about loaded documents.
    
    Args:
        documents: List of Document objects to inspect
    """
    print("\n" + "=" * 70)
    print("DOCUMENT INSPECTION")
    print("=" * 70)
    
    if not documents:
        print("⚠️  No documents were loaded!")
        return
    
    print(f"\n✓ Total documents loaded: {len(documents)}")
    
    # Group by source type
    pdf_docs = [d for d in documents if d.metadata.get('source', '').endswith('.pdf')]
    txt_docs = [d for d in documents if d.metadata.get('source', '').endswith('.txt')]
    web_docs = [d for d in documents if 'http' in d.metadata.get('source', '')]
    
    print(f"\nBreakdown by type:")
    print(f"  • PDF pages: {len(pdf_docs)}")
    print(f"  • Text files: {len(txt_docs)}")
    print(f"  • Web pages: {len(web_docs)}")
    
    # Show details for first few documents
    print("\n" + "=" * 70)
    print("DOCUMENT DETAILS (First 3)")
    print("=" * 70)
    
    for i, doc in enumerate(documents[:3], 1):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        # Show page number if it's a PDF
        if 'page' in doc.metadata:
            print(f"Page: {doc.metadata['page']}")
        
        content_length = len(doc.page_content)
        word_count = len(doc.page_content.split())
        print(f"Length: {content_length:,} characters, {word_count:,} words")
        
        # Show content preview (first 150 characters)
        preview = doc.page_content[:150].replace('\n', ' ').strip()
        print(f"Preview: {preview}...")
    
    # Show complete metadata for first document
    if documents:
        print("\n" + "=" * 70)
        print("METADATA EXAMPLE (First Document)")
        print("=" * 70)
        for key, value in documents[0].metadata.items():
            print(f"  {key}: {value}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("DEMO 08: LOAD DOCUMENTS FROM MULTIPLE SOURCES")
    print("=" * 70)
    
    # Load documents from all sources
    pdf_docs = load_pdf_documents()
    text_docs = load_text_documents()
    web_docs = load_web_documents()
    
    # Combine all documents
    all_documents = pdf_docs + text_docs + web_docs
    
    # Inspect the results
    inspect_documents(all_documents)
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("✓ PyPDFLoader creates one Document per page")
    print("✓ TextLoader creates one Document per file")
    print("✓ WebBaseLoader creates one Document per URL")
    print("✓ All Documents have standardized structure:")
    print("  - page_content: The actual text content")
    print("  - metadata: Dictionary with source information")
    print("✓ Metadata varies by source type (page numbers for PDFs)")
    print("=" * 70)


if __name__ == "__main__":
    main()
