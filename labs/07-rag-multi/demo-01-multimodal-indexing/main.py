"""
Demo 01: Multi-Modal Document Indexing

Demonstrates indexing a PDF document across three modalities and querying them:

  text   — page text split into overlapping chunks
  table  — tables preserved as pipe-delimited text representations
  image  — images converted to descriptive captions via GPT-4o Vision

All three modality types are stored in a single ChromaDB collection using
`element_type` metadata, enabling both unified and modality-filtered search.

Key insight: by converting tables and images to rich text, a standard text
embedding pipeline captures multi-modal document content.

Usage:
    # Index a PDF (text + tables + images)
    uv run python main.py --pdf path/to/document.pdf

    # Index without image captioning (faster, no extra API cost)
    uv run python main.py --pdf path/to/document.pdf --no-images

    # Search after indexing
    uv run python main.py --query "quarterly revenue figures"
    uv run python main.py --query "org chart structure" --type image
    uv run python main.py --query "pricing breakdown" --type table
"""

import argparse
import base64
import os
from pathlib import Path

import fitz  # PyMuPDF — imports as fitz, installed as pymupdf
import pdfplumber
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "./chroma_multimodal")
COLLECTION     = "multimodal_docs"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 150
IMAGE_MIN_SIZE = 5000   # bytes — skip tiny icons / decorations

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

# ── LangChain components ──────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)   # captioning only

vectorstore = Chroma(
    collection_name=COLLECTION,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# ── Extraction: TEXT ──────────────────────────────────────────────────────────

def extract_text_chunks(pdf_path: str) -> list[Document]:
    """
    Extract page text from a PDF and split into overlapping chunks.
    Each chunk carries page number and element_type='text' metadata.
    """
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) > 50:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source":       pdf_path,
                        "page":         page_num,
                        "element_type": "text",
                    },
                ))

    chunks = splitter.split_documents(docs)
    print(f"  text  → {len(chunks)} chunks from {len(docs)} pages")
    return chunks


# ── Extraction: TABLES ────────────────────────────────────────────────────────

def _table_to_text(rows: list[list[str | None]], source: str, page: int) -> str:
    """Convert a pdfplumber table (list of rows) to a readable pipe-delimited block."""
    lines = []
    for i, row in enumerate(rows):
        cells = [str(c or "").strip() for c in row]
        lines.append(" | ".join(cells))
        if i == 0:                                      # header separator
            lines.append("-" * (sum(len(c) + 3 for c in cells)))
    return (
        f"TABLE from '{Path(source).name}' (page {page}):\n\n"
        + "\n".join(lines)
    )


def extract_tables(pdf_path: str) -> list[Document]:
    """
    Extract all tables from a PDF and convert each to a structured text block.
    Tables with < 2 rows (header only) are skipped.
    """
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for raw_table in page.extract_tables() or []:
                # Filter empty rows
                rows = [r for r in raw_table if any(c for c in r)]
                if len(rows) < 2:
                    continue
                docs.append(Document(
                    page_content=_table_to_text(rows, pdf_path, page_num),
                    metadata={
                        "source":       pdf_path,
                        "page":         page_num,
                        "element_type": "table",
                    },
                ))

    print(f"  table → {len(docs)} tables")
    return docs


# ── Extraction: IMAGES ────────────────────────────────────────────────────────

def _caption_image(img_bytes: bytes, mime: str = "png") -> str:
    """
    Send a base64-encoded image to GPT-4o Vision and return a descriptive caption.
    The caption is what gets embedded — it enables text-based semantic search
    over visual content without CLIP or separate image embeddings.
    """
    b64 = base64.standard_b64encode(img_bytes).decode()
    msg = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url":    f"data:image/{mime};base64,{b64}",
                "detail": "high",
            },
        },
        {
            "type": "text",
            "text": (
                "Describe this image from a business document. Include:\n"
                "1. Content type (chart, diagram, photo, screenshot, etc.)\n"
                "2. Key data points, labels, axis values, or text visible\n"
                "3. Business context or purpose of this image\n"
                "Be specific and factual."
            ),
        },
    ])
    return vision_llm.invoke([msg]).content


def extract_images_with_captions(pdf_path: str) -> list[Document]:
    """
    Extract images from a PDF using PyMuPDF, then caption each with GPT-4o Vision.
    Small images (< IMAGE_MIN_SIZE bytes) are skipped to avoid captioning icons.
    """
    docs   = []
    source = Path(pdf_path).name
    pdf    = fitz.open(pdf_path)

    for page_num, page in enumerate(pdf, start=1):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                img_data  = pdf.extract_image(xref)
                img_bytes = img_data["image"]
                mime      = img_data.get("ext", "png")

                if len(img_bytes) < IMAGE_MIN_SIZE:
                    continue

                print(f"    captioning image on page {page_num} "
                      f"({len(img_bytes) // 1024} KB)...")
                caption = _caption_image(img_bytes, mime)

                docs.append(Document(
                    page_content=(
                        f"IMAGE from '{source}' (page {page_num}):\n\n{caption}"
                    ),
                    metadata={
                        "source":       pdf_path,
                        "page":         page_num,
                        "element_type": "image",
                    },
                ))
            except Exception as exc:
                print(f"    image skipped (page {page_num}): {exc}")

    pdf.close()
    print(f"  image → {len(docs)} image captions")
    return docs


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_pdf(pdf_path: str, caption_images: bool = True) -> dict:
    """
    Parse a PDF into all three modalities and add to the vector store.

    Flow:
        PDF → pdfplumber (text + tables) + PyMuPDF (images)
            → GPT-4o Vision captions for images
            → langchain Documents with element_type metadata
            → ChromaDB unified collection
    """
    print(f"\nIndexing: {pdf_path}")
    print("-" * 60)

    text_docs  = extract_text_chunks(pdf_path)
    table_docs = extract_tables(pdf_path)
    image_docs = extract_images_with_captions(pdf_path) if caption_images else []

    all_docs = text_docs + table_docs + image_docs
    if all_docs:
        vectorstore.add_documents(all_docs)
        print(f"\n  Indexed {len(all_docs)} total documents into '{COLLECTION}'")
    else:
        print("\n  No documents extracted — check that the PDF has readable content.")

    return {"text": len(text_docs), "tables": len(table_docs),
            "images": len(image_docs), "total": len(all_docs)}


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search(query: str, k: int = 4, element_type: str | None = None) -> list[Document]:
    """
    Retrieve documents from the vector store.
    Pass element_type='text', 'table', or 'image' to filter by modality.
    Leave as None to search across all modalities simultaneously.
    """
    filter_dict = {"element_type": element_type} if element_type else None
    return vectorstore.similarity_search(query, k=k, filter=filter_dict)


def print_results(docs: list[Document]):
    if not docs:
        print("  No results found.")
        return
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        label = meta.get("element_type", "?").upper()
        page  = meta.get("page", "?")
        src   = Path(meta.get("source", "")).name
        print(f"\n  [{i}] {label}  |  {src}  |  page {page}")
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"      {preview}...")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal PDF indexing demo — text, tables, and image captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py --pdf annual_report.pdf
  uv run python main.py --pdf annual_report.pdf --no-images
  uv run python main.py --query "quarterly revenue"
  uv run python main.py --query "bar chart" --type image
  uv run python main.py --query "pricing table" --type table --k 3
        """,
    )
    parser.add_argument("--pdf",       help="Path to PDF file to index")
    parser.add_argument("--query",     help="Search query to run")
    parser.add_argument("--type",      choices=["text", "table", "image"],
                        help="Filter results by modality type")
    parser.add_argument("--k",         type=int, default=4,
                        help="Number of results to return (default: 4)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image captioning (faster, no Vision API cost)")
    args = parser.parse_args()

    if not args.pdf and not args.query:
        parser.print_help()
        return

    if args.pdf:
        stats = index_pdf(args.pdf, caption_images=not args.no_images)
        print(f"\n  Summary: {stats}")

    if args.query:
        filter_label = f" [filter: {args.type}]" if args.type else " [all modalities]"
        print(f"\nSearching: '{args.query}'{filter_label}")
        results = search(args.query, k=args.k, element_type=args.type)
        print_results(results)


if __name__ == "__main__":
    main()
