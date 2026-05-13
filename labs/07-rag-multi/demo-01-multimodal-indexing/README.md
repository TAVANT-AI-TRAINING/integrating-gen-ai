# Demo 01 — Multi-Modal Document Indexing

Demonstrates how to index a PDF into **three modality types** in a single ChromaDB collection, enabling semantic search across text, tables, and images with a unified embedding pipeline.

## The Core Idea

| Modality | Extraction tool | How it becomes searchable |
|----------|----------------|--------------------------|
| **Text** | `pdfplumber` | Chunked directly → embedded |
| **Table** | `pdfplumber` | Converted to pipe-delimited text → embedded |
| **Image** | `PyMuPDF` + GPT-4o Vision | Captioned as descriptive text → embedded |

All three types land in the same ChromaDB collection with an `element_type` metadata field, enabling filtered or unified retrieval.

## Setup

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env

uv sync
```

## Usage

```bash
# Index a PDF (text + tables + images via GPT-4o Vision)
uv run python main.py --pdf annual_report.pdf

# Index without image captioning (no Vision API calls, faster)
uv run python main.py --pdf annual_report.pdf --no-images

# Search across all modalities
uv run python main.py --query "quarterly revenue breakdown"

# Filter to a specific modality
uv run python main.py --query "bar chart" --type image
uv run python main.py --query "pricing table" --type table
uv run python main.py --query "company overview" --type text

# Control result count
uv run python main.py --query "revenue figures" --k 6
```

## What to observe

1. **Text chunks** — standard dense retrieval over prose content
2. **Tables** — the pipe-delimited format preserves numeric structure; queries like *"Q3 EMEA revenue"* match table rows that would be lost if tables were treated as plain text
3. **Images** — the GPT-4o caption describes axis labels, chart types, and key values; a query like *"bar chart comparing regional sales"* retrieves the right image without CLIP or separate image embeddings

## Architecture

```
PDF
 ├── Text pages    → pdfplumber.extract_text()  → chunks → embeddings
 ├── Tables        → pdfplumber.extract_tables() → text repr → embeddings
 └── Images        → PyMuPDF.extract_image()
                       → GPT-4o Vision caption
                       → embedding
                              ↓
                       ChromaDB collection
                       element_type: text | table | image
```

## Dependencies

- `pdfplumber` — PDF text and table extraction
- `pymupdf` — PDF image extraction (imports as `fitz`)
- `openai` / `langchain-openai` — text embeddings + GPT-4o Vision
- `langchain-chroma` / `chromadb` — vector store
