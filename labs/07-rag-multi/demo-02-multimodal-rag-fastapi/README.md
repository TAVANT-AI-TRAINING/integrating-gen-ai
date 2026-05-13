# Demo 02 — Multi-Modal RAG FastAPI Service

A REST API that extends the demo-12 text-only RAG service with **multi-modal PDF support** — indexing text, tables, and images in a single vector store and answering questions that span all three modalities.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest/file` | Upload a PDF; returns per-modality chunk counts |
| `POST` | `/retrieve/search` | Semantic search with optional modality filter |
| `POST` | `/generate/rag` | RAG answer with modality-labelled citations |
| `GET`  | `/stats` | Document counts in the vector store |
| `GET`  | `/health` | Service configuration |
| `GET`  | `/docs` | Swagger UI |

## Setup

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env

uv sync
uv run uvicorn main:app --reload --port 8000
```

## Usage

### 1. Ingest a PDF
```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@annual_report.pdf"
```
Response:
```json
{
  "filename": "annual_report.pdf",
  "text_chunks": 42,
  "table_chunks": 6,
  "image_chunks": 4,
  "total": 52
}
```

### 2. Search across all modalities
```bash
curl -X POST http://localhost:8000/retrieve/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quarterly revenue by region", "k": 5}'
```

### 3. Search a specific modality
```bash
# Tables only
curl -X POST http://localhost:8000/retrieve/search \
  -H "Content-Type: application/json" \
  -d '{"query": "EMEA revenue Q3", "k": 3, "element_type": "table"}'

# Images only
curl -X POST http://localhost:8000/retrieve/search \
  -H "Content-Type: application/json" \
  -d '{"query": "bar chart comparing regions", "element_type": "image"}'
```

### 4. Generate a RAG answer
```bash
curl -X POST http://localhost:8000/generate/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the top performing regions in Q3 and by how much?"}'
```
Response:
```json
{
  "query": "What were the top performing regions...",
  "answer": "Based on the Q3 report, Americas was the top performer...",
  "sources": [
    {"element_type": "table", "page": 4, "source": "annual_report.pdf", "excerpt": "TABLE ..."},
    {"element_type": "image", "page": 5, "source": "annual_report.pdf", "excerpt": "IMAGE ... bar chart showing ..."}
  ],
  "modalities_used": ["image", "table", "text"],
  "context_chunks": 6
}
```

## How it differs from demo-12

| Feature | demo-12 | demo-02 |
|---------|---------|---------|
| Ingests text PDFs | ✓ | ✓ |
| Ingests tables from PDFs | — | ✓ (pipe-delimited text) |
| Ingests images from PDFs | — | ✓ (GPT-4o Vision captions) |
| Modality filter on retrieval | — | ✓ `element_type` param |
| Modality labels in citations | — | ✓ TEXT / TABLE / IMAGE |

## Architecture

```
PDF Upload
  │
  ├─ pdfplumber.extract_text()   → text chunks  ─────┐
  ├─ pdfplumber.extract_tables() → pipe text   ──────┤→ ChromaDB
  └─ PyMuPDF + GPT-4o Vision     → captions   ───────┘  element_type metadata

Query → similarity_search(filter={element_type}) → text_llm → answer
```

> **Note**: Image captioning calls `gpt-4o` once per image during ingestion.
> Text generation uses `gpt-4o-mini` (configurable via `OPENAI_MODEL`).
