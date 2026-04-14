# Demo 12: RAG Pipeline - FastAPI Service

This demo exposes a complete **RAG (Retrieval-Augmented Generation) pipeline** via REST API, combining functionality from demo-09 (ingestion), demo-10 (retrieval), and demo-11 (generation).

## What This Demo Provides

A production-ready FastAPI service with:

- **Ingestion Endpoints** (demo-09): Load and store documents
- **Retrieval Endpoints** (demo-10): Search and retrieve relevant chunks
- **Generation Endpoints** (demo-11): Generate LLM answers with context

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INGESTION (Demo-09)     RETRIEVAL (Demo-10)    GENERATION  │
│  ├─ POST /ingest/text    ├─ GET /retrieve/verify  (Demo-11) │
│  └─ POST /ingest/file    ├─ POST /retrieve/similarity       │
│                          └─ POST /retrieve/mmr   POST        │
│                                                   /generate  │
│                                                   /rag       │
├─────────────────────────────────────────────────────────────┤
│                    Vector Database                           │
│                  (ChromaDB or Pinecone)                      │
└─────────────────────────────────────────────────────────────┘
```

## Features

✅ **Complete RAG Pipeline** - All 3 phases via REST API  
✅ **FastAPI** - Modern, fast, auto-documented API  
✅ **File Upload** - Support for PDF and TXT files  
✅ **Flexible Retrieval** - Similarity search + MMR  
✅ **LLM Integration** - Generate answers with context  
✅ **Multi-Vector DB** - ChromaDB (local) or Pinecone (cloud)  
✅ **Auto Documentation** - Swagger UI + ReDoc  
✅ **Metadata Filtering** - Filter by source/properties  
✅ **Score Display** - See relevance scores

## Prerequisites

- Python 3.12+
- OpenAI API key
- (Optional) Pinecone API key for cloud vector DB

## Quick Start

### 1. Installation

```bash
# Navigate to demo-12
cd demo-12-rag-fastapi-service

# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required:
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
VECTOR_DB=chromadb  # or pinecone

# Optional (for Pinecone):
# PINECONE_API_KEY=your-key
# PINECONE_INDEX_NAME=your-index
```

### 3. Start Server

```bash
# Option 1: Using uvicorn directly
uvicorn main:app --reload --port 8000

# Option 2: Using uv
uv run uvicorn main:app --reload --port 8000

# Server will start at: http://localhost:8000
```

### 4. Access Documentation

Open your browser:

- **Swagger UI**: http://localhost:8000/docs (interactive)
- **ReDoc**: http://localhost:8000/redoc (clean docs)
- **Health Check**: http://localhost:8000/health

## API Endpoints

### General Endpoints

#### GET `/`

Root endpoint with API information

```bash
curl http://localhost:8000/
```

#### GET `/health`

Check API health and configuration

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "vector_db": "CHROMADB",
  "llm_model": "gpt-4o-mini",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

### Ingestion Endpoints (Demo-09)

#### POST `/ingest/text`

Ingest raw text into vector store

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Remote work policy allows up to 3 days per week with manager approval.",
    "metadata": {"source": "policy", "type": "remote_work"}
  }'
```

Response:

```json
{
  "status": "success",
  "chunks_created": 1,
  "message": "Successfully ingested 1 chunks into CHROMADB"
}
```

#### POST `/ingest/file`

Upload and ingest a file (PDF or TXT)

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@Documents/guidelines.txt"
```

Response:

```json
{
  "status": "success",
  "filename": "guidelines.txt",
  "documents_loaded": 1,
  "chunks_created": 3,
  "message": "Successfully ingested guidelines.txt into CHROMADB"
}
```

### Retrieval Endpoints (Demo-10)

#### GET `/retrieve/verify`

Verify vector store has data

```bash
curl http://localhost:8000/retrieve/verify
```

Response:

```json
{
  "status": "ready",
  "has_data": true,
  "chunk_count": 6,
  "sample_chunk": {
    "source": "guidelines.txt",
    "content_preview": "Company Policy Guidelines\n\n1. Remote Work Policy\nEmployees are authorized to work remotely up to 3 days per week with manager approval. Remote work requires maintaining availability during core...",
    "metadata": { "source": "guidelines.txt" }
  }
}
```

#### POST `/retrieve/similarity`

Similarity search with optional scores and filters

```bash
# Basic retrieval
curl -X POST http://localhost:8000/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the remote work policy?",
    "k": 3,
    "include_scores": true
  }'

# With metadata filter
curl -X POST http://localhost:8000/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query": "remote work",
    "k": 3,
    "filter": {"source": "guidelines.txt"}
  }'
```

Response:

```json
{
  "query": "What is the remote work policy?",
  "results": [
    {
      "content": "Remote Work Policy\nEmployees are authorized to work remotely up to 3 days per week with manager approval...",
      "content_preview": "Remote Work Policy\nEmployees are authorized to work remotely up to 3 days per week with manager approval. Remote work requires maintaining availab...",
      "metadata": { "source": "guidelines.txt" },
      "score": 0.3245,
      "relevance": "high"
    }
  ],
  "count": 3
}
```

**Score Interpretation**:

- **Lower score = Higher relevance** (distance-based metric)
- `< 0.6`: High relevance
- `0.6-0.8`: Medium relevance
- `> 0.8`: Low relevance

#### POST `/retrieve/mmr`

MMR (Maximum Marginal Relevance) search for diverse results

```bash
curl -X POST http://localhost:8000/retrieve/mmr \
  -H "Content-Type: application/json" \
  -d '{
    "query": "company policies",
    "k": 4,
    "fetch_k": 20
  }'
```

### Generation Endpoints (Demo-11)

#### POST `/generate/rag`

Generate answer using RAG (retrieval + LLM)

```bash
curl -X POST http://localhost:8000/generate/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many days can I work remotely?",
    "k": 3,
    "include_sources": true,
    "temperature": 0.0
  }'
```

Response:

```json
{
  "query": "How many days can I work remotely?",
  "answer": "Based on the company's remote work policy, employees are authorized to work remotely up to 3 days per week with manager approval. Remote work requires maintaining availability during core hours (10 AM - 3 PM).",
  "sources": [
    {
      "content": "Remote Work Policy\nEmployees are authorized to work remotely up to 3 days per week...",
      "content_preview": "Remote Work Policy\nEmployees are authorized to work remotely...",
      "metadata": { "source": "guidelines.txt" }
    }
  ],
  "context_count": 3
}
```

## Usage Examples

### Complete RAG Workflow

```bash
# Step 1: Ingest documents
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@Documents/guidelines.txt"

curl -X POST http://localhost:8000/ingest/file \
  -F "file=@Documents/policy.txt"

# Step 2: Verify data
curl http://localhost:8000/retrieve/verify

# Step 3: Test retrieval
curl -X POST http://localhost:8000/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{"query": "remote work", "k": 3, "include_scores": true}'

# Step 4: Generate answer
curl -X POST http://localhost:8000/generate/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the remote work guidelines?", "k": 3}'
```

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Ingest text
response = requests.post(
    f"{BASE_URL}/ingest/text",
    json={
        "text": "Employees get 15 vacation days per year.",
        "metadata": {"source": "hr_policy", "type": "vacation"}
    }
)
print(response.json())

# 2. Retrieve relevant chunks
response = requests.post(
    f"{BASE_URL}/retrieve/similarity",
    json={
        "query": "vacation days",
        "k": 3,
        "include_scores": True
    }
)
results = response.json()
print(f"Found {results['count']} results")

# 3. Generate answer
response = requests.post(
    f"{BASE_URL}/generate/rag",
    json={
        "query": "How many vacation days do employees get?",
        "k": 3,
        "include_sources": True
    }
)
answer = response.json()
print(f"Answer: {answer['answer']}")
```

## Configuration Options

### Vector Database

**ChromaDB (Local)**:

```env
VECTOR_DB=chromadb
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=company_policies
```

**Pinecone (Cloud)**:

```env
VECTOR_DB=pinecone
PINECONE_API_KEY=your-api-key
PINECONE_INDEX_NAME=company-policies-demo
```

### Chunking

```env
CHUNK_SIZE=1000          # Characters per chunk
CHUNK_OVERLAP=200        # Overlap between chunks
```

### LLM

```env
OPENAI_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo
```

## Demo Integration

This service combines:

| Demo        | Functionality        | Endpoints                               |
| ----------- | -------------------- | --------------------------------------- |
| **Demo-09** | Document ingestion   | `/ingest/text`, `/ingest/file`          |
| **Demo-10** | Retrieval strategies | `/retrieve/similarity`, `/retrieve/mmr` |
| **Demo-11** | LLM generation       | `/generate/rag`                         |

## Testing

### Using Swagger UI

1. Open http://localhost:8000/docs
2. Click "Try it out" on any endpoint
3. Fill in parameters
4. Click "Execute"

### Using cURL

See examples above for each endpoint.

### Using Python

```python
import requests

# Test health
r = requests.get("http://localhost:8000/health")
print(r.json())

# Ingest and query
requests.post("http://localhost:8000/ingest/text",
    json={"text": "Test content"})

result = requests.post("http://localhost:8000/generate/rag",
    json={"query": "test query", "k": 3})
print(result.json()['answer'])
```

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (no documents)
- `500`: Server Error

Example error response:

```json
{
  "detail": "No relevant documents found. Please ingest documents first using /ingest endpoints."
}
```

## Production Considerations

### Security

- Add API authentication (OAuth2, API keys)
- Enable CORS if needed
- Use HTTPS in production
- Validate and sanitize inputs

### Performance

- Add caching for frequent queries
- Use connection pooling
- Monitor rate limits
- Consider async processing for large files

### Deployment

```bash
# Production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Monitoring

- Add logging
- Track metrics (request count, latency, errors)
- Monitor vector DB performance
- Set up health checks

## Troubleshooting

### Issue: "Vector store is empty"

**Solution**: Ingest documents first using `/ingest/text` or `/ingest/file`

### Issue: "OPENAI_API_KEY not found"

**Solution**: Copy `.env.example` to `.env` and add your API key

### Issue: "No relevant documents found"

**Solution**:

1. Verify data exists: `GET /retrieve/verify`
2. Try different search queries
3. Ingest more documents

### Issue: Port already in use

**Solution**: Use a different port:

```bash
uvicorn main:app --port 8001
```

## Related Demos

- **demo-09**: RAG Ingestion Pipeline (standalone)
- **demo-10**: RAG Retrieval Pipeline (standalone)
- **demo-11**: Complete RAG with LLM (standalone)
- **demo-12**: FastAPI Service (this demo - combines all)

## License

MIT License - See main project LICENSE file

## Contributing

Feel free to submit issues and enhancement requests!
