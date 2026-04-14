# Quick Start - RAG FastAPI Service

Get your RAG API running in under 2 minutes!

## What You'll Build

A production-ready REST API with:
- Document ingestion (PDF, TXT)
- Semantic search retrieval
- LLM-powered answer generation

## 60-Second Setup

```bash
# 1. Navigate
cd demo-12-rag-fastapi-service

# 2. Install
uv venv && source .venv/bin/activate
uv pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env: Add your OPENAI_API_KEY

# 4. Start Server
uv run uvicorn main:app --reload --port 8000
```

## First API Calls

### 1. Check Health

```bash
curl http://localhost:8000/health
```

### 2. Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Remote work is allowed 3 days per week.",
    "metadata": {"source": "policy"}
  }'
```

### 3. Ask a Question

```bash
curl -X POST http://localhost:8000/generate/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many days can I work remotely?",
    "k": 3
  }'
```

## Interactive Testing

Open in your browser:
- **Swagger UI**: http://localhost:8000/docs
- Click "Try it out" on any endpoint
- Fill in parameters and click "Execute"

## Common Workflows

### Complete RAG Flow

```bash
# 1. Ingest documents
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@Documents/guidelines.txt"

# 2. Verify data
curl http://localhost:8000/retrieve/verify

# 3. Search
curl -X POST http://localhost:8000/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{"query": "remote work", "k": 3, "include_scores": true}'

# 4. Generate answer
curl -X POST http://localhost:8000/generate/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the remote work policy?", "k": 3}'
```

### Using Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Ingest
requests.post(f"{BASE_URL}/ingest/text", json={
    "text": "Vacation policy: 15 days per year"
})

# Generate answer
response = requests.post(f"{BASE_URL}/generate/rag", json={
    "query": "How many vacation days?",
    "k": 3
})
print(response.json()['answer'])
```

## Running Tests

```bash
# Bash script (requires jq)
./test_api.sh

# Python script
python test_api.py
```

## Key Features

### Ingestion
- Text input: `POST /ingest/text`
- File upload: `POST /ingest/file` (PDF, TXT)

### Retrieval
- Similarity search: `POST /retrieve/similarity`
- Diverse results (MMR): `POST /retrieve/mmr`
- With scores and filters

### Generation
- RAG answers: `POST /generate/rag`
- Includes source citations
- Adjustable temperature

## Configuration

### Quick Configs

**Local (ChromaDB)**:
```env
VECTOR_DB=chromadb
OPENAI_API_KEY=sk-your-key
```

**Cloud (Pinecone)**:
```env
VECTOR_DB=pinecone
OPENAI_API_KEY=sk-your-key
PINECONE_API_KEY=your-pinecone-key
```

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :8000

# Use different port
uvicorn main:app --port 8001
```

### "Vector store is empty"
```bash
# Ingest documents first
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here"}'
```

### "OPENAI_API_KEY not found"
```bash
# Copy and edit .env
cp .env.example .env
# Add your API key to .env
```

## Next Steps

1. **Explore API**: http://localhost:8000/docs
2. **Read full docs**: Check README.md
3. **Customize**: Modify chunk size, k values, temperature
4. **Deploy**: Add authentication, use production server

## Production Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn main:app --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Demo Integration

This service combines:
- **Demo-09**: Ingestion functionality
- **Demo-10**: Retrieval strategies
- **Demo-11**: LLM generation

All accessible via clean REST API! 🚀
