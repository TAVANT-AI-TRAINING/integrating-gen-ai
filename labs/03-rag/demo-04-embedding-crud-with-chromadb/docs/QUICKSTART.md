# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup Environment

```bash
# Install dependencies
uv sync

# Create .env file
cp .env.example .env

# Add your OpenAI API key to .env
# OPENAI_API_KEY=sk-your-key-here
```

### Step 2: Run the Server

```bash
uv run python main.py
```

Server runs at: http://localhost:8000

### Step 3: Test the API

**Option A: Interactive Docs (Easiest!)**
Open in browser: http://localhost:8000/docs

**Option B: Run Test Script**

```bash
./test_api.sh
```

**Option C: Manual cURL Commands**

```bash
# Add a document
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "test-001",
    "text": "This is a test document",
    "metadata": {"category": "test"}
  }'

# Get the document
curl http://localhost:8000/documents/test-001

# Search for similar documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "test document",
    "n_results": 5
  }'
```

## 📖 API Quick Reference

| Method | Endpoint              | Purpose             |
| ------ | --------------------- | ------------------- |
| GET    | `/`                   | Health check        |
| POST   | `/documents`          | Create new document |
| GET    | `/documents/{doc_id}` | Get document by ID  |
| PUT    | `/documents/{doc_id}` | Update document     |
| DELETE | `/documents/{doc_id}` | Delete document     |
| POST   | `/query`              | Semantic search     |
| GET    | `/documents`          | List all documents  |

## 💡 Key Concepts

- **Embedding**: Text converted to numerical vector (1536 dimensions)
- **ChromaDB**: Vector database for storing and searching embeddings
- **Semantic Search**: Find similar documents by meaning, not keywords
- **CRUD**: Create, Read, Update, Delete operations

## 🎯 What Makes This Simple?

1. ✅ Single file (`main.py`) - everything in one place
2. ✅ Clear comments - explains every step
3. ✅ Basic setup - no complex configuration
4. ✅ OpenAI embeddings - no model training needed
5. ✅ Persistent storage - data saved automatically

## 📚 Learn More

Check [README.md](README.md) for:

- Detailed explanations
- More examples
- Troubleshooting
- Learning notes
