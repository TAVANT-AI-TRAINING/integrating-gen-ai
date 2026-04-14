# Quick Start Guide

## 🚀 Get Started in 4 Steps

### Step 1: Get Pinecone API Key

1. Visit [https://www.pinecone.io/](https://www.pinecone.io/)
2. Sign up for a free account
3. Create a new project
4. Copy your API key from the dashboard

### Step 2: Setup Environment

```bash
# Install dependencies
uv sync

# Create .env file
cp .env.example .env

# Edit .env and add your keys:
# OPENAI_API_EMBEDDING_KEY=sk-your-openai-key
# PINECONE_API_KEY=your-pinecone-key
```

### Step 3: Run the Server

```bash
uv run python main.py
```

**Note:** The Pinecone index will be created automatically on first run!

Server runs at: http://localhost:8000

### Step 4: Test the API

**Option A: Interactive Docs (Easiest!)**
Open in browser: http://localhost:8000/docs

**Option B: Run Test Script**

```bash
./test_api.sh
```

**Option C: Manual cURL Commands**

```bash
# Add a document (stored in Pinecone cloud!)
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

| Method | Endpoint              | Purpose              |
| ------ | --------------------- | -------------------- |
| GET    | `/`                   | Health check & stats |
| POST   | `/documents`          | Create new document  |
| GET    | `/documents/{doc_id}` | Get document by ID   |
| PUT    | `/documents/{doc_id}` | Update document      |
| DELETE | `/documents/{doc_id}` | Delete document      |
| POST   | `/query`              | Semantic search      |
| GET    | `/documents`          | Get vector count     |

## 💡 Key Concepts

- **Pinecone**: Managed cloud vector database
- **Index**: Collection of vectors (created automatically)
- **Upsert**: Insert or update operation
- **Serverless**: Auto-scaling infrastructure
- **Metadata**: Store original text and attributes

## 🎯 Why Pinecone?

1. ✅ **Fully managed** - No server setup needed
2. ✅ **Auto-scaling** - Handles any load
3. ✅ **Production-ready** - Built for scale
4. ✅ **Cloud-native** - Global availability
5. ✅ **Free tier** - Great for learning

## 🆚 vs ChromaDB (Demo 04)

| Feature     | ChromaDB    | Pinecone   |
| ----------- | ----------- | ---------- |
| Storage     | Local files | Cloud      |
| Scaling     | Manual      | Automatic  |
| Maintenance | You         | Managed    |
| Best for    | Development | Production |

## ⚠️ Important Notes

1. **Index Creation**: Takes 30-60 seconds on first run
2. **Free Tier**: 100K vectors, 1 index
3. **No List All**: Pinecone doesn't list all vectors (by design)
4. **Network Latency**: ~50-200ms per operation (cloud-based)

## 📚 Learn More

- [README.md](README.md) - Full documentation
- [PINECONE_VS_CHROMA.md](PINECONE_VS_CHROMA.md) - Detailed comparison
- Compare with [Demo 04](../demo-04-embedding-crud-with-chromadb/) to see the difference!
