# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup Environment

```bash
# Install dependencies (includes LangChain)
uv sync

# Create .env file
cp .env.example .env

# Add your OpenAI API key to .env
# OPENAI_API_EMBEDDING_KEY=sk-your-key-here
```

### Step 2: Run the Server

```bash
uv run python main.py
```

Server runs at: http://localhost:8000

### Step 3: Test the API

**Option A: Interactive Docs (Recommended!)**
Open in browser: http://localhost:8000/docs

**Option B: Run Test Script**

```bash
./test_api.sh
```

**Option C: Quick cURL Test**

```bash
# Add a document (LangChain handles embeddings automatically!)
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "test-001",
    "text": "This is a test document",
    "metadata": {"category": "test"}
  }'

# Search semantically (LangChain makes it easy!)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "test document",
    "n_results": 5
  }'
```

## 🎯 Why This Demo?

This is Demo 05 - it uses **LangChain** to make everything simpler:

- ✅ **Less code** - LangChain handles complexity
- ✅ **Cleaner** - High-level abstractions
- ✅ **Easier** - One-line operations
- ✅ **Production-ready** - Industry-standard patterns

## 📊 Comparison

| Operation        | Demo 04 (Direct)         | Demo 05 (LangChain) |
| ---------------- | ------------------------ | ------------------- |
| Create embedding | Manual API call          | Automatic           |
| Store document   | 4-5 lines                | 1-2 lines           |
| Search           | Manual embedding + query | 1 line              |
| Code complexity  | Medium                   | Low                 |

## 💡 Key Differences from Demo 04

1. **No manual OpenAI API calls** - LangChain handles it
2. **Document class** - Structured data format
3. **One-line semantic search** - `similarity_search()`
4. **Less error handling needed** - LangChain abstracts it
5. **More maintainable** - Industry-standard approach

## 📚 Learn More

- [README.md](README.md) - Full documentation
- [LANGCHAIN_BENEFITS.md](LANGCHAIN_BENEFITS.md) - Why LangChain?
- Compare with [Demo 04](../demo-04-embedding-crud-with-chromadb/) to see the difference!
