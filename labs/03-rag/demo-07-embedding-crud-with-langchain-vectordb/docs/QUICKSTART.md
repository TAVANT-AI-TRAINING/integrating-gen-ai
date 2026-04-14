# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
uv sync
```

This installs support for BOTH ChromaDB and Pinecone!

### Step 2: Choose Your Database

**Option A: ChromaDB (Easiest for Learning)**

```bash
# Create .env file
cp .env.example .env

# Edit .env:
OPENAI_API_EMBEDDING_KEY=sk-your-key-here
VECTOR_DB=chromadb
```

**Option B: Pinecone (For Cloud/Production)**

```bash
# Create .env file
cp .env.example .env

# Edit .env:
OPENAI_API_EMBEDDING_KEY=sk-your-key-here
VECTOR_DB=pinecone
PINECONE_API_KEY=your-pinecone-key
```

### Step 3: Run the Server

```bash
uv run python main.py
```

The server will automatically use your selected database!

```
✓ Selected Vector Database: CHROMADB
# or
✓ Selected Vector Database: PINECONE
```

## 📖 Test the API

**Interactive Docs:**
http://localhost:8000/docs

**Quick Test:**

```bash
# Add a document
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "test-001", "text": "Test document", "metadata": {"category": "test"}}'

# Search
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "test", "n_results": 5}'
```

## 🔄 Switch Databases

### From ChromaDB to Pinecone

1. Edit `.env`:

   ```
   VECTOR_DB=pinecone
   PINECONE_API_KEY=your-key
   ```

2. Restart server:

   ```bash
   # Stop server (Ctrl+C)
   uv run python main.py
   ```

3. **That's it!** Same code, different database! 🎉

### From Pinecone to ChromaDB

1. Edit `.env`:

   ```
   VECTOR_DB=chromadb
   ```

2. Restart server

Done!

## 💡 Key Concept

**One codebase, multiple backends:**

```python
# This code works with BOTH databases!
vectorstore.add_documents([doc])
results = vectorstore.similarity_search(query)
vectorstore.delete([id])
```

**Just change configuration:**

```env
VECTOR_DB=chromadb  # or pinecone
```

## 🎯 Recommendations

**For Learning:**

- Start with `VECTOR_DB=chromadb`
- No cloud account needed
- Fast and free

**For Production:**

- Switch to `VECTOR_DB=pinecone`
- Auto-scaling
- Managed infrastructure

**For Development:**

- Use ChromaDB locally
- Use Pinecone in staging/production
- Same code everywhere!

## 📊 Quick Comparison

| Database     | Setup             | Speed              | Cost          | Best For    |
| ------------ | ----------------- | ------------------ | ------------- | ----------- |
| **ChromaDB** | ✅ Easy           | ⚡ Fast            | 💰 Free       | Development |
| **Pinecone** | 📝 Account needed | 🌐 Network latency | 💳 Paid tiers | Production  |

## 🚨 Common Issues

### "PINECONE_API_KEY not found"

You set `VECTOR_DB=pinecone` but didn't add your API key.

- Get key from [pinecone.io](https://www.pinecone.io/)
- Add to `.env`: `PINECONE_API_KEY=your-key`

### Port 8000 already in use

Stop other servers or change port in `main.py`.

### ChromaDB permission error

Ensure write permissions in project directory.

## 📚 Learn More

- [README.md](README.md) - Full documentation
- [ABSTRACTION_POWER.md](ABSTRACTION_POWER.md) - Why this is amazing
- Compare with:
  - [Demo 04](../demo-04-embedding-crud-with-chromadb/) - Direct ChromaDB
  - [Demo 06](../demo-06-embedding-crud-with-pinecone/) - Direct Pinecone
  - [Demo 05](../demo-05-embedding-crud-with-langchain-chromadb/) - LangChain basics

## 🎉 The Magic

Change ONE line:

```env
VECTOR_DB=chromadb  →  VECTOR_DB=pinecone
```

Zero code changes needed! That's the power of LangChain! 🦜🔗
