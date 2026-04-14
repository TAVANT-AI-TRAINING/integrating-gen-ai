# Quick Start Guide - Complete RAG Pipeline

Get started with the complete RAG pipeline in under 2 minutes!

## What You'll Build

A complete **Retrieval-Augmented Generation (RAG)** system that:

1. Loads documents (PDF, text, web)
2. Chunks and embeds them
3. Stores in vector database
4. Retrieves relevant information
5. Generates answers using LLM

## 60-Second Setup

```bash
# 1. Navigate to demo
cd demo-10-complete-rag-pipeline

# 2. Create environment
uv venv && source .venv/bin/activate

# 3. Install dependencies
uv pip install -e .

# 4. Configure
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here

# 5. Run!
uv run python main.py
```

## Configuration Options

### Minimal Setup (ChromaDB - Local)

```bash
# .env
OPENAI_API_KEY=sk-...
VECTOR_DB=chromadb
```

### Cloud Setup (Pinecone)

```bash
# .env
OPENAI_API_KEY=sk-...
VECTOR_DB=pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=company-policies
```

## What Happens When You Run

### Console Output Flow:

```
1. Configuration Summary
   ├── Vector database: CHROMADB
   ├── LLM model: gpt-4o-mini
   └── Chunk settings: 1000 chars, 200 overlap

2. Load Documents
   ├── PDF (5 pages)
   ├── Text files (2 files)
   └── Web page (Python.org)

3. Chunk Documents
   └── Created 127 chunks

4. Store with Embeddings
   └── Stored 127 chunks in ChromaDB

5. Demonstrate Retrieval
   ├── Compare k=2 vs k=4
   └── Show retriever interface

6. Test Complete RAG
   ├── Query: "What is the remote work policy?"
   ├── Retrieved: 3 relevant chunks
   └── Generated: Complete answer from context
```

## Test Queries

The demo automatically tests:

1. **"What is the remote work policy?"**
   - Tests retrieval from company policy document
2. **"What are the code review guidelines?"**
   - Tests retrieval from guidelines document
3. **"Tell me about Python programming"**
   - Tests retrieval from web content

## Customize Your Test

Edit `main.py` at the bottom:

```python
# Around line 480
queries = [
    "Your custom question here?",
    "Another question?",
]
```

## Key Files

```
demo-10-complete-rag-pipeline/
├── main.py              # Single file with complete RAG (500 lines)
├── .env                 # Your configuration (create from .env.example)
├── .env.example         # Configuration template
├── pyproject.toml       # Dependencies
└── Documents/           # Your knowledge base
    ├── company_policy.pdf
    ├── guidelines.txt
    └── policies.txt
```

## Common Commands

```bash
# Run with ChromaDB (default, local)
VECTOR_DB=chromadb uv run python main.py

# Run with Pinecone (cloud)
VECTOR_DB=pinecone uv run python main.py

# Clean ChromaDB storage
rm -rf chroma_db/

# Check code structure
wc -l main.py  # Should show ~500 lines
```

## Understanding the Output

### Step 1: Loading

```
[1.1] Loading PDF...
  ✓ Loaded 5 page(s) from PDF
[1.2] Loading text files...
  ✓ Loaded: guidelines.txt
```

### Step 4: Retrieval

```
✓ Retrieved 3 relevant chunks:
[1] Source: Documents/company_policy.pdf
    Page: 2
    Length: 892 characters
    Preview: Remote Work Policy  All employees are eligible...
```

### Step 5: Generation

```
❓ Question: What is the remote work policy?

💡 Answer:
All employees are eligible for remote work arrangements.
They must submit a remote work request form...
```

## Switching Vector Databases

Already running? Easy to switch!

```bash
# Currently using ChromaDB? Switch to Pinecone:
# 1. Edit .env
VECTOR_DB=pinecone
PINECONE_API_KEY=your-key

# 2. Run again
uv run python main.py

# That's it! Same code, different storage.
```

## Troubleshooting

### "OPENAI_API_KEY not found"

```bash
# Create .env from template
cp .env.example .env

# Edit and add your key
OPENAI_API_KEY=sk-your-actual-key-here
```

### "No documents loaded"

```bash
# Check Documents folder exists
ls Documents/

# Add your own documents
cp ~/my-docs/*.pdf Documents/
cp ~/my-docs/*.txt Documents/
```

### Want to start fresh?

```bash
# Clear vector database
rm -rf chroma_db/

# Reinstall
uv pip install -e . --force-reinstall

# Run again
uv run python main.py
```

## What Makes This "Complete" RAG?

Most RAG demos show either:

- **Ingestion only**: Load → Chunk → Store
- **Retrieval only**: Query → Find similar chunks

This demo shows **BOTH**:

- **Ingestion**: Load → Chunk → Embed → Store
- **Retrieval + Generation**: Query → Retrieve → Generate Answer

It's the full cycle from raw documents to AI-generated answers!

## Next Experiments

Try these modifications:

### 1. Different Chunk Sizes

```python
# Line ~54
CHUNK_SIZE = 500  # Try: 500, 1500, 2000
CHUNK_OVERLAP = 100  # Try: 50, 100, 300
```

### 2. More Retrieval Results

```python
# Line ~487
run_rag_pipeline(query, k=5)  # Try: 2, 5, 10
```

### 3. Different LLM

```bash
# .env
OPENAI_MODEL=gpt-4  # More powerful but expensive
OPENAI_MODEL=gpt-3.5-turbo  # Faster and cheaper
```

### 4. Add Your Documents

```bash
# Add your files
cp ~/Downloads/my-report.pdf Documents/
cp ~/notes/*.txt Documents/

# Run again - it will process your documents!
uv run python main.py
```

## Understanding RAG in 30 Seconds

```
Traditional LLM:
User: "What's our remote work policy?"
LLM: "I don't know your specific policy..."

RAG System:
User: "What's our remote work policy?"
System: [Searches company documents]
        [Finds relevant policy sections]
        [Provides to LLM as context]
LLM: "According to your policy, all employees
      are eligible for remote work..."
```

## Performance Notes

- **ChromaDB**: Instant setup, no config needed
  - ✓ Perfect for learning
  - ✓ Perfect for local development
  - ✓ Stores in `./chroma_db/` folder

- **Pinecone**: Requires API key setup
  - ✓ Better for production
  - ✓ Handles larger datasets
  - ✓ Cloud-based, always available

## File Size

```bash
# Check implementation size
wc -l main.py
# Output: ~500 lines (complete RAG in single file!)

# Compare to multi-file architectures:
# Original demo-10: 7 files, PostgreSQL required
# This demo: 1 file, no database setup needed
```

## Quick Reference

| Feature         | Configuration                      |
| --------------- | ---------------------------------- |
| Vector DB       | `VECTOR_DB=chromadb` or `pinecone` |
| LLM Model       | `OPENAI_MODEL=gpt-4o-mini`         |
| Chunk Size      | Edit `CHUNK_SIZE` in main.py       |
| Retrieval Count | Change `k=3` in queries            |
| Documents       | Add to `Documents/` folder         |

## Learning Path

1. **Run as-is** - See complete RAG in action
2. **Read output** - Understand each step
3. **Try different queries** - Test retrieval quality
4. **Adjust chunk size** - See impact on results
5. **Change k values** - More/fewer retrieved chunks
6. **Add your documents** - Use your own knowledge base
7. **Switch vector DB** - Compare ChromaDB vs Pinecone

## Full Documentation

For detailed explanations, see [README.md](README.md)

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Review console output for step-by-step progress
- Examine main.py comments for code explanations
- Ensure Documents/ folder has content
- Verify .env has valid API keys

---

**Total setup time**: ~2 minutes  
**Concepts demonstrated**: 7 (loading, chunking, embedding, storing, retrieving, generating, config-driven)  
**Lines of code**: ~500 (single file)  
**Dependencies**: Minimal (ChromaDB has zero config)

Ready? Run `uv run python main.py` and watch RAG in action! 🚀
