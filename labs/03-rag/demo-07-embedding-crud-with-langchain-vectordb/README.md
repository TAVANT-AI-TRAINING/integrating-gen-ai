# Demo 07: LangChain with Multiple Vector Databases

The ultimate demonstration of **LangChain's power**: Same code works with ChromaDB OR Pinecone - just change configuration!

## 🎯 What You'll Learn

- LangChain's unified vector store interface
- Switch between ChromaDB and Pinecone without code changes
- Production-ready multi-database architecture
- Configuration-driven database selection
- The power of abstraction layers

## 🌟 Why This Demo?

**The Problem:**

- Demo 04: ChromaDB code
- Demo 06: Pinecone code (different API)
- To switch? Rewrite everything! 😱

**The LangChain Solution:**

```python
# Same code for both!
vectorstore.add_documents([doc], ids=[id])
vectorstore.similarity_search(query)
vectorstore.delete(ids=[id])

# Just change VECTOR_DB in .env:
VECTOR_DB=chromadb  # or pinecone
```

**That's the power of LangChain!** 🦜🔗

## 📊 Architecture

```
Your Code (one version)
        ↓
LangChain Abstraction Layer
        ↓
    ┌───┴───┐
    ↓       ↓
ChromaDB  Pinecone
```

Same API, different backends!

## 🚀 Setup

### 1. Install Dependencies

```bash
uv sync
```

This installs support for BOTH ChromaDB and Pinecone!

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env`:

**For ChromaDB (Local Development):**

```env
OPENAI_API_EMBEDDING_KEY=sk-your-key-here
VECTOR_DB=chromadb
CHROMA_DB_DIR=./chroma_db
```

**For Pinecone (Cloud Production):**

```env
OPENAI_API_EMBEDDING_KEY=sk-your-key-here
VECTOR_DB=pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=company-policies
```

### 3. Run the Server

```bash
uv run python main.py
```

Server will use the database specified in `VECTOR_DB`!

## 🔄 Switching Databases

### Start with ChromaDB

```bash
# .env
VECTOR_DB=chromadb
```

```bash
uv run python main.py
# ✓ Selected Vector Database: CHROMADB
# ✓ ChromaDB initialized
```

### Switch to Pinecone

```bash
# .env
VECTOR_DB=pinecone
PINECONE_API_KEY=your-key
```

```bash
uv run python main.py
# ✓ Selected Vector Database: PINECONE
# ✓ Pinecone initialized
```

**No code changes needed!** 🎉

## 📚 API Endpoints

All endpoints work identically with both databases!

### Health Check

```bash
curl http://localhost:8000/
# Shows which database is active
```

### CREATE - Add a Document

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-001",
    "text": "All employees must complete annual security training.",
    "metadata": {"category": "security", "date": "2024-01-15"}
  }'
```

### READ - Get a Document

```bash
curl http://localhost:8000/documents/policy-001
```

### UPDATE - Update a Document

```bash
curl -X PUT http://localhost:8000/documents/policy-001 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Updated policy text",
    "metadata": {"category": "security", "updated": "2024-02-01"}
  }'
```

### DELETE - Remove a Document

```bash
curl -X DELETE http://localhost:8000/documents/policy-001
```

### QUERY - Semantic Search

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What are the security requirements?",
    "n_results": 3
  }'
```

### LIST - Get All Documents

```bash
curl http://localhost:8000/documents
```

## 🧪 Try Both Databases

### Test ChromaDB (Local)

1. Set `VECTOR_DB=chromadb` in `.env`
2. Start server: `uv run python main.py`
3. Add documents (fast, < 10ms latency)
4. Data stored in `./chroma_db/`

### Test Pinecone (Cloud)

1. Get Pinecone API key from [pinecone.io](https://www.pinecone.io/)
2. Set `VECTOR_DB=pinecone` in `.env`
3. Add your `PINECONE_API_KEY`
4. Start server: `uv run python main.py`
5. Add same documents (slower, ~100ms latency)
6. Data stored in Pinecone cloud

### Compare Performance

```bash
# ChromaDB - Local speed
time curl -X POST http://localhost:8000/documents -d '...'
# ~0.05s

# Pinecone - Network overhead
time curl -X POST http://localhost:8000/documents -d '...'
# ~0.15s
```

## 🔍 Interactive API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 Project Structure

```
demo-07-embedding-crud-with-langchain-vectordb/
├── main.py                      # Multi-database support!
├── pyproject.toml               # Both ChromaDB and Pinecone
├── .env.example                 # Configuration guide
├── .env                         # Your settings (git-ignored)
├── README.md                    # This file
├── QUICKSTART.md                # Quick setup
├── ABSTRACTION_POWER.md         # Why LangChain is amazing
└── chroma_db/                   # Created if using ChromaDB
```

## 💡 The Power of LangChain

### Without LangChain (Demo 04 vs Demo 06)

**ChromaDB Code:**

```python
import chromadb

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("docs")

collection.add(
    ids=["doc-1"],
    embeddings=embedding,
    documents=[text]
)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
```

**Pinecone Code (Completely Different!):**

```python
from pinecone import Pinecone

pc = Pinecone(api_key=key)
index = pc.Index("docs")

index.upsert(
    vectors=[{
        "id": "doc-1",
        "values": embedding,
        "metadata": {"text": text}
    }]
)

results = index.query(
    vector=query_embedding,
    top_k=5
)
```

**To switch: Rewrite everything!**

### With LangChain (Demo 07)

**Same Code for Both:**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # or langchain_pinecone

embeddings = OpenAIEmbeddings()

# Initialize based on config
if VECTOR_DB == "chromadb":
    vectorstore = Chroma(embedding_function=embeddings, ...)
else:
    vectorstore = PineconeVectorStore(embedding=embeddings, ...)

# Everything else is identical!
vectorstore.add_documents([doc], ids=[id])
results = vectorstore.similarity_search(query)
```

**To switch: Change one line in .env!**

## 🎓 Learning Progression

1. **Demo 04** → Direct ChromaDB (understand fundamentals)
2. **Demo 05** → LangChain + ChromaDB (abstraction)
3. **Demo 06** → Direct Pinecone (cloud databases)
4. **Demo 07** → LangChain + Both (ultimate flexibility)

## 🔧 Code Highlights

### Unified Interface

**Add Documents (works for both):**

```python
vectorstore.add_documents(
    documents=[document],
    ids=[doc_id]
)
```

**Search (works for both):**

```python
results = vectorstore.similarity_search_with_score(
    query=query_text,
    k=n_results
)
```

**Delete (works for both):**

```python
vectorstore.delete(ids=[doc_id])
```

### Configuration-Driven Selection

```python
if VECTOR_DB == "chromadb":
    from langchain_chroma import Chroma
    vectorstore = Chroma(...)

elif VECTOR_DB == "pinecone":
    from langchain_pinecone import PineconeVectorStore
    vectorstore = PineconeVectorStore(...)
```

## 📊 Comparison Table

| Feature       | ChromaDB            | Pinecone       | LangChain Code                  |
| ------------- | ------------------- | -------------- | ------------------------------- |
| **Add**       | collection.add()    | index.upsert() | vectorstore.add_documents()     |
| **Search**    | collection.query()  | index.query()  | vectorstore.similarity_search() |
| **Delete**    | collection.delete() | index.delete() | vectorstore.delete()            |
| **Switch DB** | Rewrite code        | Rewrite code   | **Change .env only!**           |

## 🚀 Production Use Cases

### Development Workflow

```
1. Develop with ChromaDB (fast, local)
   VECTOR_DB=chromadb

2. Test with Pinecone (production environment)
   VECTOR_DB=pinecone

3. Deploy with Pinecone (scale automatically)
   VECTOR_DB=pinecone
```

### Multi-Environment

```
Development: VECTOR_DB=chromadb
Staging:     VECTOR_DB=pinecone
Production:  VECTOR_DB=pinecone
```

### Migration Path

```
1. Start with ChromaDB
2. Grow too large
3. Switch to Pinecone
4. Change one config line
5. Done! ✅
```

## 💰 Cost Optimization

**Development:**

- Use ChromaDB (free)
- Fast iteration
- No cloud costs

**Production:**

- Switch to Pinecone (paid)
- Auto-scaling
- High availability

**Best of both worlds!**

## 🎯 Best Practices

### 1. Use Environment Variables

```python
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb")
```

### 2. Validate Configuration

```python
if VECTOR_DB not in ["chromadb", "pinecone"]:
    raise ValueError(f"Unsupported VECTOR_DB: {VECTOR_DB}")
```

### 3. Conditional Imports

```python
if VECTOR_DB == "chromadb":
    from langchain_chroma import Chroma
elif VECTOR_DB == "pinecone":
    from langchain_pinecone import PineconeVectorStore
```

### 4. Document Switching Process

```markdown
# In README.md

To switch databases:

1. Change VECTOR_DB in .env
2. Restart server
3. That's it!
```

## 📚 Next Steps

After mastering this demo:

1. Add more vector databases (Weaviate, Qdrant)
2. Implement database-specific optimizations
3. Add fallback mechanisms
4. Build migration tools
5. Create multi-database testing suite

## 🤝 Why This Matters

**For Students:**

- Learn abstraction principles
- Understand dependency inversion
- See real-world flexibility

**For Production:**

- Start small (ChromaDB)
- Scale easily (Pinecone)
- No code rewrite needed

**For Teams:**

- Developer choice flexibility
- Easy environment parity
- Reduced vendor lock-in

## 🎉 The LangChain Advantage

One codebase, multiple backends!

```
Same Application Code
        ↓
    LangChain
    /      \
ChromaDB  Pinecone
```

**That's what makes LangChain so powerful!** 🦜🔗
