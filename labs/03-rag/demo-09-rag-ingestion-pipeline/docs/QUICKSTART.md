# Quick Start Guide

## Installation & Setup

```bash
# 1. Navigate to project
cd demo-09-rag-ingestion-pipeline

# 2. Install dependencies
uv sync

# 3. Setup environment
cp .env.example .env
```

## Configure .env

### For ChromaDB (Recommended for Learning)

```env
OPENAI_API_KEY=sk-your_key_here
VECTOR_DB=chromadb
```

### For Pinecone (Cloud Option)

```env
OPENAI_API_KEY=sk-your_key_here
VECTOR_DB=pinecone
PINECONE_API_KEY=your_pinecone_key
```

## Run the Pipeline

```bash
uv run python main.py
```

## The 5-Step Pipeline

### 1. Load Documents

```
✓ PDF: 5 pages
✓ Text files: 2 files
✓ Web: 1 page
Total: 8 documents
```

### 2. Chunk Documents

```
✓ Created 12 chunks
✓ Average: 850 characters
✓ Range: 98-1000 characters
```

### 3. Generate Embeddings

```
✓ OpenAI text-embedding-3-small
✓ 1536 dimensions per chunk
```

### 4. Store in Vector DB

```
✓ ChromaDB or Pinecone
✓ 12 chunks stored
```

### 5. Query Documents

```
Query: "What is the remote work policy?"
✓ Returns top 2 similar chunks
```

## Pipeline Flow

```
Documents/
├── PDFs     ──┐
├── Text     ──┤
└── Web      ──┘
                │
                ▼
         Load Documents
                │
                ▼
      Chunk (1000 chars, 200 overlap)
                │
                ▼
      Generate Embeddings (OpenAI)
                │
                ▼
    Store in Vector DB (ChromaDB/Pinecone)
                │
                ▼
        Similarity Search
```

## Code Overview

```python
# main.py - 376 lines, all pipeline steps

# Configuration (lines 1-120)
load_dotenv()
embeddings = OpenAIEmbeddings()
vectorstore = Chroma() or PineconeVectorStore()

# Step 1: Load (lines 121-180)
documents = load_documents()

# Step 2: Chunk (lines 181-230)
chunks = chunk_documents(documents)

# Steps 3 & 4: Embed & Store (lines 231-260)
store_chunks(chunks)

# Step 5: Query (lines 261-300)
query_documents("search query", k=3)
```

## Key Concepts

### Document Object

```python
Document(
    page_content="Text content here...",
    metadata={"source": "file.pdf", "page": 0}
)
```

### Chunking

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Target size
    chunk_overlap=200,    # Overlap
    separators=["\n\n", "\n", " ", ""]
)
```

### Vector Store (Same API!)

```python
# Add documents (embeds automatically)
vectorstore.add_documents(chunks)

# Search similar
results = vectorstore.similarity_search(query, k=3)
```

## Configuration Options

### Chunk Size

```python
# In main.py
CHUNK_SIZE = 1000       # Larger = more context
CHUNK_OVERLAP = 200     # Higher = more redundancy
```

### Vector Database

```bash
# In .env
VECTOR_DB=chromadb      # Local, no setup
# OR
VECTOR_DB=pinecone      # Cloud, requires API key
```

### Embedding Model

```python
# In main.py
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # or text-embedding-3-large
)
```

## Quick Modifications

### Add Your Documents

```bash
# Copy files to Documents/
cp my_file.pdf Documents/
cp my_file.txt Documents/
```

### Change Query

```python
# In main.py, modify queries list:
queries = [
    "Your question here",
]
```

### Switch Database

```bash
# Just change .env
VECTOR_DB=pinecone  # or chromadb
```

## Typical Output

```
======================================================================
STEP 1: LOADING DOCUMENTS
✓ Total documents loaded: 8

STEP 2: CHUNKING DOCUMENTS
✓ Created 12 chunks

STEPS 3 & 4: GENERATE EMBEDDINGS & STORE
✓ Successfully stored 12 chunks with embeddings!

STEP 5: QUERYING FOR SIMILAR DOCUMENTS
Query: "What is the remote work policy?"
--- Result 1 ---
Source: Documents/policy.txt
Content: Remote work requires manager approval...

PIPELINE COMPLETE!
======================================================================
```

## Troubleshooting

| Error                      | Fix                              |
| -------------------------- | -------------------------------- |
| `OPENAI_API_KEY not found` | Add key to `.env` file           |
| `Module not found`         | Run `uv sync`                    |
| `VECTOR_DB unsupported`    | Use `chromadb` or `pinecone`     |
| `Documents not loaded`     | Check `Documents/` folder exists |
| Pinecone error             | Switch to `chromadb` in `.env`   |

## Understanding the Output

### Step 1: Shows Documents Loaded

- PDF: One document per page
- Text: One document per file
- Web: One document per URL

### Step 2: Shows Chunking Stats

- Total chunks created
- Average, min, max lengths
- Sample chunk preview

### Steps 3 & 4: Shows Storage

- Embedding generation
- Vector database storage
- Success confirmation

### Step 5: Shows Search Results

- Query used
- Top k results
- Source, page, content preview

## Vector Databases Compared

| Feature      | ChromaDB     | Pinecone          |
| ------------ | ------------ | ----------------- |
| **Setup**    | None         | API key needed    |
| **Storage**  | Local files  | Cloud             |
| **Speed**    | Fast (local) | Very fast (cloud) |
| **Scaling**  | Limited      | Unlimited         |
| **Cost**     | Free         | Free tier + paid  |
| **Learning** | ✓ Best       | Alternative       |

## Next Steps

1. Run: `uv run python main.py`
2. Try different documents
3. Experiment with chunk sizes
4. Test various queries
5. Compare ChromaDB vs Pinecone

## Complete RAG Flow

```
This Demo: Ingestion (R in RAG)
├── Load documents
├── Chunk content
├── Generate embeddings
├── Store vectors
└── Search similar → Next: Generation (AG in RAG)
```

## Key Takeaways

✓ **5 steps**: Load → Chunk → Embed → Store → Query  
✓ **Single file**: 376 lines, all in one place  
✓ **Config-driven**: Switch databases via .env  
✓ **Standard OpenAI**: No Azure account needed  
✓ **No external DB**: ChromaDB works out of the box  
✓ **Complete pipeline**: Production-ready pattern

---

See `README.md` for detailed documentation.
