# Demo 09: Complete RAG Ingestion Pipeline

A **simplified, single-file demonstration** of the complete RAG (Retrieval-Augmented Generation) ingestion pipeline with config-driven vector database support.

## What You'll Learn

- **Complete RAG pipeline**: Load → Chunk → Embed → Store → Query
- **Multi-source document loading**: PDF, text files, and web pages
- **Document chunking**: Using RecursiveCharacterTextSplitter
- **Vector embeddings**: With OpenAI text-embedding-3-small
- **Config-driven vector databases**: ChromaDB or Pinecone
- **Similarity search**: Query stored documents

## Features

- **Single-file implementation** - See entire pipeline at once (376 lines)
- **Config-driven database** - Switch between ChromaDB and Pinecone via .env
- **Standard OpenAI embeddings** - No Azure account needed
- **Complete pipeline** - All 5 steps in one place
- **Clear output** - See each step as it executes
- **Error handling** - Graceful failure with informative messages
- **Pattern consistency** - Follows demos 04-08 single-file approach

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- For Pinecone: Pinecone API key (optional)
- Internet connection (for web loading)

## Installation

```bash
# Navigate to project directory
cd demo-09-rag-ingestion-pipeline

# Install dependencies
uv sync
```

## Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

### Option 1: Using ChromaDB (Local, No Setup)

```env
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_DB=chromadb
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=company_policies
```

### Option 2: Using Pinecone (Cloud-Based)

```env
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_DB=pinecone
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=company-policies
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Quick Start

```bash
# Run the complete pipeline
uv run python main.py
```

## The 5-Step Pipeline

### Step 1: Load Documents

```python
# Loads from multiple sources
documents = load_documents()
# - PDF files (one Document per page)
# - Text files (one Document per file)
# - Web pages (one Document per URL)
```

### Step 2: Chunk Documents

```python
# Splits documents into smaller chunks
chunks = chunk_documents(documents)
# - Uses RecursiveCharacterTextSplitter
# - Default: 1000 characters, 200 overlap
# - Preserves metadata from source documents
```

### Step 3: Generate Embeddings

```python
# Creates vector embeddings (combined with Step 4)
# - Uses OpenAI text-embedding-3-small (1536 dimensions)
# - Each chunk becomes a vector
```

### Step 4: Store in Vector Database

```python
# Stores chunks with embeddings
store_chunks(chunks)
# - Uses ChromaDB or Pinecone (config-driven)
# - Automatically creates index if needed
```

### Step 5: Query Similar Documents

```python
# Semantic similarity search
query_documents("What is the remote work policy?", k=3)
# - Finds most similar chunks
# - Returns top-k results with metadata
```

## Expected Output

```
======================================================================
RAG INGESTION PIPELINE CONFIGURATION
======================================================================
Vector Database: CHROMADB
Chunk Size: 1000 characters
Chunk Overlap: 200 characters
======================================================================

✓ OpenAI embeddings initialized: text-embedding-3-small
✓ ChromaDB initialized
  - Storage: ./chroma_db
  - Collection: company_policies
✓ Vector store ready!

======================================================================
STEP 1: LOADING DOCUMENTS
======================================================================

[1.1] Loading PDF...
  ✓ Loaded 5 page(s) from PDF

[1.2] Loading text files...
  ✓ Loaded: guidelines.txt
  ✓ Loaded: policy.txt

[1.3] Loading web page...
  ✓ Loaded web page
  ✓ Content length: 7,382 characters

✓ Total documents loaded: 8

======================================================================
STEP 2: CHUNKING DOCUMENTS
======================================================================

✓ Created 12 chunks
  - Average length: 850 characters
  - Min length: 98 characters
  - Max length: 1000 characters

📄 Sample Chunk:
  Source: Documents/company_policy.pdf
  Length: 987 characters
  Preview: Company Policy Manual  Document Version: 1.0...

======================================================================
STEPS 3 & 4: GENERATE EMBEDDINGS & STORE
======================================================================

🔄 Processing 12 chunks...
  - Generating embeddings with OpenAI
  - Storing in CHROMADB

✓ Successfully stored 12 chunks with embeddings!

======================================================================
STEP 5: QUERYING FOR SIMILAR DOCUMENTS
======================================================================

Query: "What is the remote work policy?"
Top 2 results:

--- Result 1 ---
Source: Documents/policy.txt
Length: 98 characters
Content: Remote work requires manager approval. Employees must
be available during core hours 10 AM - 3 PM....

--- Result 2 ---
Source: Documents/company_policy.pdf
Page: 2
Length: 983 characters
Content: • Public Holidays: As per the official holiday calendar...

======================================================================
PIPELINE COMPLETE!
======================================================================

✓ Summary:
  1. Loaded 8 documents
  2. Created 12 chunks
  3. Generated embeddings with OpenAI
  4. Stored in CHROMADB
  5. Demonstrated similarity search

✓ RAG ingestion pipeline completed successfully!
======================================================================
```

## Understanding the Code

### Pipeline Architecture

```python
# Single file, clear flow:
main.py (376 lines)
├── Configuration & Setup (70 lines)
│   ├── Load environment variables
│   ├── Initialize OpenAI embeddings
│   └── Initialize vector store (ChromaDB or Pinecone)
├── Step 1: load_documents() (50 lines)
├── Step 2: chunk_documents() (40 lines)
├── Steps 3 & 4: store_chunks() (30 lines)
├── Step 5: query_documents() (35 lines)
└── Main execution (50 lines)
```

### Key Concepts

**Document Loading:**

```python
# PyPDFLoader - One Document per page
pdf_loader = PyPDFLoader("file.pdf")
pdf_docs = pdf_loader.load()  # [doc_page1, doc_page2, ...]

# TextLoader - One Document per file
text_loader = TextLoader("file.txt")
text_docs = text_loader.load()  # [doc]

# WebBaseLoader - One Document per URL
web_loader = WebBaseLoader("https://example.com")
web_docs = web_loader.load()  # [doc]
```

**Document Chunking:**

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Target chunk size
    chunk_overlap=200,      # Overlap between chunks
    length_function=len,    # How to measure length
    separators=["\n\n", "\n", " ", ""]  # Split hierarchy
)
chunks = text_splitter.split_documents(documents)
```

**Vector Store (Config-Driven):**

```python
# ChromaDB (local)
vectorstore = Chroma(
    collection_name="company_policies",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# OR Pinecone (cloud)
vectorstore = PineconeVectorStore(
    index_name="company-policies",
    embedding=embeddings
)

# Same API for both!
vectorstore.add_documents(chunks)  # Store
results = vectorstore.similarity_search(query, k=3)  # Query
```

## Project Structure

```
demo-09-rag-ingestion-pipeline/
├── main.py                  # Complete pipeline (376 lines)
├── Documents/               # Sample documents
│   ├── company_policy.pdf   # 5-page PDF
│   ├── guidelines.txt       # Sample text
│   └── policy.txt           # Sample text
├── pyproject.toml           # Dependencies
├── .env.example             # Configuration template
├── .python-version          # Python 3.12
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── QUICKSTART.md            # Quick reference
```

## Switching Vector Databases

Simply change the `.env` file:

```bash
# Use ChromaDB (local, no setup)
VECTOR_DB=chromadb

# Use Pinecone (cloud-based)
VECTOR_DB=pinecone
```

Both use the same API in code - no code changes needed!

## Why This Matters for RAG

This demo shows the **complete RAG ingestion pipeline**:

1. **Load**: Get documents from various sources
2. **Chunk**: Break into manageable pieces (important for context windows)
3. **Embed**: Convert text to vectors (semantic meaning)
4. **Store**: Save in vector database (fast similarity search)
5. **Query**: Find relevant chunks (retrieval for RAG)

This is the "R" (Retrieval) part of RAG!

## Comparison with Original Demo

| Metric            | Original (demo-09-injection-pipeline) | New (demo-09-rag-ingestion-pipeline) |
| ----------------- | ------------------------------------- | ------------------------------------ |
| **Files**         | 14 Python files                       | 1 Python file                        |
| **Lines**         | 1,624 lines                           | 376 lines                            |
| **Complexity**    | Service architecture                  | Simple, linear                       |
| **Database**      | PostgreSQL + pgvector                 | ChromaDB or Pinecone                 |
| **Embeddings**    | Azure OpenAI                          | Standard OpenAI                      |
| **Setup**         | Database installation                 | No external setup (ChromaDB)         |
| **Learning Time** | 60+ minutes                           | 20-30 minutes                        |
| **Pattern**       | Multi-file                            | Single-file (consistent)             |

**Result**: **77% less code**, no external database required, standard OpenAI API.

## Common Issues

| Issue                  | Solution                                      |
| ---------------------- | --------------------------------------------- |
| OpenAI API key missing | Add `OPENAI_API_KEY` to `.env` file           |
| Pinecone errors        | Check API key, use ChromaDB instead           |
| Documents not found    | Ensure `Documents/` folder exists with files  |
| Import errors          | Run `uv sync` to install dependencies         |
| Web loading fails      | Check internet connection; pipeline continues |

## Modifying the Demo

### Change Chunking Parameters

```python
# In main.py, modify these constants:
CHUNK_SIZE = 500        # Smaller chunks
CHUNK_OVERLAP = 100     # Less overlap
```

### Add More Documents

```bash
# Add files to Documents/
cp your_file.pdf Documents/
cp your_file.txt Documents/
```

### Try Different Queries

```python
# In main.py, modify the queries list:
queries = [
    "Your custom query here",
    "Another query",
]
```

### Use Only ChromaDB or Pinecone

```python
# Remove the unused database code from main.py
# Keep only the if block you need
```

## Next Steps

After mastering this demo:

1. Experiment with different chunk sizes
2. Try your own documents
3. Add more queries
4. Compare ChromaDB vs Pinecone performance
5. Build a simple RAG application using this pipeline

## Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)

---

**Compare with**: See `demo-09-injection-pipeline/` for a production-ready, service-based architecture (14 files, 1,624 lines).
