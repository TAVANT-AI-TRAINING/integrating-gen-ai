# Complete RAG Ingestion Pipeline

## Objective

Build a complete RAG ingestion pipeline that loads documents from multiple sources, chunks them, and stores them in PgVector with embeddings.

## Prerequisites

- PostgreSQL with pgvector extension installed
- Azure OpenAI API credentials
- Understanding of RecursiveCharacterTextSplitter basics

## Project Structure

```
demo-2-injection-pipeline/
├── main.py                                  # FastAPI application entry point
├── pyproject.toml                           # UV project configuration
├── uv.lock                                  # UV lock file for dependency versions
├── README.md                                # Project documentation
├── .gitignore                               # Git ignore rules
├── main/                                    # Main application package
│   ├── __init__.py                          # Package initialization
│   ├── routes/                              # FastAPI routes
│   │   ├── __init__.py                      # Routes package initialization
│   │   └── ingestion_routes.py              # FastAPI routes for ingestion & query
│   └── services/                            # Service layer modules
│       ├── __init__.py                      # Services package initialization
│       ├── pdf_loader_service.py            # PDF document loading service
│       ├── text_loader_service.py           # Text file loading service
│       ├── html_loader_service.py           # HTML file loading service
│       ├── web_loader_service.py            # Web page loading service
│       ├── document_processing_service.py   # Multi-source processing & file type detection
│       ├── chunking_service.py              # Document chunking and metadata service
│       ├── injection_pipeline_service.py    # Complete pipeline orchestration
│       ├── embedding_service.py             # Embedding generation & vector storage
│       └── query_service.py                 # Vector similarity search service
├── Documents/                               # Sample documents directory
    ├── company_policy.pdf                   # Company policy document
    ├── policy.txt                           # Remote work policy
    └── guidelines.txt                       # Code review guidelines

```

## Setup Instructions

### 1. Verify Documents Directory

Ensure the `Documents/` directory contains the sample files:

```bash
# Check if files exist
ls -la Documents/

# Should contain:
# - sample.pdf (2-page PDF)
# - policy.txt
# - guidelines.txt
```

If files don't exist, add your own documents to the `Documents/` directory. Supported formats: PDF, TXT, HTML.

### 2. Initialize the UV Environment

From the project root directory:

```bash
uv sync
```

This installs all required packages:

- `langchain` - Core framework
- `langchain-community` - Document loaders
- `langchain-text-splitters` - Text splitting utilities
- `pypdf` - PDF processing
- `beautifulsoup4` - HTML parsing

### 3. Configure Environment Variables

Create a `.env` file in the project root with your database and Azure OpenAI credentials:

```bash
# Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=your_user
DB_PASSWORD=your_password
COLLECTION_NAME=company_policies

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-01
```

### 4. Start the FastAPI Server

**Simple approach (recommended):**

```bash
uv run python main.py
```

Or:

```bash
python main.py
```

**Alternative approach (using uvicorn directly):**

```bash
uv run uvicorn main.routes.ingestion_routes:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**Note**: You can customize the host, port, and reload settings using environment variables:

- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `RELOAD` (default: `true`)

### 5. Use the API Endpoints

#### Upload and Process Documents

```bash
# Upload a PDF file
curl -X POST "http://localhost:8000/api/v1/upload?chunk_size=1000&chunk_overlap=100" \
  -F "file=@Documents/company_policy.pdf"

# Upload a text file
curl -X POST "http://localhost:8000/api/v1/upload?chunk_size=1000&chunk_overlap=100" \
  -F "file=@Documents/policy.txt"

# Upload an HTML file
curl -X POST "http://localhost:8000/api/v1/upload?chunk_size=1000&chunk_overlap=100" \
  -F "file=@Documents/guidelines.html"
```

#### Query Documents

```bash
# Query the vector database
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the company policy on remote work?"}'
```

## What the API Does

### Upload Endpoint (`/api/v1/upload`)

The upload endpoint processes documents through the complete ingestion pipeline:

1. **Load**: Uses LangChain loaders (PyPDFLoader, TextLoader, BSHTMLLoader) based on file type
2. **Chunk**: Splits documents using RecursiveCharacterTextSplitter
3. **Inject Metadata**: Adds unique chunk IDs to each chunk
4. **Embed**: Generates embeddings using Azure OpenAI
5. **Store**: Saves chunks with embeddings in PostgreSQL vector database

**Supported file types**: PDF, TXT, MD, HTML

**Query parameters**:

- `chunk_size`: Size of chunks in characters (default: 1000, range: 1-8000)
- `chunk_overlap`: Overlap between chunks (default: 100, range: 0-4000)

**Response**: JSON with processing statistics including:

- File type and name
- Total chunks created
- Chunk length statistics
- Processing time
- Collection name
- Chunk IDs

### Query Endpoint (`/api/v1/query`)

The query endpoint performs semantic similarity search:

1. **Generate Query Embedding**: Creates embedding for the query using Azure OpenAI
2. **Similarity Search**: Searches for most similar document chunks using cosine similarity
3. **Return Results**: Returns top-k (k=1) most relevant chunk with metadata

**Request body**: `{"query": "your search query"}`

**Response**: JSON with the most relevant document chunk including content and metadata

## Expected API Response

### Upload Response

```json
{
  "status": "success",
  "filename": "company_policy.pdf",
  "file_type": "pdf",
  "total_chunks": 5,
  "average_chunk_length": 856.2,
  "min_chunk_length": 420,
  "max_chunk_length": 1000,
  "processing_time_seconds": 2.34,
  "collection_name": "company_policies",
  "chunk_ids": ["chunk_abc123", "chunk_def456", "chunk_ghi789", "chunk_jkl012", "chunk_mno345"],
  "source_documents": 5
}
```

### Query Response

```json
{
  "query": "What is the company policy on remote work?",
  "results": [
    {
      "content": "Remote work requires manager approval. Employees must be available during core business hours...",
      "metadata": {
        "source": "/path/to/policy.txt",
        "chunk_id": "chunk_abc123"
      },
      "similarity_score": 0.85
    }
  ],
  "collection": "company_policies"
}
```

## Key Learning Points

### 1. Multi-Source Document Processing

The practice demonstrates loading and chunking from:

- **PDFs**: Page-level documents that become multiple chunks
- **Text files**: File-level documents (may or may not need chunking)
- **Web pages**: Large documents that definitely need chunking

### 2. Chunk Size Selection

Different content types need different chunk sizes:

| Content Type     | Recommended chunk_size | Reasoning                       |
| ---------------- | ---------------------- | ------------------------------- |
| Web content      | 1000-1500              | Dense, continuous text          |
| PDF pages        | 500-800                | Structured, sectioned content   |
| Short text files | 300-500                | Already small, minimal chunking |

### 3. Metadata Preservation

Critical observation: **Metadata flows through the entire pipeline**

```python
# Original document
Document(
    page_content="Page content...",
    metadata={"source": "file.pdf", "page": 0}
)

# After splitting
Document(page_content="Chunk 1...", metadata={"source": "file.pdf", "page": 0})
Document(page_content="Chunk 2...", metadata={"source": "file.pdf", "page": 0})
```

This enables:

- Source tracking for citations
- Page-level filtering
- Content provenance

### 4. Chunk Overlap Importance

The 100-character overlap ensures:

- Context preservation at boundaries
- No information loss between chunks
- Better semantic coherence

## Experimentation Ideas

### 1. Try Different Chunk Sizes

Test different chunk sizes via API:

```bash
# Small chunks
curl -X POST "http://localhost:8000/api/v1/upload?chunk_size=500&chunk_overlap=50" \
  -F "file=@Documents/company_policy.pdf"

# Medium chunks (default)
curl -X POST "http://localhost:8000/api/v1/upload?chunk_size=1000&chunk_overlap=100" \
  -F "file=@Documents/company_policy.pdf"

# Large chunks
curl -X POST "http://localhost:8000/api/v1/upload?chunk_size=1500&chunk_overlap=150" \
  -F "file=@Documents/company_policy.pdf"
```

**Observe**: How chunk count changes in the response

### 2. Test Different File Types

Upload different file formats:

```bash
# PDF file
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@Documents/company_policy.pdf"

# Text file
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@Documents/policy.txt"

# HTML file
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@Documents/guidelines.html"
```

**Observe**: How different file types are processed

### 3. Query with Different Questions

Test semantic search with various queries:

```bash
# Specific question
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the remote work requirements?"}'

# General question
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about company policies"}'
```

**Observe**: How query phrasing affects results

## Common Questions

### Q: Why use chunk_size=1000 for web content but 500 for PDFs?

**A**: Web content is typically dense, continuous text that benefits from larger chunks. PDFs are often structured with sections and headings, so smaller chunks align better with natural boundaries.

### Q: What if my text files are very short (< 100 characters)?

**A**: They won't be split! The splitter only creates chunks if the content exceeds chunk_size. Short files remain as single chunks.

### Q: How do I know if my chunk_overlap is appropriate?

**A**: A good rule of thumb is 10-15% of chunk_size. Test by checking if related information at chunk boundaries is preserved.

### Q: Can I use different chunk sizes for different source types?

**A**: Yes! In production, you might split each source type separately with appropriate parameters, then combine the chunks.

## Troubleshooting

### Issue: Documents Not Found

```
✗ PDF file not found: /path/to/sample.pdf
```

**Solution**: Ensure documents exist in the `Documents/` directory:

```bash
# Check Documents directory
ls -la Documents/

# Add your own documents if needed
# Supported formats: PDF, TXT, MD, HTML
```

### Issue: Too Many/Few Chunks

**Solution**: Adjust chunk_size parameter:

- Too many chunks → Increase chunk_size
- Too few chunks → Decrease chunk_size

### Issue: API Server Won't Start

```
ModuleNotFoundError: No module named 'langchain'
```

**Solution**:

```bash
uv sync
```

### Issue: Upload Fails with 400 Error

**Solution**: Check that:

- File exists and is readable
- File type is supported (PDF, TXT, MD, HTML)
- File is not corrupted

### Issue: Query Returns No Results

**Solution**: Ensure:

- Documents have been uploaded successfully
- Database connection is working
- Azure OpenAI credentials are valid

## Next Steps

After completing this practice:

1. **Understand the complete RAG ingestion pipeline**: Load → Chunk → Embed → Store
2. **Test with different document types**: PDF, text, HTML files
3. **Optimize chunk parameters**: Fine-tune chunk_size and chunk_overlap for your use case
4. **Query your documents**: Use semantic search to find relevant information
5. **Verify storage**: Check PostgreSQL to confirm embeddings are stored correctly

## Related Files

- `main/routes/ingestion_routes.py` - FastAPI routes for ingestion and query
- `main/services/injection_pipeline_service.py` - Complete pipeline orchestration
- `main/services/embedding_service.py` - Embedding generation and vector storage
- `main/services/query_service.py` - Vector similarity search
- `pyproject.toml` - UV dependencies
- `Documents/` - Sample documents directory

## Code Pattern

```python
# Complete LO1 + LO2 workflow
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load documents (LO1)
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

txt_loader = TextLoader("document.txt")
txt_docs = txt_loader.load()

web_loader = WebBaseLoader(web_paths=["https://example.com"])
web_docs = web_loader.load()

all_docs = pdf_docs + txt_docs + web_docs

# 2. Split documents (LO2)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

# 3. Metadata is preserved
for chunk in chunks:
    print(f"Source: {chunk.metadata['source']}")
    if 'page' in chunk.metadata:
        print(f"Page: {chunk.metadata['page']}")
```

Happy chunking! 🔪📄

## Verify embeddings in PostgreSQL (step-by-step)

Use these steps to confirm that chunks and embeddings were stored correctly in your Postgres database via pgvector.

### 1) Ensure your database is reachable

Update your `.env` with valid credentials:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=your_user
DB_PASSWORD=your_password
COLLECTION_NAME=completed_policies   # or the name you chose
```

If you see “password authentication failed”, verify the user/password or create a role with permissions:

```bash
# Connect as an admin/superuser (e.g., postgres)
psql -h localhost -U postgres -d postgres -W

-- Inside psql, create role and grant access
CREATE USER your_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO your_user;
\c rag_db
GRANT USAGE ON SCHEMA public TO your_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
```

### 2) Install/verify pgvector extension

```sql
-- In the target database (rag_db)
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname FROM pg_extension WHERE extname = 'vector';
```

### 3) Run the ingestion (via API)

Start the API and upload a document (PDF/TXT/MD):

```bash
# Start the API server
uv run python main.py

# In another terminal, upload a file
curl -X POST "http://localhost:8000/api/v1/upload?chunk_size=1000&chunk_overlap=100" \
  -F "file=@Documents/company_policy.pdf"
```

The response includes summary fields and `chunk_ids`. These IDs are saved in `cmetadata->>'id'` for each chunk row.

### 4) Connect to Postgres and inspect tables

```bash
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
```

List the LangChain tables (created by langchain-postgres):

```sql
\dt langchain_pg_*
```

You should see (names may vary by version):

- `langchain_pg_collection`
- `langchain_pg_embedding`

### 5) Verify the collection

```sql
SELECT uuid, name, cmetadata
FROM public.langchain_pg_collection
ORDER BY name ASC
LIMIT 5;
```

Confirm your collection exists and matches `COLLECTION_NAME`.

If you want to see the most recently populated collections (by last embedding row id):

```sql
SELECT
  c.uuid,
  c.name,
  c.cmetadata,
  MAX(e.id) AS last_embedding_row_id
FROM public.langchain_pg_collection c
LEFT JOIN public.langchain_pg_embedding e
  ON e.collection_id = c.uuid
GROUP BY c.uuid, c.name, c.cmetadata
ORDER BY last_embedding_row_id DESC NULLS LAST
LIMIT 5;
```

### 6) Count stored embeddings for the collection

```sql
-- Replace :collection with your collection name (or use the env value)
WITH coll AS (
  SELECT uuid FROM langchain_pg_collection WHERE name = :collection
)
SELECT COUNT(*) AS chunk_count
FROM langchain_pg_embedding e
JOIN coll ON e.collection_id = coll.uuid;
```

Quick overview by collection:

```sql
SELECT c.name, COUNT(e.*) AS chunk_count
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
GROUP BY c.name
ORDER BY chunk_count DESC;
```

### 7) Peek at stored documents and metadata

```sql
SELECT e.id,
       LEFT(e.document, 120) AS preview,
       e.cmetadata->>'id' AS chunk_id,
       e.cmetadata->>'source' AS source
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = :collection
LIMIT 10;
```

To see chunk IDs only (should match the API upload response):

```sql
SELECT e.cmetadata->>'id' AS chunk_id
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = :collection
LIMIT 20;
```

### 8) Verify the embedding column

Check type and (optionally) dimensionality:

```sql
-- Column type should be 'vector'
SELECT pg_typeof(embedding) AS embedding_type
FROM langchain_pg_embedding
LIMIT 1;

-- If your pgvector version supports it, this returns dimensions (e.g., 1536)
-- Otherwise, skip this check.
SELECT vector_dims(embedding) AS dims
FROM langchain_pg_embedding
LIMIT 1;
```

Self-similarity sanity check (distance to itself should be 0):

```sql
SELECT (embedding <-> embedding) AS self_distance
FROM langchain_pg_embedding
LIMIT 1;
```

### 9) (Optional) Create a vector index for faster search

```sql
-- Requires pgvector; choose lists based on data size
CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_ivfflat
ON langchain_pg_embedding
USING ivfflat (embedding)
WITH (lists = 100);

ANALYZE langchain_pg_embedding;
```

### 10) Common issues and fixes

- ❌ Password authentication failed
  - Fix `.env` credentials or create/assign a role (see step 1)
- ❌ pgvector extension not found
  - Install extension and run `CREATE EXTENSION vector;` (step 2)
- ❌ No rows stored
  - Ensure the API upload succeeded (HTTP 200) and Azure OpenAI credentials are valid
  - Check logs for `Failed to store documents` and fix DB connectivity
