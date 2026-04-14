# RAG Retrieval Pipeline

## Overview

A production-ready retrieval system for querying vector embeddings stored in PgVector. Provides configurable retrievers with support for different k values, metadata filtering, and search types.

## Prerequisites

- PostgreSQL with pgvector extension installed
- Azure OpenAI API credentials
- Vector embeddings already stored in PgVector (from ingestion pipeline)
- Understanding of vector similarity search basics

## Project Structure

```
demo-1-rag-retrieval/
├── main.py                                  # Application entry point
├── pyproject.toml                           # Project configuration
├── uv.lock                                  # Dependency lock file
├── README.md                                # Documentation
├── main/                                    # Main application package
    ├── __init__.py                          # Package initialization
    ├── runner.py                            # Test runner script
    ├── retriever.py                         # Retriever creation utilities
    ├── lcel.py                              # LCEL chain creation
    ├── vector_store.py                      # Vector store connection and configuration
    └── utils.py                             # Utility functions for formatting and testing

```

## Setup Instructions

### 1. Verify Database Connection

Ensure your PostgreSQL database with pgvector is accessible and contains vector embeddings:

```bash
# Check database connection
psql -h localhost -U your_user -d your_database

# Verify pgvector extension
SELECT extname FROM pg_extension WHERE extname = 'vector';

# Check if embeddings exist
SELECT COUNT(*) FROM langchain_pg_embedding;
```

### 2. Initialize the UV Environment

From the project root directory:

```bash
uv sync
```

This installs all required packages:

- `langchain-community` - Document loaders and vector stores
- `langchain-core` - Core LangChain framework
- `langchain-openai` - Azure OpenAI embeddings
- `langchain-postgres` - PgVector integration
- `psycopg` - PostgreSQL adapter
- `python-dotenv` - Environment variable management

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DB_USER=rag_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
COLLECTION_NAME=company_policies

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-02-01
```

**Important**: The `COLLECTION_NAME` must match the collection used during document ingestion.

### 4. Run the Application

```bash
uv run python main.py
```

Or directly:

```bash
uv run python main/runner.py
```

## Application Workflow

### Initialization

- Loads environment variables
- Connects to PostgreSQL database
- Initializes Azure OpenAI embeddings
- Creates PgVector connection
- Verifies collection exists

### Test Scenarios

The application runs 5 test scenarios:

#### Scenario 1: Basic Retrieval

- Standard retrieval with `k=3`
- Tests default similarity search behavior

#### Scenario 2: K-Value Comparison

- Compares `k=2` vs `k=6` for the same query
- Demonstrates tradeoff between context and noise

#### Scenario 3: Metadata Filtering

- Searches without filter (all documents)
- Searches with source filter (specific document)
- Tests scoped search capabilities

#### Scenario 4: Search Types

- **Similarity Search**: Returns most similar documents
- **MMR (Maximum Marginal Relevance)**: Balances relevance with diversity
- Compares results from both approaches

#### Scenario 5: LCEL Retrieval Chains

- Tests LangChain Expression Language (LCEL) chains
- Demonstrates retriever chaining with formatting
- Tests different k values in LCEL chains

## Expected Output

```
================================================================================
Build and Test a Configurable Retriever
================================================================================

================================================================================
[Scenario 1] Basic Retrieval
********************************************************************************
[Basic Retrieval] k=3
================================================================================
Query: 'What is the vacation policy?'
================================================================================

Retrieved 3 documents:

[Document 1]
  Source: hr_handbook.pdf
  Page: 12
  Content Preview: Employees receive 20 days of paid vacation annually...
--------------------------------------------------------------------------------

================================================================================
[Scenario 2] Comparing Different K Values
********************************************************************************
[Comparing Different K Values]

--- With k=2 ---
[Retrieval results...]

--- With k=6 ---
[Retrieval results...]

================================================================================
[Scenario 3] Metadata Filtering
********************************************************************************
[Metadata Filtering]

--- Without Filter (searches all documents) ---
[Retrieval results...]

--- With Source Filter (searches specific document) ---
[Retrieval results...]

================================================================================
[Scenario 4] Different Search Types
********************************************************************************
[Different Search Types]

--- Similarity Search (default) ---
[Retrieval results...]

--- MMR Search (Maximum Marginal Relevance - diverse results) ---
[Retrieval results...]

================================================================================
[Scenario 5] LCEL Retrieval Chains
********************************************************************************
[LCEL Retrieval Chains]

--- LCEL with k=3 ---
[LCEL chain results...]

--- LCEL with k=5 ---
[LCEL chain results...]

--- LCEL with k=4 and metadata filter ---
[LCEL chain results...]

================================================================================
Complete - Key Takeaways
********************************************************************************
✓ Retrievers standardize the search interface
✓ K value controls context vs noise tradeoff
✓ Metadata filtering enables scoped searches
✓ Always use the same embedding model as ingestion
********************************************************************************
```

## Configuration Guide

### 1. K-Value Selection

The `k` parameter controls how many documents are retrieved:

| K Value | Use Case                       | Tradeoff                              |
| ------- | ------------------------------ | ------------------------------------- |
| k=1-2   | Simple, specific queries       | May miss relevant context             |
| k=3-5   | General queries (recommended)  | Good balance of context and precision |
| k=6-10  | Complex, multi-faceted queries | May include irrelevant documents      |

**Rule of thumb**: Start with `k=4` and adjust based on query complexity.

### 2. Metadata Filtering

Filtering enables scoped searches:

```python
# Search only in specific document
metadata_filter={"source": "hr_handbook.pdf"}

# Search specific page range
metadata_filter={"page": {"$gte": 10, "$lte": 20}}

# Search by custom metadata
metadata_filter={"department": "HR"}

# Combine multiple filters
metadata_filter={"source": "manual.pdf", "page": 5}
```

**Benefits:**

- Faster retrieval (smaller search space)
- More relevant results (focused context)
- Better for multi-document collections

### 3. Search Types

#### Similarity Search (Default)

- Returns documents most similar to query
- Best for: Direct question-answer scenarios
- Fast and straightforward

#### MMR (Maximum Marginal Relevance)

- Balances relevance with diversity
- Reduces redundant results
- Best for: Exploratory queries, research tasks

**When to use MMR:**

- Query might return many similar chunks
- Want diverse perspectives on a topic
- Avoiding redundant information

### 4. Embedding Model Consistency

**Critical**: Always use the same embedding model for retrieval as was used during ingestion. Different models produce different vector spaces, making similarity search meaningless.

## Usage Examples

### Basic Retrieval

```python
from main.vector_store import load_env, get_embeddings, get_connection_string, get_vector_store
from main.retriever import create_retriever

env = load_env()
embeddings = get_embeddings(env)
connection_string = get_connection_string(env)
vector_store = get_vector_store(connection_string, env["COLLECTION_NAME"], embeddings)

retriever = create_retriever(vector_store, k=4)
docs = retriever.invoke("What is the vacation policy?")
```

### LCEL Chain

```python
from main.lcel import create_lcel_chain

chain = create_lcel_chain(vector_store, k=4)
result = chain.invoke("What are the employee benefits?")
```

## Common Questions

### Q: Why do I get no results?

**A**: Check the following:

1. Collection name matches ingestion collection
2. Embeddings exist in database
3. Query is semantically related to stored content
4. Database connection is working

### Q: How do I know if k is too high or too low?

**A**:

- **Too low (k=1-2)**: Missing relevant information, incomplete answers
- **Too high (k>10)**: Irrelevant documents, noise in results
- **Just right**: All relevant info present, minimal noise

### Q: Can I use different k values for different queries?

**A**: Yes! In production, you might:

- Use k=3 for simple queries
- Use k=5-7 for complex queries
- Use k=10+ for research/exploratory queries

### Q: What's the difference between similarity and MMR?

**A**:

- **Similarity**: Returns most similar documents (may be redundant)
- **MMR**: Returns diverse documents (avoids redundancy, may sacrifice some relevance)

### Q: How do I filter by multiple metadata fields?

**A**: Combine filters in a dictionary:

```python
metadata_filter={
    "source": "hr_handbook.pdf",
    "page": {"$gte": 10},
    "department": "HR"
}
```

## Troubleshooting

### Issue: No Results Returned

```
✗ No documents retrieved for query
```

**Solution**:

1. Verify collection name matches ingestion: `SELECT name FROM langchain_pg_collection;`
2. Check if embeddings exist: `SELECT COUNT(*) FROM langchain_pg_embedding;`
3. Verify query is semantically related to stored content
4. Try increasing k value

### Issue: Irrelevant Results

**Solution**:

1. Lower k value (try k=3 instead of k=10)
2. Add metadata filters to narrow search scope
3. Verify embedding model matches ingestion model
4. Check if query is too vague or ambiguous

### Issue: Missing Relevant Information

**Solution**:

1. Increase k value (try k=5-7)
2. Remove metadata filters if too restrictive
3. Verify relevant documents were ingested
4. Check if query needs to be more specific

### Issue: Connection Errors

```
✗ Error: Unable to initialize vector store
```

**Solution**:

1. Verify database credentials in `.env`
2. Test database connection: `psql -h localhost -U your_user -d your_database`
3. Ensure pgvector extension is installed: `CREATE EXTENSION IF NOT EXISTS vector;`
4. Check network connectivity to database

### Issue: MMR Not Working

```
✗ MMR not available: [error message]
```

**Solution**:

1. Some vector stores require additional configuration for MMR
2. Verify your pgvector version supports MMR
3. Fall back to similarity search if MMR unavailable
4. Check LangChain documentation for MMR requirements

### Issue: Import Errors

```
ModuleNotFoundError: No module named 'langchain'
```

**Solution**:

```bash
uv sync
```

## Next Steps

1. **Fine-tune retrieval parameters** - Optimize k values, filters, and search types for your use case
2. **Integrate with LLM** - Combine retrieval with language models for complete RAG pipeline
3. **Production deployment** - Deploy as a service with proper error handling and monitoring


## Related Files

- `main/runner.py` - Test runner script
- `main/retriever.py` - Retriever creation utilities
- `main/lcel.py` - LCEL chain creation
- `main/vector_store.py` - Vector store connection management
- `main/utils.py` - Formatting and testing utilities

## Code Pattern

```python
# Complete retrieval workflow

from main.vector_store import load_env, get_embeddings, get_connection_string, get_vector_store
from main.retriever import create_retriever

# 1. Initialize components
env = load_env()
embeddings = get_embeddings(env)
connection_string = get_connection_string(env)
vector_store = get_vector_store(connection_string, env["COLLECTION_NAME"], embeddings)

# 2. Create retriever with configuration
retriever = create_retriever(
    vector_store,
    k=4,                                    # Number of documents to retrieve
    search_type="similarity",               # or "mmr"
    metadata_filter={"source": "hr_handbook.pdf"}  # Optional filter
)

# 3. Query the retriever
docs = retriever.invoke("What is the vacation policy?")

# 4. Process results
for doc in docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...")
    print()
```

## Verify Retrieval in PostgreSQL (step-by-step)

Use these steps to confirm that retrieval is working correctly with your Postgres database.

### 1) Ensure your database is reachable

Update your `.env` with valid credentials:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=your_user
DB_PASSWORD=your_password
COLLECTION_NAME=company_policies   # Must match ingestion collection
```

### 2) Verify collection exists

```sql
SELECT uuid, name, cmetadata
FROM public.langchain_pg_collection
WHERE name = 'company_policies';
```

### 3) Check stored embeddings

```sql
SELECT COUNT(*) AS chunk_count
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'company_policies';
```

### 4) Test retrieval query

Run the application and verify it retrieves documents:

```bash
uv run python main.py
```

### 5) Verify embedding model consistency

Ensure you're using the same embedding model as ingestion:

```sql
-- Check embedding dimensions (should match your model)
SELECT vector_dims(embedding) AS dims
FROM langchain_pg_embedding
LIMIT 1;
```

**Common dimensions:**

- `text-embedding-3-small`: 1536
- `text-embedding-3-large`: 3072
- `text-embedding-ada-002`: 1536

### 6) Test similarity search manually (optional)

```sql
-- Get a sample embedding
SELECT embedding
FROM langchain_pg_embedding
LIMIT 1;

-- Use this embedding to test similarity search
-- (This requires generating a query embedding first)
```

### 7) Common issues and fixes

- ❌ Collection not found

  - Verify collection name in `.env` matches database
  - Check if ingestion completed successfully

- ❌ No embeddings in collection

  - Run ingestion pipeline first
  - Verify embeddings were stored correctly

- ❌ Wrong embedding model

  - Ensure `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` matches ingestion
  - Check embedding dimensions match

- ❌ Connection errors
  - Verify database credentials
  - Test connection: `psql -h localhost -U your_user -d your_database`

