# Demo 10: RAG Retrieval Pipeline

This demo focuses **PURELY on the RETRIEVAL phase** by connecting to an **EXISTING vector database** and demonstrating various retrieval strategies with detailed chunk content display.

## What This Demo Covers

**Prerequisites**: Run **demo-09-rag-ingestion-pipeline** first to ingest documents  
**Focus**: Connect to existing vector store and demonstrate retrieval strategies  
**Does NOT include**:

- Document ingestion (see demo-09)
- LLM-based answer generation (see demo-11)

**Key Feature**: Shows 200 characters of content for each retrieved chunk!

## What is RAG Retrieval?

RAG retrieval is the process of finding the most relevant pieces of information from a knowledge base to answer a query. This demo connects to an **existing vector database** (populated by demo-09) and demonstrates various retrieval strategies.

### Workflow

```
PREREQUISITE: Run demo-09 first to populate vector database
   ┌─────────────┐
   │  Documents  │ → (Already ingested by demo-09)
   └─────────────┘
          ↓
   ┌─────────────┐
   │  Chunks     │ → (Already in vector DB)
   └─────────────┘
          ↓
   ┌─────────────┐
   │ Embeddings  │ → (Already generated)
   └─────────────┘
          ↓
   ┌─────────────┐
   │ Vector DB   │ → (ChromaDB or Pinecone)
   └─────────────┘

THIS DEMO: Connect & Retrieve
   ┌─────────────┐
   │   Connect   │ → Connect to existing vector DB
   └─────────────┘
          ↓
   ┌─────────────┐
   │   Verify    │ → Ensure data exists
   └─────────────┘
          ↓
   ┌─────────────┐
   │   Query     │ → User's question
   └─────────────┘
          ↓
   ┌─────────────┐
   │   Search    │ → Find similar chunks (6 strategies)
   └─────────────┘
          ↓
   ┌─────────────┐
   │  Results    │ → Show detailed chunks (200 chars each)
   └─────────────┘
```

## Key Features

✅ **Retrieval-Only** - Connects to existing vector database  
✅ **No Ingestion** - Assumes demo-09 already ran  
✅ **Multiple Scenarios** - 7 different retrieval demonstrations  
✅ **Detailed Output** - Shows 200 characters of each chunk  
✅ **Quality Analysis** - Compare results across k values  
✅ **Config-Driven** - Switch between ChromaDB and Pinecone  
✅ **Single File** - All code in main.py (~450 lines)  
✅ **Clear Output** - Relevance indicators (🟢🟡🔴)  
✅ **Standard OpenAI** - Only embeddings needed

## Prerequisites

**IMPORTANT**: You must run **demo-09-rag-ingestion-pipeline** first!

Demo-09 will:

- Load your documents (PDF, text, web)
- Chunk them into pieces
- Generate embeddings
- Store in vector database (ChromaDB or Pinecone)

Then this demo (demo-10) will connect to that populated vector store and demonstrate retrieval strategies.

Find top-k most similar documents:

```python
results = vectorstore.similarity_search(query, k=3)
```

### 2. Similarity Search with Scores

Get relevance scores for each result:

```python
results = vectorstore.similarity_search_with_score(query, k=3)
# Returns: [(doc, 0.5234), (doc, 0.7456), ...]
```

**Understanding Scores** (Distance-Based Metric):
- **Lower score = More relevant** (closer in vector space)
- **Higher score = Less relevant** (farther in vector space)
- Visual indicators:
  - 🟢 **< 0.6**: High relevance (close distance)
  - 🟡 **0.6-0.8**: Medium relevance
  - 🔴 **> 0.8**: Low relevance (far distance)

**Note**: These are distance scores, not similarity percentages. A score of 0.3 is better than 0.9!

### 3. MMR (Maximum Marginal Relevance)

Balance relevance with diversity:

```python
results = vectorstore.max_marginal_relevance_search(
    query, k=4, fetch_k=10
)
```

### 4. Retriever Interface

Use LangChain's retriever abstraction:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
results = retriever.invoke(query)
```

### 5. Metadata Filtering

Filter results by document properties (source, page, type, etc.):

```python
# Filter by specific document source
results = vectorstore.similarity_search(
    query, 
    k=3,
    filter={'source': 'company_policy.pdf'}
)

# Or use retriever interface
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"source": "company_policy.pdf"}
    }
)
```

**Common Filter Examples**:
- `{'source': 'file.pdf'}` - Specific document
- `{'page': 5}` - Specific page
- `{'type': 'policy'}` - Document category

**Benefits**:
- Limit results to specific documents
- Search within document subsets
- Combine semantic search with structured filtering

### 6. Quality Analysis

Compare retrieval quality across different k values:

```python
analyze_retrieval_quality(query, k_values=[1, 2, 3, 5])
```

## Prerequisites

- Python 3.12+
- OpenAI API key

## Installation

1. Set up Python environment:

```bash
cd demo-10-rag-retrieval-pipeline
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
uv pip install -e .
```

3. Configure environment:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Configuration

Edit `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Vector Database Selection (chromadb or pinecone)
VECTOR_DB=chromadb

# ChromaDB Settings (if using VECTOR_DB=chromadb)
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=company_policies

# Pinecone Settings (if using VECTOR_DB=pinecone)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=company-policies
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Usage

**Step 1**: First run demo-09 to ingest documents:

```bash
cd ../demo-09-rag-ingestion-pipeline
uv run python main.py
# This will populate the vector database
```

**Step 2**: Then run this demo to demonstrate retrieval:

```bash
cd ../demo-10-rag-retrieval-pipeline
uv run python main.py
# This will connect and show retrieval strategies
```

## What the Demo Does

### Phase 1: Connect & Verify

1. **Connect to Vector Store** - ChromaDB or Pinecone (config-driven)
2. **Verify Data Exists** - Checks if documents were ingested (by demo-09)
3. **Show Sample** - Displays a sample chunk to confirm connection

⚠️ **If vector store is empty**: Demo will guide you to run demo-09 first.

### Phase 2: Retrieval Demonstrations (7 Scenarios)

#### Scenario 1: Different K Values

Compare retrieval with k=2, k=4, k=6:

```
Query: "What are the key policies?"

[k=2] Retrieved 2 documents
  [1] Documents/company_policy.pdf...
  [2] Documents/guidelines.txt...

[k=4] Retrieved 4 documents
  [1] Documents/company_policy.pdf...
  [2] Documents/guidelines.txt...
  [3] Documents/policy.txt...
  [4] https://www.python.org/...
```

#### Scenario 2: Relevance Scores

See how relevant each result is with visual indicators (distance-based scores):

**Note**: Lower score = Higher relevance (closer in vector space)

```
Query: "remote work guidelines"
  (Lower score = Higher relevance)

✓ Retrieved 3 documents with scores

  [1] Score: 0.5234 🟢 High relevance (close distance)
      Source: Documents/company_policy.pdf
      Page: 3
      Content: Remote Work Policy  All employees are eligible for remote
               work arrangements up to 3 days per week. Requests must be
               submitted through the HR portal at least one week in advance...
      Length: 892 characters

  [2] Score: 0.6891 🟡 Medium relevance
      Source: Documents/guidelines.txt
      Content: Work from home guidelines require maintaining regular
               communication with team members during designated work hours.
               All remote workers must be available for video calls...
      Length: 654 characters

  [3] Score: 0.8234 🔴 Low relevance (far distance)
      Source: Documents/policy.txt
      Content: General company policies apply to all employees regardless
               of work location...
      Length: 423 characters
```

**Score Interpretation**:
- Scores represent **distance** in vector space (Euclidean/Cosine)
- **Lower scores are better** - they indicate semantic closeness
- Example: 0.3 is MORE relevant than 0.9

#### Scenario 3: MMR Search

Get diverse results (avoid redundancy):

```
Query: "company policies"

✓ Retrieved 4 diverse documents
  (Balances relevance with diversity)
  [1] Documents/company_policy.pdf (page 1)
  [2] Documents/company_policy.pdf (page 3)
  [3] Documents/guidelines.txt
  [4] https://www.python.org/...
```

#### Scenario 4: Retriever Interface

Use standard LangChain retriever pattern:

```
Query: "code review process"

✓ Retriever returned 4 documents
  [1] Documents/guidelines.txt...
  [2] Documents/company_policy.pdf...
```

#### Scenario 5: Quality Analysis

Compare metrics across k values:

```
Query: "What are the guidelines?"

--- k=1 ---
  Average relevance score: 0.8542
  Top result score: 0.8542

--- k=2 ---
  Average relevance score: 0.7888
  Top result score: 0.8542
  Bottom result score: 0.7234

--- k=3 ---
  Average relevance score: 0.7556
  Top result score: 0.8542
  Bottom result score: 0.6891
```

#### Scenario 6: Metadata Filtering

Filter results by document source or other metadata:

```
Query: "What are the guidelines?"
Filter: {'source': 'Documents/guidelines.txt'}
  (Only returns results matching metadata criteria)

✓ Retrieved 3 documents matching filter

  [1] Metadata:
      source: Documents/guidelines.txt
      Content: All employees must follow established guidelines for code
               reviews, including mandatory peer review for all pull requests,
               automated testing requirements, and documentation standards...
      Length: 745 characters

  [2] Metadata:
      source: Documents/guidelines.txt
      Content: Development guidelines require following PEP 8 for Python code,
               using type hints for all function signatures, maintaining test
               coverage above 80%, and writing clear commit messages...
      Length: 612 characters

  [3] Metadata:
      source: Documents/guidelines.txt
      Content: Security guidelines mandate code scanning before deployment,
               regular dependency updates, API key rotation, and security
               audit participation for all team members...
      Length: 523 characters
```

**Use Cases**:
- Search only within specific documents
- Filter by document type, page number, or custom metadata
- Combine semantic search with structured filtering

#### Scenario 7: Document Inspection

Deep dive into a retrieved document:

```
Query: "employee benefits"

======================================================================
DOCUMENT #1 DETAILS
======================================================================

[Metadata]
  source: Documents/company_policy.pdf
  page: 4

[Content]
  Length: 892 characters
  Preview: Employee Benefits  Full-time employees are eligible for
           comprehensive benefits including health insurance...
```

## Understanding the Output

### Phase 1: Verification

```
======================================================================
RAG RETRIEVAL PIPELINE - CONNECT TO EXISTING VECTOR STORE
======================================================================
Vector Database: CHROMADB
Chunk Preview Length: 200 characters

⚠️  Prerequisites: Run demo-09 first to ingest documents!
======================================================================

✓ OpenAI embeddings initialized: text-embedding-3-small
✓ ChromaDB initialized
  - Storage: ./chroma_db
  - Collection: company_policies
✓ Vector store ready!

======================================================================
VERIFYING VECTOR STORE DATA
======================================================================

✓ Vector store has data!
  - Found at least 127 chunks
  - Ready for retrieval demonstrations

📄 Sample chunk:
  Source: Documents/company_policy.pdf
  Content: Employee Handbook Introduction  This handbook outlines the
           key policies and procedures for all employees...
```

### Phase 2: Retrieval Demonstrations

```
======================================================================
DEMONSTRATING RETRIEVAL SCENARIOS
======================================================================

[Scenario 1] Comparing Different K Values
----------------------------------------------------------------------

[Retrieval] Similarity Search (k=2)
Query: "What are the key policies?"
----------------------------------------------------------------------

✓ Retrieved 2 documents

  [1] Metadata:
      Source: Documents/company_policy.pdf
      Page: 1
      Content: Company Policies Overview  The following policies apply to
               all employees. Please review carefully and direct questions
               to your supervisor or HR department. Key areas include...
      Length: 956 characters

  [2] Metadata:
      Source: Documents/guidelines.txt
      Content: General Guidelines  All employees must adhere to these
               guidelines for professional conduct and workplace safety...
      Length: 742 characters
```

## Key Concepts Explained

### 1. Similarity Search

Finds documents with embeddings most similar to the query embedding using cosine similarity.

**When to use**: When you want the most relevant results regardless of diversity.

### 2. Relevance Scores

Numerical scores (0-1) indicating how similar each result is to the query.

**Lower score = More similar** (in terms of distance)

### 3. K Value

Number of results to retrieve.

- **Low k (1-3)**: Only top matches (precision)
- **Medium k (4-6)**: Balance (recommended)
- **High k (10+)**: More coverage (recall)

### 4. MMR (Maximum Marginal Relevance)

Algorithm that balances relevance with diversity to avoid redundant results.

**Parameters**:

- `k`: Final number of results
- `fetch_k`: Initial candidates to consider
- `lambda_mult`: Balance factor (0=diversity, 1=relevance)

### 5. Metadata Filtering

Restrict search to documents matching specific criteria:

```python
filter = {"source": "company_policy.pdf"}
filter = {"page": 2}
filter = {"department": "engineering"}
```

### 6. Retriever Interface

LangChain abstraction for consistent retrieval across different vector stores.

**Benefits**:

- Consistent API
- Easy to swap vector databases
- Works with LCEL chains

## Code Structure

```python
# Configuration
- Load environment variables
- Initialize embeddings (OpenAI)
- Initialize vector store (ChromaDB or Pinecone)

# Step 1: load_documents()
- Load PDF, text, web

# Step 2: chunk_documents()
- Split into chunks
- Calculate statistics

# Step 3: store_chunks()
- Generate embeddings
- Store in vector database

# Step 4: Retrieval Functions
- similarity_search_basic()
- similarity_search_with_score()
- max_marginal_relevance_search()
- retriever_interface_demo()
- retriever_with_filter()
- analyze_retrieval_quality()
- display_document_details()

# Demonstration: demonstrate_retrieval_scenarios()
- Run 7 different scenarios
- Show various retrieval strategies
```

## Switching Vector Databases

### ChromaDB (Default):

```bash
# .env
VECTOR_DB=chromadb
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=company_policies
```

### Pinecone:

```bash
# .env
VECTOR_DB=pinecone
PINECONE_API_KEY=your-api-key
PINECONE_INDEX_NAME=company-policies
```

## Common Issues

### "OPENAI_API_KEY not found"

Copy `.env.example` to `.env` and add your API key.

### "No documents loaded"

Create `Documents/` folder and add PDF or text files.

### "MMR not supported"

Some vector stores don't support MMR. This is expected and handled gracefully.

### Low relevance scores

- Try different queries
- Adjust chunk size
- Add more documents
- Check document content quality

## Learning Objectives

After running this demo, you should understand:

1. ✓ How vector similarity search works
2. ✓ What relevance scores indicate
3. ✓ How k value affects results
4. ✓ When to use MMR for diversity
5. ✓ How to use the retriever interface
6. ✓ How metadata filtering works
7. ✓ How to analyze retrieval quality
8. ✓ What factors affect retrieval performance

## Improving Retrieval Quality

### 1. Adjust Chunk Size

```python
CHUNK_SIZE = 500   # Smaller chunks (more precise)
CHUNK_SIZE = 1500  # Larger chunks (more context)
```

### 2. Tune Chunk Overlap

```python
CHUNK_OVERLAP = 100  # Less overlap
CHUNK_OVERLAP = 300  # More overlap (better context preservation)
```

### 3. Optimize K Value

```python
k = 3   # Precision-focused
k = 5   # Balanced
k = 10  # Recall-focused
```

### 4. Use MMR for Diversity

When getting redundant results, use MMR instead of similarity search.

### 5. Add Metadata

Enrich documents with metadata for filtering:

```python
doc.metadata["department"] = "engineering"
doc.metadata["date"] = "2024-01-15"
doc.metadata["category"] = "policy"
```

## Next Steps

- **Add your documents**: Put your PDF/text files in `Documents/`
- **Try different queries**: Test retrieval with your questions
- **Adjust chunk size**: See how it affects results
- **Compare k values**: Find optimal k for your use case
- **Experiment with MMR**: Test diversity vs relevance trade-off
- **Complete RAG**: Move to demo-11 for LLM-based generation

## Related Demos

- **demo-07**: Vector database CRUD operations
- **demo-08**: Loading documents from multiple sources
- **demo-09**: RAG ingestion pipeline
- **demo-11**: Complete RAG with LLM generation

## Performance Notes

**ChromaDB**:

- ✓ Fast for < 100K documents
- ✓ No external dependencies
- ✓ Perfect for development and learning

**Pinecone**:

- ✓ Fast for millions of documents
- ✓ Cloud-based, always available
- ✓ Better for production use

## References

- [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [MMR Paper](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)

## License

Educational use only.
