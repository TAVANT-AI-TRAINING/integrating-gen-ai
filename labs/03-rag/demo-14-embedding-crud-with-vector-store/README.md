# Demo 3: Embedding API with ChromaDB

A **FastAPI** application that demonstrates **CRUD operations** for embeddings using **ChromaDB** (embedded vector database) and LangChain. This example shows how to embed documents, store them in a vector database, and perform create/read/delete operations via REST API.

## Objective

To demonstrate the embedding storage and retrieval workflow:

1. **Initialize Vector Database**: Configure ChromaDB with local persistence
2. **Generate Embeddings**: Convert text documents into high-dimensional vectors using OpenAI
3. **Store Embeddings**: Persist embeddings in ChromaDB using LangChain's Chroma
4. **CRUD Operations**: Create, read, and delete embeddings via REST API endpoints
5. **Semantic Search**: Query similar documents using `similarity_search_with_score` method
6. **Modular Architecture**: Demonstrate separation of concerns with models, routes, and service layers

## Project Structure

```
demo-3-embedding-API-with-ChromaDB/
├── .env                    # Environment variables (API credentials)
├── .env.example            # Environment template
├── .gitignore
├── .python-version
├── main.py                 # FastAPI server entry point
├── main/                   # Main package
│   ├── __init__.py
│   ├── app.py              # FastAPI application instance with centralized exception handling
│   ├── models/             # Pydantic models package
│   │   ├── __init__.py     # Model exports
│   │   └── models.py       # Request/response models
│   ├── routes/             # API routes package
│   │   ├── __init__.py     # Router exports
│   │   └── routes.py       # API endpoint definitions
│   └── service/            # Service layer package
│       ├── __init__.py
│       └── embedding_service.py  # Core database and embedding functions
├── test/                   # Test package
│   ├── __init__.py
│   ├── conftest.py         # Pytest configuration and shared fixtures
│   ├── test_routes.py      # Tests for FastAPI endpoints
│   └── test_store_embeddings.py  # Tests for embedding_service module
├── chroma_db/              # ChromaDB persistence directory (created automatically)
├── pytest.ini              # Pytest configuration
├── pyproject.toml          # UV dependency configuration
└── README.md               # This file
```

## Setup Instructions

1. **Install Dependencies**

   ```bash
   uv sync
   ```

2. **Environment Configuration**

   Create a `.env` file in the project root:

   ```env
   # ChromaDB configuration
   COLLECTION_NAME=company_policies
   PERSIST_DIRECTORY=./chroma_db

   # OpenAI Configuration
   OPENAI_API_EMBEDDING_KEY=your_openai_api_key
   OPENAI_MODEL=text-embedding-3-small
   ```

   **Get your OpenAI API key:** [OpenAI Platform](https://platform.openai.com/api-keys)

3. **Run the FastAPI Server**

   ```bash
   # Using the entry point script
   uv run python main.py

   # Or directly with uvicorn
   uv run uvicorn main.app:app --host 0.0.0.0 --port 8000 --reload
   ```

   The server will start on `http://localhost:8000`

4. **Test the API**
   - Open your browser to `http://localhost:8000/docs` for interactive API documentation
   - Or send requests to the API endpoints

5. **Run Tests**

   ```bash
   uv run pytest
   ```

   Run with coverage:

   ```bash
   uv run pytest --cov=main --cov-report=html
   ```

## Features

- ✅ **Embedded Vector Database**: Persists embeddings locally using ChromaDB (no external database required)
- ✅ **CRUD Operations**: Create, read, and delete embeddings via REST API
- ✅ **Semantic Search**: Query similar documents using `similarity_search_with_score`
- ✅ **LangChain Integration**: Uses `langchain_chroma.Chroma` for vector operations
- ✅ **OpenAI Embeddings**: Generates high-quality text embeddings using OpenAI
- ✅ **FastAPI Interface**: RESTful endpoints with automatic API documentation
- ✅ **Comprehensive Swagger Documentation**: Interactive API docs with examples for all endpoints
- ✅ **Environment-Based Configuration**: Secure API key management via `.env` file
- ✅ **Centralized Exception Handling**: Single exception handler for consistent error responses
- ✅ **Modular Architecture**: Separated models, routes, and service layers
- ✅ **Universal Vector Store**: Single reusable vector store instance for all operations
- ✅ **No Database Setup Required**: ChromaDB runs embedded - no PostgreSQL installation needed

## API Endpoints

### POST /embed_and_store

Embed and persist a single document.

**Request Body:**

```json
{
  "text": "Our vacation policy allows 20 days off.",
  "doc_id": "optional-custom-id",
  "metadata": { "source": "hr_manual" }
}
```

**Response:**

```json
{
  "message": "Document embedded and stored successfully",
  "doc_id": "optional-custom-id"
}
```

**Example (cURL):**

```bash
curl -X POST http://localhost:8000/embed_and_store \
  -H "Content-Type: application/json" \
  -d '{"text":"Our vacation policy allows 20 days off.","metadata":{"source":"hr_manual"}}'
```

### POST /get_embedding

Retrieve embedding by document ID.

**Request Body:**

```json
{
  "doc_id": "<uuid or custom id>"
}
```

**Response:**

```json
{
  "doc_id": "abc123",
  "embedding": [0.123, -0.456, 0.789, ...],
  "metadata": { "source": "hr_manual", "id": "abc123" },
  "page_content": "Our vacation policy allows 20 days off."
}
```

**Example (cURL):**

```bash
curl -X POST http://localhost:8000/get_embedding \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"<doc_id>"}'
```

### DELETE /delete_embedding

Delete embedding by document ID.

**Request Body:**

```json
{
  "doc_id": "<uuid or custom id>"
}
```

**Response:**

```json
{
  "message": "Document with ID 'abc123' deleted successfully",
  "doc_id": "abc123"
}
```

**Example (cURL):**

```bash
curl -X DELETE http://localhost:8000/delete_embedding \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"<doc_id>"}'
```

### POST /query

Query similar documents using semantic similarity search. This endpoint uses `similarity_search_with_score` to find the most similar documents to the query.

**Request Body:**

```json
{
  "query": "AI in healthcare",
  "top_k": 5,
  "filters": null
}
```

**Response:**

```json
{
  "query": "AI in healthcare",
  "results": [
    {
      "doc_id": "it_security_policy_007",
      "score": 0.8547,
      "metadata": {
        "id": "it_security_policy_007",
        "source": "it_security_policy",
        "page": 7
      },
      "page_content": "All devices must be password protected and use multi-factor authentication when accessing company systems."
    },
    {
      "doc_id": "data_policy_009",
      "score": 0.8234,
      "metadata": {
        "id": "data_policy_009",
        "source": "data_policy",
        "page": 9
      },
      "page_content": "Data backups are performed automatically every 24 hours to ensure business continuity."
    }
  ]
}
```

**Example (cURL):**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"AI in healthcare","top_k":5}'
```

**Example (Python):**

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "AI in healthcare",
        "top_k": 5
    }
)
results = response.json()
for hit in results["results"]:
    print(f"Score: {hit['score']:.4f} - {hit['page_content'][:100]}...")
```

**Query with Filters:**

Filters allow you to restrict search results to documents matching specific metadata criteria. Available metadata fields include:

- `source`: Document source (e.g., "hr_manual", "it_security_policy", "data_policy")
- `page`: Page number
- `id`: Document ID

**Note:** Filters are optional. You can:

- Omit the `filters` field entirely
- Set `filters` to `null`
- Set `filters` to an empty object `{}` (will be treated as no filter)
- Provide a non-empty dictionary with filter criteria

**Example 1: Filter by source (cURL):**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "security policies",
    "top_k": 5,
    "filters": {
      "source": "it_security_policy"
    }
  }'
```

**Example 2: Filter by source (Python):**

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "employee benefits",
        "top_k": 5,
        "filters": {
            "source": "hr_manual"
        }
    }
)
results = response.json()
print(f"Found {len(results['results'])} HR manual documents")
for hit in results["results"]:
    print(f"Score: {hit['score']:.4f} - {hit['page_content'][:80]}...")
```

**Example 3: Filter by multiple metadata fields:**

```python
import requests

# Filter by source and page number
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "security",
        "top_k": 10,
        "filters": {
            "source": "it_security_policy",
            "page": 7
        }
    }
)
results = response.json()
```

**Example 4: Query without filters (returns all matching documents):**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "company policies",
    "top_k": 10,
    "filters": null
  }'
```

**Available source values for filtering:**

- `"hr_manual"` - HR policy documents
- `"it_security_policy"` - IT security policies
- `"data_policy"` - Data management policies
- `"compliance_guide"` - Compliance documentation
- `"performance_policy"` - Performance review policies
- `"ethics_manual"` - Ethics and compliance
- `"confidentiality_policy"` - Confidentiality policies
- `"operations_guide"` - Operations documentation
- `"communication_policy"` - Communication policies
- `"facilities_policy"` - Facilities management policies

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI) with:

- **Try-it-out functionality**: Test endpoints directly from the browser
- **Request/response schemas**: Complete data models with validation rules
- **Example requests and responses**: Pre-filled examples for all endpoints
- **Error code documentation**: Detailed error responses
- **Multiple examples**: Different scenarios for query endpoints (with/without filters)

The Swagger documentation includes comprehensive examples for:

- Embedding and storing documents
- Retrieving embeddings by ID
- Deleting documents
- Querying similar documents (with and without metadata filters)

## Use Cases

This embedding storage functionality is useful for:

- **RAG Systems**: Building retrieval-augmented generation pipelines with persistent vector storage
- **Semantic Search**: Storing and retrieving documents based on semantic similarity
- **Document Management**: Managing embeddings for large document collections
- **Knowledge Bases**: Creating searchable knowledge bases with vector embeddings
- **Content Recommendation**: Storing content embeddings for recommendation systems
- **Enterprise Applications**: HR/IT policy management and retrieval systems

## Expected Behavior

When you run the application, you'll observe:

- ✅ **ChromaDB Initialization**: Local vector database created in the `chroma_db` directory
- ✅ **Embedding Generation**: Text documents converted to high-dimensional vectors
- ✅ **Vector Storage**: Embeddings persisted locally in ChromaDB
- ✅ **API Endpoints**: RESTful endpoints for create, read, and delete operations
- ✅ **Structured Responses**: Consistent JSON format with proper validation
- ✅ **Error Handling**: Graceful handling of missing documents and API failures

## Testing

The project includes comprehensive test suites for all components:

### Test Files

- **`test/conftest.py`**: Pytest configuration and shared fixtures
  - Sets up mock environment variables for testing
  - Provides fixtures for mocked database connections, embeddings model, and vectorstore
  - Configures test isolation with automatic mock resets

- **`test/test_routes.py`**: Tests for FastAPI endpoints
  - Tests for `POST /embed_and_store` endpoint
  - Tests for `POST /get_embedding` endpoint
  - Tests for `DELETE /delete_embedding` endpoint
  - Request/response validation tests
  - Error handling and HTTP status code tests

- **`test/test_store_embeddings.py`**: Tests for embedding_service module
  - Tests for `embed_and_store_text()` function
  - Tests for `get_embedding_by_id()` function
  - Tests for `delete_by_doc_id()` function
  - Tests for `query_similar_documents()` function
  - Edge cases and error handling tests
  - Input validation tests

### Running Tests

Run all tests:

```bash
uv run pytest
```

Run specific test file:

```bash
uv run pytest test/test_routes.py
uv run pytest test/test_store_embeddings.py
```

Run with verbose output:

```bash
uv run pytest -v
```

Run with coverage report:

```bash
uv run pytest --cov=main --cov-report=html
```

View coverage report:

```bash
# HTML report will be generated in htmlcov/index.html
```

## Verify Data in ChromaDB

ChromaDB stores data locally in the persistence directory. You can verify and inspect the data programmatically:

### Using Python Script

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_EMBEDDING_KEY"),
    model=os.getenv("OPENAI_MODEL", "text-embedding-3-small")
)

# Connect to existing ChromaDB
vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name=os.getenv("COLLECTION_NAME", "company_policies"),
    persist_directory=os.getenv("PERSIST_DIRECTORY", "./chroma_db")
)

# Get the collection
collection = vectorstore._collection

# Count documents
count = collection.count()
print(f"Total documents: {count}")

# Get all document IDs
results = collection.get()
print(f"Document IDs: {results['ids']}")

# Get a specific document by ID
doc_id = "your-doc-id"
doc = collection.get(ids=[doc_id], include=["embeddings", "metadatas", "documents"])
print(f"Document: {doc}")
```

### Using ChromaDB Client

```python
import chromadb

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Get collection
collection = client.get_collection("company_policies")

# Peek at first few documents
print(collection.peek())

# Count documents
print(f"Total documents: {collection.count()}")
```

## Troubleshooting

| Issue                           | Solution                                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **OpenAI credentials not set**  | Add `OPENAI_API_EMBEDDING_KEY` to your `.env` file. Get your API key from https://platform.openai.com/api-keys |
| **ChromaDB persistence errors** | Ensure `PERSIST_DIRECTORY` path is writable and has sufficient disk space                                      |
| **Import errors**               | Ensure all dependencies are installed with `uv sync`                                                           |
| **Rate limit errors**           | OpenAI has rate limits. Consider adding retry logic or upgrading your OpenAI plan                              |
| **Collection not found**        | Verify the `COLLECTION_NAME` in your `.env` file matches the collection you're querying                        |
