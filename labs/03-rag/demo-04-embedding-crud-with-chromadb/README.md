# Demo 04: Embedding CRUD with ChromaDB

A simple demonstration of CRUD (Create, Read, Update, Delete) operations with ChromaDB using OpenAI embeddings, exposed via FastAPI.

## 🎯 What You'll Learn

- How to use ChromaDB for vector storage
- Generate embeddings using OpenAI
- Perform CRUD operations on embeddings
- Semantic similarity search
- Build a RESTful API with FastAPI

## 📦 What's Inside

This demo is **intentionally kept simple** - everything is in a single `main.py` file to make it easy to understand:

1. **ChromaDB Setup** - Persistent vector database
2. **OpenAI Embeddings** - Convert text to vectors
3. **CRUD Operations** - Create, Read, Update, Delete
4. **Semantic Search** - Find similar documents
5. **FastAPI** - REST API endpoints

## 🚀 Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_DB_DIR=./chroma_db
```

### 3. Run the Server

```bash
uv run python main.py
```

The server will start at `http://localhost:8000`

## 📚 API Endpoints

### Health Check

```bash
curl http://localhost:8000/
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
    "text": "All employees must complete annual security training and pass the assessment.",
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

## 🧪 Try It Out

### Example 1: Add Company Policies

```bash
# Add vacation policy
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-vacation",
    "text": "Employees receive 15 days of paid vacation annually.",
    "metadata": {"category": "benefits", "department": "HR"}
  }'

# Add remote work policy
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-remote",
    "text": "Remote work is allowed up to 3 days per week with manager approval.",
    "metadata": {"category": "workplace", "department": "HR"}
  }'

# Add security policy
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-security",
    "text": "All company data must be encrypted and passwords must be changed every 90 days.",
    "metadata": {"category": "security", "department": "IT"}
  }'
```

### Example 2: Search for Related Policies

```bash
# Find policies about working from home
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Can I work from home?",
    "n_results": 2
  }'

# Find policies about time off
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "How much time off do I get?",
    "n_results": 2
  }'
```

## 🔍 Interactive API Documentation

FastAPI provides automatic interactive documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 Project Structure

```
demo-04-embedding-crud-with-chromadb/
├── main.py              # All code in one file (for simplicity)
├── pyproject.toml       # Dependencies
├── .env.example         # Environment template
├── .env                 # Your configuration (git-ignored)
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── chroma_db/           # ChromaDB storage (created automatically)
```

## 🧠 How It Works

### 1. Embedding Generation

When you add text, OpenAI's `text-embedding-3-small` model converts it into a vector (array of numbers).

### 2. Vector Storage

ChromaDB stores:

- The original text
- The embedding vector
- Metadata (any additional info)
- A unique ID

### 3. Semantic Search

When you query:

1. Your query text is converted to an embedding
2. ChromaDB finds similar embeddings (cosine similarity)
3. Returns the most relevant documents

### 4. CRUD Operations

- **Create**: Generate embedding + store in ChromaDB
- **Read**: Retrieve document by ID
- **Update**: Generate new embedding + update in ChromaDB
- **Delete**: Remove from ChromaDB

## 💡 Key Concepts

### What is a Vector Database?

A database optimized for storing and searching high-dimensional vectors (embeddings).

### What are Embeddings?

Numerical representations of text that capture semantic meaning. Similar texts have similar embeddings.

### Why ChromaDB?

- Simple to use
- No separate server needed
- Persistent storage
- Fast similarity search

### Why OpenAI Embeddings?

- High quality semantic representations
- Pre-trained on vast amounts of data
- Easy to use via API

## 🎓 Learning Notes

### Simple Design

This demo uses a single file to make it easier to understand. In production, you'd typically split into:

- `models.py` - Pydantic models
- `database.py` - ChromaDB operations
- `routes.py` - FastAPI endpoints
- `main.py` - App initialization

### Embedding Model

We use `text-embedding-3-small` which:

- Creates 1536-dimensional vectors
- Costs $0.02 per 1M tokens
- Good balance of quality and cost

### Metadata

Store additional info like:

- Category, department, date
- Useful for filtering and organization
- Doesn't affect embedding

## 🔧 Troubleshooting

### Error: "OPENAI_API_KEY not found"

Make sure you've created `.env` file with your API key.

### Error: Port 8000 already in use

Stop other servers or change the port in `main.py`.

### ChromaDB Permission Error

Ensure you have write permissions in the project directory.

## 📚 Next Steps

After mastering this demo:

1. Try different embedding models
2. Add filtering to queries
3. Implement batch operations
4. Add authentication
5. Deploy to production

## 🤝 Collection Name

This demo uses `company_policies` as the collection name, perfect for storing organizational policies, procedures, and documentation.
