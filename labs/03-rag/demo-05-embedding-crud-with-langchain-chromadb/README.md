# Demo 05: Embedding CRUD with LangChain & ChromaDB

A demonstration of CRUD operations using **LangChain's abstractions** for ChromaDB and OpenAI embeddings, exposed via FastAPI.

## 🎯 Why LangChain?

**Demo 04 (Direct ChromaDB)** vs **Demo 05 (LangChain)**

| Feature        | Demo 04 (Direct)        | Demo 05 (LangChain)      | Benefit                  |
| -------------- | ----------------------- | ------------------------ | ------------------------ |
| **Embeddings** | Manual OpenAI API calls | `OpenAIEmbeddings()`     | Automatic handling       |
| **Storage**    | Direct ChromaDB calls   | `Chroma.add_documents()` | Higher-level API         |
| **Search**     | Manual query embedding  | `similarity_search()`    | One-line semantic search |
| **Documents**  | Dict/manual structure   | `Document` class         | Structured data          |
| **Code**       | More verbose            | More concise             | Easier to read           |

## 📦 What You'll Learn

- How LangChain simplifies embedding workflows
- Using `OpenAIEmbeddings` for automatic embedding generation
- Using `Chroma` vector store with high-level operations
- Working with LangChain's `Document` class
- Building production-ready APIs with minimal code

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
OPENAI_API_EMBEDDING_KEY=sk-your-actual-api-key-here
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

### Example: Add and Search Company Policies

```bash
# Add vacation policy
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-vacation",
    "text": "Employees receive 15 days of paid vacation annually.",
    "metadata": {"category": "benefits"}
  }'

# Add remote work policy
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-remote",
    "text": "Remote work is allowed up to 3 days per week.",
    "metadata": {"category": "workplace"}
  }'

# Search for related policies
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Can I work from home?",
    "n_results": 2
  }'
```

## 🔍 Interactive API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 Project Structure

```
demo-05-embedding-crud-with-langchain-chromadb/
├── main.py              # LangChain implementation
├── pyproject.toml       # Dependencies (includes LangChain)
├── .env.example         # Environment template
├── .env                 # Your configuration (git-ignored)
├── README.md            # This file
├── QUICKSTART.md        # Quick setup guide
├── LANGCHAIN_BENEFITS.md # Why use LangChain?
└── chroma_db/           # ChromaDB storage
```

## 🔑 Key LangChain Components

### 1. OpenAIEmbeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# LangChain handles all API calls automatically!
```

### 2. Chroma Vector Store

```python
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="company_policies",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# High-level operations built-in
```

### 3. Document Class

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Your text here",
    metadata={"key": "value"}
)

# Structured, consistent data format
```

## 💡 LangChain Advantages

### Before (Demo 04 - Direct ChromaDB):

```python
# Manual embedding generation
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)
embedding = response.data[0].embedding

# Manual storage
collection.add(
    ids=[doc_id],
    embeddings=[embedding],
    documents=[text],
    metadatas=[metadata]
)
```

### After (Demo 05 - LangChain):

```python
# Automatic embedding + storage
document = Document(page_content=text, metadata=metadata)
vectorstore.add_documents(documents=[document], ids=[doc_id])

# That's it! LangChain handles everything
```

### Semantic Search Comparison:

**Before (Demo 04):**

```python
# Generate query embedding
query_embedding = create_embedding(query_text)

# Search manually
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=n
)

# Parse results manually
for i in range(len(results['ids'][0])):
    # Extract and format each result...
```

**After (Demo 05):**

```python
# One line!
results = vectorstore.similarity_search_with_score(
    query=query_text,
    k=n
)

# Results are already structured Document objects
```

## 🎓 Learning Path

1. **Start with Demo 04** - Understand the fundamentals
2. **Move to Demo 05** - See how LangChain simplifies things
3. **Compare the code** - Notice the reduction in complexity
4. **Use LangChain for production** - More maintainable, less code

## 🔧 When to Use What?

### Use Direct ChromaDB (Demo 04) When:

- Learning the fundamentals
- Need fine-grained control
- Custom embedding logic
- Very simple use case

### Use LangChain (Demo 05) When:

- Building production applications
- Want cleaner, more maintainable code
- Need to switch between vector stores easily
- Using other LangChain features (chains, agents)

## 🚀 Next Steps

After mastering this demo:

1. Explore LangChain chains (combine multiple operations)
2. Try different vector stores (switch from Chroma to Pinecone, etc.)
3. Add filters to similarity search
4. Implement RAG (Retrieval-Augmented Generation)
5. Use LangChain agents with your vector store

## 📖 Key Concepts

**LangChain Benefits:**

- **Abstraction**: Hide complexity, expose simple APIs
- **Portability**: Easy to switch between providers
- **Integration**: Works with many tools and services
- **Maintainability**: Less code, fewer bugs
- **Community**: Large ecosystem of tools and examples

**When LangChain Helps Most:**

- Rapid prototyping
- Production applications
- Complex workflows
- Multiple integrations
- Team collaboration

## 🤝 Collection Name

Uses `company_policies` collection for storing organizational documents.
