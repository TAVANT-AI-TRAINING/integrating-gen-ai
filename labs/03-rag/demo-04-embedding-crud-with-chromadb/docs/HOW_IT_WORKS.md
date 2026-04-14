# How ChromaDB Embedding CRUD Works

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                           │
│  (main.py - handles HTTP requests and routes to functions)      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
        │   CREATE    │  │    READ     │  │   UPDATE    │  ...
        │             │  │             │  │             │
        └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
               │                │                │
               ▼                │                ▼
        ┌─────────────┐         │         ┌─────────────┐
        │   OpenAI    │         │         │   OpenAI    │
        │  Embedding  │         │         │  Embedding  │
        │     API     │         │         │     API     │
        └──────┬──────┘         │         └──────┬──────┘
               │                │                │
               │  (vector)      │                │  (new vector)
               │                │                │
               └────────────────┼────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      ChromaDB         │
                    │  (Vector Database)    │
                    │                       │
                    │  Stores:              │
                    │  - Text               │
                    │  - Embeddings         │
                    │  - Metadata           │
                    │  - Document IDs       │
                    └───────────────────────┘
```

## 📝 Step-by-Step: Creating a Document

### What Happens When You POST a Document

```
1. Client sends request:
   POST /documents
   {
     "doc_id": "policy-001",
     "text": "All employees must complete training.",
     "metadata": {"category": "security"}
   }

2. FastAPI receives request
   ↓
3. create_document() function called
   ↓
4. create_embedding() sends text to OpenAI
   "All employees must complete training."
   ↓
5. OpenAI returns embedding vector:
   [0.123, -0.456, 0.789, ... 1533 more numbers]
   ↓
6. collection.add() stores in ChromaDB:
   - ID: "policy-001"
   - Text: "All employees must complete training."
   - Embedding: [0.123, -0.456, ...]
   - Metadata: {"category": "security"}
   ↓
7. Return response to client:
   {
     "doc_id": "policy-001",
     "text": "All employees must complete training.",
     "metadata": {"category": "security"},
     "embedding_length": 1536
   }
```

## 🔍 Step-by-Step: Semantic Search

### What Happens When You Query

```
1. Client sends query:
   POST /query
   {
     "query_text": "What are the training requirements?",
     "n_results": 3
   }

2. FastAPI receives request
   ↓
3. query_similar_documents() function called
   ↓
4. create_embedding() converts query to vector
   "What are the training requirements?"
   ↓
5. OpenAI returns query embedding:
   [0.134, -0.423, 0.801, ... ]
   ↓
6. collection.query() searches ChromaDB
   - Compares query embedding to all stored embeddings
   - Uses cosine similarity
   - Finds 3 most similar documents
   ↓
7. ChromaDB returns matches:
   [
     {
       "doc_id": "policy-001",
       "text": "All employees must complete training.",
       "distance": 0.123  (closer = more similar)
     },
     {
       "doc_id": "policy-015",
       "text": "Annual security training is mandatory.",
       "distance": 0.234
     },
     ...
   ]
   ↓
8. Results returned to client
```

## 🧠 Understanding Embeddings

### What is an Embedding?

An embedding is a **numerical representation** of text:

```
Text: "Remote work policy"
         ↓ (OpenAI API)
Embedding: [0.123, -0.456, 0.789, ... 1536 numbers total]
```

### Why Embeddings?

**Without Embeddings (Keyword Search):**

- Query: "work from home"
- Finds: Documents with exact words "work", "from", "home"
- Misses: "remote work", "telecommuting", "WFH"

**With Embeddings (Semantic Search):**

- Query: "work from home"
- Embedding: [0.234, -0.567, ...]
- Finds similar embeddings regardless of exact words
- Finds: "remote work", "telecommuting", "WFH", "flexible location"

### Similarity Calculation

ChromaDB uses **cosine similarity**:

```
Vector A: [0.1, 0.2, 0.3]
Vector B: [0.15, 0.25, 0.35]
Vector C: [0.9, -0.8, 0.1]

Similarity(A, B) = 0.99  (very similar!)
Similarity(A, C) = 0.23  (not similar)
```

## 🗄️ ChromaDB Storage

### What Gets Stored?

```python
Document {
    id: "policy-001",                    # Unique identifier
    document: "Original text here...",   # Actual text
    embedding: [0.123, -0.456, ...],    # 1536 numbers
    metadata: {                          # Extra info
        "category": "security",
        "date": "2024-01-15"
    }
}
```

### Persistent Storage

```
./chroma_db/
├── chroma.sqlite3        # Database file
└── [collection folders]  # Vector data
```

Data persists across restarts!

## 🔄 CRUD Operations Summary

| Operation  | Endpoint               | What Happens                            |
| ---------- | ---------------------- | --------------------------------------- |
| **CREATE** | POST /documents        | Generate embedding → Store in DB        |
| **READ**   | GET /documents/{id}    | Retrieve from DB by ID                  |
| **UPDATE** | PUT /documents/{id}    | Generate new embedding → Update DB      |
| **DELETE** | DELETE /documents/{id} | Remove from DB                          |
| **QUERY**  | POST /query            | Generate query embedding → Find similar |

## 💡 Key Insights

1. **Every create/update generates a new embedding**
   - Costs API calls to OpenAI
   - Takes ~100-500ms

2. **Reads are fast**
   - No API call needed
   - Just database lookup

3. **Similarity search is the magic**
   - Finds semantically related content
   - Even if words are different

4. **Metadata is powerful**
   - Doesn't affect the embedding
   - Useful for filtering/organizing
   - Returned with search results

## 🎯 Why This Architecture?

- **Simple**: One file, easy to understand
- **Practical**: Real-world CRUD operations
- **Educational**: See exactly what happens at each step
- **Scalable**: Pattern works for production too (with modifications)
