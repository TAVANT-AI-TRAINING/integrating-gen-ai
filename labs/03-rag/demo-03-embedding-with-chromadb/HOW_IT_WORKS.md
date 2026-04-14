# How It Works

## Architecture Overview

```
┌─────────────┐
│   main.py   │
└──────┬──────┘
       │
       ├─────────────┐
       │             │
       ▼             ▼
┌──────────┐  ┌──────────────┐
│  OpenAI  │  │   ChromaDB   │
│Embeddings│  │ (Persistent) │
└──────────┘  └──────────────┘
```

## Step-by-Step Flow

### 1. Initialization

```python
# Load environment variables
load_dotenv()

# Create OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Create ChromaDB persistent client
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Demo documents collection"}
)
```

**What happens:**

- Environment variables are loaded from `.env`
- OpenAI client is initialized for generating embeddings
- ChromaDB client is created with persistent storage (data saved to disk)
- A collection is created or loaded (like a table in a database)

### 2. Creating Embeddings

```python
def create_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding
```

**What happens:**

- Text is sent to OpenAI's API
- OpenAI returns a vector (list of 1536 numbers for `text-embedding-3-small`)
- This vector represents the semantic meaning of the text
- Similar texts produce similar vectors

**Example:**

```
Input: "Python is a programming language"
Output: [0.123, -0.456, 0.789, ..., 0.321]  # 1536 numbers
```

### 3. Adding Documents (CREATE)

```python
def add_document(doc_id: str, text: str, metadata: Dict):
    # Generate embedding
    embedding = create_embedding(text)

    # Add to ChromaDB
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )
```

**What happens:**

1. Text is converted to an embedding vector
2. Document is stored in ChromaDB with:
   - **ID**: Unique identifier (e.g., "doc-001")
   - **Embedding**: The vector representation
   - **Document**: Original text
   - **Metadata**: Extra info (category, tags, etc.)
3. Data is persisted to disk in `./chroma_db/`

**Storage Structure:**

```
chroma_db/
├── chroma.sqlite3          # Metadata and document text
└── [uuid]/                 # Vector embeddings
    ├── data_level0.bin
    └── index_metadata.pickle
```

### 4. Retrieving Documents (READ)

```python
def get_document(doc_id: str):
    result = collection.get(
        ids=[doc_id],
        include=["documents", "metadatas", "embeddings"]
    )
    return result
```

**What happens:**

1. ChromaDB looks up the document by ID
2. Returns all requested fields:
   - Original text
   - Metadata
   - Embedding vector
3. Fast lookup - O(1) complexity

### 5. Updating Documents (UPDATE)

```python
def update_document(doc_id: str, new_text: str, new_metadata: Dict):
    # Generate new embedding
    embedding = create_embedding(new_text)

    # Update in ChromaDB
    collection.update(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[new_text],
        metadatas=[new_metadata]
    )
```

**What happens:**

1. New embedding is generated for the updated text
2. Old document is replaced with new data
3. ID remains the same
4. Changes are persisted to disk

**Why regenerate embedding?**

- The text changed, so the meaning might have changed
- New embedding ensures accurate similarity search

### 6. Deleting Documents (DELETE)

```python
def delete_document(doc_id: str):
    collection.delete(ids=[doc_id])
```

**What happens:**

1. Document is removed from the collection
2. Embedding is deleted
3. Metadata is removed
4. Space is reclaimed

### 7. Semantic Search (QUERY)

```python
def search_similar(query_text: str, n_results: int):
    # Generate query embedding
    query_embedding = create_embedding(query_text)

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results
```

**What happens:**

1. Query text is converted to an embedding
2. ChromaDB compares this embedding to all stored embeddings
3. Returns the N most similar documents
4. Similarity is measured by distance (lower = more similar)

**Distance Metric:**

- ChromaDB uses cosine distance by default
- Range: 0.0 (identical) to 2.0 (opposite)
- Formula: `distance = 1 - cosine_similarity`

**Example Search:**

```
Query: "artificial intelligence"
Results:
  1. "Machine learning is a branch of AI..." (distance: 0.15)
  2. "NLP enables computers to understand..."  (distance: 0.32)
  3. "Python is a programming language..."    (distance: 0.78)
```

## Vector Similarity Explained

### How Embeddings Work

Think of embeddings as coordinates in a high-dimensional space:

```
Simple 2D Example (real embeddings are 1536D):

"AI" = [0.8, 0.9]
"ML" = [0.7, 0.85]  ← Close to AI
"Python" = [0.2, 0.3]  ← Far from AI

Distance(AI, ML) = small → Similar!
Distance(AI, Python) = large → Different!
```

### Why This Works

- Similar meanings → Similar vectors
- Semantic relationships are preserved
- "king" - "man" + "woman" ≈ "queen"
- Context-aware understanding

## ChromaDB Under the Hood

### Data Storage

1. **SQLite Database** (`chroma.sqlite3`):
   - Stores metadata
   - Stores original documents
   - Manages IDs and collection info

2. **Vector Index** (UUID folder):
   - Stores embedding vectors
   - Uses HNSW (Hierarchical Navigable Small World) algorithm
   - Enables fast approximate nearest neighbor search

### Query Performance

- **Indexing**: O(log N) insertion time
- **Search**: O(log N) query time
- **Scalability**: Handles millions of vectors
- **Memory**: Loads index into RAM for speed

## Embedding Models

### text-embedding-3-small

- **Dimensions**: 1536
- **Speed**: Fast
- **Cost**: $0.02 / 1M tokens
- **Use case**: General purpose

### text-embedding-3-large

- **Dimensions**: 3072
- **Speed**: Slower
- **Cost**: $0.13 / 1M tokens
- **Use case**: Higher accuracy needed

## Memory and Cost Estimates

### For 1,000 Documents (avg 200 tokens each)

**Embedding Costs:**

- text-embedding-3-small: $0.004 (200k tokens)
- text-embedding-3-large: $0.026

**Storage:**

- Embeddings: ~6 MB (1536 dims × 1000 docs × 4 bytes)
- Documents: ~200 KB (raw text)
- Metadata: ~10 KB
- Total: ~7 MB

**Performance:**

- Add 1000 docs: ~30 seconds (API limited)
- Search query: <10ms (after index loaded)
- Retrieval by ID: <1ms

## Best Practices

### 1. Batch Operations

```python
# Instead of:
for doc in documents:
    add_document(doc)

# Do:
collection.add(
    ids=[doc.id for doc in documents],
    embeddings=[create_embedding(doc.text) for doc in documents],
    documents=[doc.text for doc in documents]
)
```

### 2. Error Handling

```python
try:
    add_document(doc_id, text)
except Exception as e:
    print(f"Failed to add document: {e}")
    # Handle retry or logging
```

### 3. Metadata Strategy

```python
metadata = {
    "category": "product",
    "timestamp": "2026-02-13",
    "source": "manual_entry",
    "version": 1
}
```

### 4. Collection Management

```python
# List all collections
collections = chroma_client.list_collections()

# Delete a collection
chroma_client.delete_collection(name="old_collection")

# Reset everything
chroma_client.reset()
```

## Common Patterns

### 1. Document Versioning

```python
add_document("doc-001-v1", text_v1, {"version": 1})
add_document("doc-001-v2", text_v2, {"version": 2})
```

### 2. Filtered Search

```python
results = collection.query(
    query_embeddings=[embedding],
    where={"category": "ai"},  # Metadata filter
    n_results=5
)
```

### 3. Batch Updates

```python
for doc_id, new_text in updates.items():
    update_document(doc_id, new_text)
```

## Troubleshooting

### Issue: Slow Performance

- **Cause**: Large collection not yet indexed
- **Solution**: Wait for indexing, or batch operations

### Issue: High Memory Usage

- **Cause**: Large embeddings loaded into RAM
- **Solution**: Use smaller batches, restart process

### Issue: Inaccurate Search Results

- **Cause**: Poor quality embeddings or metadata
- **Solution**: Use better embedding model, add filters

## Next Level: Production Considerations

1. **Error Recovery**: Retry logic for API failures
2. **Monitoring**: Track embedding costs and latency
3. **Backup**: Regular backups of `chroma_db/`
4. **Scaling**: Consider hosted ChromaDB or Pinecone
5. **Security**: Encrypt sensitive documents
6. **Caching**: Cache embeddings for repeated queries

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Vector Search Algorithms](https://www.pinecone.io/learn/vector-search/)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
