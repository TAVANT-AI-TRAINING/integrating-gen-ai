# Demo 03: Embedding with ChromaDB

A simple console application demonstrating ChromaDB integration with OpenAI embeddings. This demo shows all CRUD operations without the complexity of a web server.

## 🎯 What You'll Learn

- Initialize ChromaDB with persistent storage
- Generate embeddings using OpenAI's API
- Perform CRUD operations (Create, Read, Update, Delete)
- Conduct semantic similarity searches
- Work with document metadata

## 📦 What's Inside

Everything is in a **single `main.py` file** for simplicity:

1. **ChromaDB Setup** - Persistent vector database configuration
2. **OpenAI Embeddings** - Text-to-vector conversion
3. **CRUD Operations**:
   - **CREATE**: Add documents with embeddings
   - **READ**: Retrieve documents by ID
   - **UPDATE**: Modify existing documents
   - **DELETE**: Remove documents
4. **Semantic Search** - Find similar documents using vector similarity
5. **Console Output** - Easy-to-follow demo flow

## 🚀 Quick Start

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

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_DB_DIR=./chroma_db
```

### 3. Run the Demo

```bash
uv run python main.py
```

## 📖 What the Demo Does

The demo runs through a complete workflow:

### 1. **CREATE** - Adding Documents

- Adds 4 sample documents about Python, ML, ChromaDB, and NLP
- Generates embeddings for each document
- Stores them in ChromaDB with metadata

### 2. **READ** - Retrieving Documents

- Fetches a document by ID
- Displays the text, metadata, and embedding info

### 3. **UPDATE** - Modifying Documents

- Updates a document with new text
- Regenerates the embedding
- Updates metadata

### 4. **QUERY** - Semantic Search

- Searches for documents similar to a query
- Returns top N most similar documents
- Shows similarity distances

### 5. **DELETE** - Removing Documents

- Deletes a document from the collection
- Verifies the deletion

## 🔧 Key Functions

### `create_embedding(text: str) -> List[float]`

Generate an embedding vector for the given text using OpenAI.

### `add_document(doc_id, text, metadata)`

Add a new document to ChromaDB with its embedding.

### `get_document(doc_id) -> Dict`

Retrieve a document by its ID.

### `update_document(doc_id, new_text, new_metadata)`

Update an existing document and regenerate its embedding.

### `delete_document(doc_id)`

Remove a document from the collection.

### `search_similar(query_text, n_results) -> List[Dict]`

Find the most similar documents to a query using semantic search.

## 📊 Sample Output

```
======================================================================
ChromaDB Embedding Demo
======================================================================
✓ ChromaDB initialized: ./chroma_db
✓ Collection: demo_documents
✓ Documents in collection: 0

======================================================================
DEMO 1: CREATE - Adding Documents
======================================================================

Adding document: doc-001
✓ Document added successfully (embedding dim: 1536)

...

======================================================================
DEMO 4: QUERY - Semantic Search
======================================================================

Searching for: 'Tell me about artificial intelligence and deep learning'
Top 3 results:

  1. ID: doc-002
     Text: Machine learning is a branch of artificial intelligence...
     Distance: 0.2456
     Metadata: {'category': 'ai', 'topic': 'machine_learning'}
```

## 🎓 Learning Points

### ChromaDB Basics

- **Persistent Storage**: Data is saved to disk and survives restarts
- **Collections**: Organize related documents together
- **Metadata**: Attach custom data to each document

### Embedding Operations

- **Automatic Vectorization**: Text is converted to embeddings
- **Semantic Understanding**: Similar meanings have similar vectors
- **Distance Metrics**: Lower distance = more similar

### CRUD Pattern

- **Scalable**: Works with thousands of documents
- **Fast Queries**: Vector similarity is efficient
- **Flexible Metadata**: Add any custom fields you need

## 🔄 Persistence

ChromaDB stores data persistently in the `./chroma_db` directory:

- Run the script multiple times - data persists
- The collection accumulates documents over runs
- Delete the `chroma_db` folder to start fresh

## 🧪 Experiment Ideas

1. **Add Your Own Documents**: Modify the sample_docs list
2. **Try Different Queries**: Change the search queries
3. **Adjust n_results**: Get more or fewer search results
4. **Custom Metadata**: Add your own metadata fields
5. **Different Embedding Models**: Try `text-embedding-3-large`

## 📝 Code Structure

```python
# Configuration
- Load environment variables
- Set up API keys and settings

# Initialize Clients
- OpenAI client for embeddings
- ChromaDB client for storage

# Helper Functions
- create_embedding()
- add_document()
- get_document()
- update_document()
- delete_document()
- search_similar()

# Main Demo
- Runs through all operations with sample data
```

## 🚨 Common Issues

### "OPENAI_API_KEY not found"

- Make sure you copied `.env.example` to `.env`
- Add your actual OpenAI API key

### "Collection already exists" warnings

- This is normal - ChromaDB reuses existing collections
- Delete `chroma_db/` folder to start fresh

### Out of memory

- Reduce the number of sample documents
- Use a smaller embedding model

## 🎯 Next Steps

After mastering this demo, you can:

1. **Add More Documents**: Scale to larger document sets
2. **Build a UI**: Connect to Streamlit or FastAPI
3. **Filter Searches**: Use metadata filters in queries
4. **Multiple Collections**: Organize different document types
5. **Batch Operations**: Add/update many documents at once

## 📚 Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)

## 🤝 Need Help?

- Check the console output for detailed error messages
- Review the comments in `main.py`
- Verify your OpenAI API key is valid
- Ensure you have enough OpenAI credits

---

**Happy Learning! 🚀**
