"""
Demo 03: Embedding with ChromaDB - Simple Console Application

This demo shows how to:
1. Initialize ChromaDB with persistent storage
2. Create embeddings using OpenAI
3. Add documents to ChromaDB (CREATE)
4. Retrieve documents by ID (READ)
5. Update existing documents (UPDATE)
6. Delete documents (DELETE)
7. Perform similarity search (QUERY)

All in a simple console application - no web server needed!
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "demo_documents"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# ============================================================================
# INITIALIZE CLIENTS
# ============================================================================

# Initialize OpenAI client for embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Demo documents collection"}
)

print("=" * 70)
print("ChromaDB Embedding Demo")
print("=" * 70)
print(f"✓ ChromaDB initialized: {CHROMA_DB_DIR}")
print(f"✓ Collection: {COLLECTION_NAME}")
print(f"✓ Documents in collection: {collection.count()}")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def add_document(doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
    """Add a document to ChromaDB with its embedding."""
    print(f"Adding document: {doc_id}")
    
    # Generate embedding
    embedding = create_embedding(text)
    
    # Add to ChromaDB
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata or {}]
    )
    
    print(f"✓ Document added successfully (embedding dim: {len(embedding)})")
    print()


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a document by its ID."""
    print(f"Retrieving document: {doc_id}")
    
    result = collection.get(
        ids=[doc_id],
        include=["documents", "metadatas", "embeddings"]
    )
    
    if result["ids"]:
        doc = {
            "id": result["ids"][0],
            "document": result["documents"][0],
            "metadata": result["metadatas"][0],
            "embedding_length": len(result["embeddings"][0])
        }
        print(f"✓ Document found")
        print(f"  Text: {doc['document'][:100]}...")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Embedding length: {doc['embedding_length']}")
        print()
        return doc
    else:
        print(f"✗ Document not found")
        print()
        return None


def update_document(doc_id: str, new_text: str, new_metadata: Optional[Dict[str, Any]] = None):
    """Update an existing document."""
    print(f"Updating document: {doc_id}")
    
    # Generate new embedding
    embedding = create_embedding(new_text)
    
    # Update in ChromaDB
    collection.update(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[new_text],
        metadatas=[new_metadata or {}]
    )
    
    print(f"✓ Document updated successfully")
    print()


def delete_document(doc_id: str):
    """Delete a document from ChromaDB."""
    print(f"Deleting document: {doc_id}")
    
    collection.delete(ids=[doc_id])
    
    print(f"✓ Document deleted successfully")
    print()


def search_similar(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Search for similar documents using semantic similarity."""
    print(f"Searching for: '{query_text}'")
    print(f"Top {n_results} results:")
    print()
    
    # Generate query embedding
    query_embedding = create_embedding(query_text)
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    search_results = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            result = {
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            search_results.append(result)
            
            print(f"  {i+1}. ID: {result['id']}")
            print(f"     Text: {result['document'][:80]}...")
            print(f"     Distance: {result['distance']:.4f}")
            print(f"     Metadata: {result['metadata']}")
            print()
    else:
        print("  No results found")
        print()
    
    return search_results


def get_collection_stats():
    """Display collection statistics."""
    count = collection.count()
    print(f"Collection Statistics:")
    print(f"  Total documents: {count}")
    print()


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run the demo showing all CRUD operations."""
    
    # Sample documents about different topics
    sample_docs = [
        {
            "id": "doc-001",
            "text": "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and AI.",
            "metadata": {"category": "programming", "topic": "python"}
        },
        {
            "id": "doc-002",
            "text": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data. It powers recommendation systems and image recognition.",
            "metadata": {"category": "ai", "topic": "machine_learning"}
        },
        {
            "id": "doc-003",
            "text": "ChromaDB is an open-source embedding database that makes it easy to build AI applications with embeddings. It provides simple APIs for storing and querying vectors.",
            "metadata": {"category": "database", "topic": "vector_db"}
        },
        {
            "id": "doc-004",
            "text": "Natural language processing (NLP) enables computers to understand and generate human language. It's used in chatbots, translation, and sentiment analysis.",
            "metadata": {"category": "ai", "topic": "nlp"}
        }
    ]
    
    # ========================================================================
    # DEMO 1: CREATE - Add documents
    # ========================================================================
    print("=" * 70)
    print("DEMO 1: CREATE - Adding Documents")
    print("=" * 70)
    print()
    
    for doc in sample_docs:
        add_document(doc["id"], doc["text"], doc["metadata"])
    
    get_collection_stats()
    
    # ========================================================================
    # DEMO 2: READ - Retrieve a document
    # ========================================================================
    print("=" * 70)
    print("DEMO 2: READ - Retrieving Document")
    print("=" * 70)
    print()
    
    get_document("doc-001")
    
    # ========================================================================
    # DEMO 3: UPDATE - Modify a document
    # ========================================================================
    print("=" * 70)
    print("DEMO 3: UPDATE - Updating Document")
    print("=" * 70)
    print()
    
    update_document(
        "doc-001",
        "Python is a versatile programming language with extensive libraries for data science, web development, and automation. It's beginner-friendly and powers many AI applications.",
        {"category": "programming", "topic": "python", "updated": True}
    )
    
    # Verify the update
    get_document("doc-001")
    
    # ========================================================================
    # DEMO 4: QUERY - Semantic similarity search
    # ========================================================================
    print("=" * 70)
    print("DEMO 4: QUERY - Semantic Search")
    print("=" * 70)
    print()
    
    # Search for AI-related content
    search_similar("Tell me about artificial intelligence and deep learning", n_results=3)
    
    # Search for database-related content
    search_similar("What are vector databases?", n_results=2)
    
    # ========================================================================
    # DEMO 5: DELETE - Remove a document
    # ========================================================================
    print("=" * 70)
    print("DEMO 5: DELETE - Removing Document")
    print("=" * 70)
    print()
    
    delete_document("doc-004")
    
    # Verify deletion
    get_document("doc-004")
    
    get_collection_stats()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("You've learned how to:")
    print("  ✓ Initialize ChromaDB with persistent storage")
    print("  ✓ Create embeddings using OpenAI")
    print("  ✓ Add documents with embeddings (CREATE)")
    print("  ✓ Retrieve documents by ID (READ)")
    print("  ✓ Update existing documents (UPDATE)")
    print("  ✓ Delete documents (DELETE)")
    print("  ✓ Perform semantic similarity search (QUERY)")
    print()
    print(f"ChromaDB data persisted to: {CHROMA_DB_DIR}")
    print()


if __name__ == "__main__":
    main()
