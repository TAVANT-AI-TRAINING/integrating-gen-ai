"""
Embedding CRUD with Pinecone - Simple Demo

This demo shows how to:
1. Create embeddings using OpenAI
2. Store embeddings in Pinecone (cloud vector database)
3. Read/retrieve embeddings and documents
4. Update existing documents
5. Delete documents
6. Expose all operations via FastAPI

Simple all-in-one file for easy learning!
Pinecone = Managed, scalable cloud vector database
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ============================================================================
# STEP 1: LOAD CONFIGURATION
# ============================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_EMBEDDING_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-policies")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_EMBEDDING_KEY not found in environment variables")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# ============================================================================
# STEP 2: INITIALIZE CLIENTS
# ============================================================================

# Initialize OpenAI client for embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Get embedding dimension (text-embedding-3-small = 1536 dimensions)
EMBEDDING_DIMENSION = 1536

# Create or get index
# Pinecone stores indexes in the cloud (persistent by default)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",  # Cosine similarity for semantic search
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )
    print(f"✓ Index created: {PINECONE_INDEX_NAME}")
else:
    print(f"✓ Using existing index: {PINECONE_INDEX_NAME}")

# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

print(f"✓ Pinecone initialized")
print(f"✓ Index: {PINECONE_INDEX_NAME}")
print(f"✓ Index stats: {index.describe_index_stats()}")

# ============================================================================
# STEP 3: HELPER FUNCTIONS
# ============================================================================

def create_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# ============================================================================
# STEP 4: FASTAPI MODELS
# ============================================================================

class DocumentCreate(BaseModel):
    """Model for creating a new document."""
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentUpdate(BaseModel):
    """Model for updating an existing document."""
    text: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    """Model for document response."""
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    embedding_length: int

class QueryRequest(BaseModel):
    """Model for similarity search query."""
    query_text: str
    n_results: int = 5

class QueryResult(BaseModel):
    """Model for query result item."""
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    score: float

# ============================================================================
# STEP 5: CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Pinecone Embedding CRUD API",
    description="Simple API for CRUD operations with Pinecone and OpenAI embeddings",
    version="1.0.0"
)

# ============================================================================
# STEP 6: API ENDPOINTS - CRUD OPERATIONS
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint."""
    stats = index.describe_index_stats()
    return {
        "status": "healthy",
        "index": PINECONE_INDEX_NAME,
        "vector_count": stats.get('total_vector_count', 0),
        "cloud": f"{PINECONE_CLOUD}/{PINECONE_REGION}"
    }


@app.post("/documents", response_model=DocumentResponse)
def create_document(doc: DocumentCreate):
    """
    CREATE: Add a new document with embeddings to Pinecone.
    
    Pinecone stores:
    - Vector (embedding)
    - Metadata (including the original text)
    - ID (document identifier)
    
    Example:
    {
        "doc_id": "policy-001",
        "text": "All employees must complete annual security training.",
        "metadata": {"category": "security", "date": "2024-01-15"}
    }
    """
    try:
        # Generate embedding
        embedding = create_embedding(doc.text)
        
        # Prepare metadata (Pinecone stores text in metadata)
        metadata = doc.metadata or {}
        metadata["text"] = doc.text  # Store original text in metadata
        
        # Upsert to Pinecone (upsert = insert or update)
        index.upsert(
            vectors=[
                {
                    "id": doc.doc_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        return DocumentResponse(
            doc_id=doc.doc_id,
            text=doc.text,
            metadata={k: v for k, v in metadata.items() if k != "text"},
            embedding_length=len(embedding)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating document: {str(e)}")


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str):
    """
    READ: Retrieve a document by its ID.
    
    Example: GET /documents/policy-001
    """
    try:
        # Fetch from Pinecone
        result = index.fetch(ids=[doc_id])
        
        if doc_id not in result.get('vectors', {}):
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        vector_data = result['vectors'][doc_id]
        metadata = vector_data.get('metadata', {})
        text = metadata.get('text', '')
        
        # Clean metadata (remove text field)
        clean_metadata = {k: v for k, v in metadata.items() if k != 'text'}
        
        return DocumentResponse(
            doc_id=doc_id,
            text=text,
            metadata=clean_metadata if clean_metadata else None,
            embedding_length=len(vector_data.get('values', []))
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(doc_id: str, doc: DocumentUpdate):
    """
    UPDATE: Update an existing document's text and/or metadata.
    
    Pinecone's upsert handles updates automatically!
    
    Example: PUT /documents/policy-001
    {
        "text": "Updated policy text",
        "metadata": {"category": "security", "updated": "2024-02-01"}
    }
    """
    try:
        # Check if document exists
        result = index.fetch(ids=[doc_id])
        if doc_id not in result.get('vectors', {}):
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        # Generate new embedding for updated text
        embedding = create_embedding(doc.text)
        
        # Prepare metadata
        metadata = doc.metadata or {}
        metadata["text"] = doc.text
        
        # Upsert (update) in Pinecone
        index.upsert(
            vectors=[
                {
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        return DocumentResponse(
            doc_id=doc_id,
            text=doc.text,
            metadata={k: v for k, v in metadata.items() if k != "text"},
            embedding_length=len(embedding)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    DELETE: Remove a document from Pinecone.
    
    Example: DELETE /documents/policy-001
    """
    try:
        # Check if document exists
        result = index.fetch(ids=[doc_id])
        if doc_id not in result.get('vectors', {}):
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        # Delete from Pinecone
        index.delete(ids=[doc_id])
        
        return {
            "message": f"Document '{doc_id}' deleted successfully",
            "doc_id": doc_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# ============================================================================
# STEP 7: ADDITIONAL OPERATIONS
# ============================================================================

@app.post("/query", response_model=List[QueryResult])
def query_similar_documents(query: QueryRequest):
    """
    QUERY: Find similar documents using semantic search.
    
    Pinecone's query API:
    - Fast vector similarity search
    - Returns top-k most similar vectors
    - Includes similarity scores
    
    Example:
    {
        "query_text": "What are the security requirements?",
        "n_results": 3
    }
    """
    try:
        # Generate embedding for query
        query_embedding = create_embedding(query.query_text)
        
        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=query.n_results,
            include_metadata=True
        )
        
        # Format response
        formatted_results = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            clean_metadata = {k: v for k, v in metadata.items() if k != 'text'}
            
            formatted_results.append(QueryResult(
                doc_id=match['id'],
                text=text,
                metadata=clean_metadata if clean_metadata else None,
                score=match['score']  # Pinecone returns similarity score (higher = more similar)
            ))
        
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.get("/documents")
def list_all_documents():
    """
    LIST: Get all documents in the index.
    
    Note: Pinecone doesn't have a native "list all" operation.
    This is a limitation - you'd typically maintain a separate database
    for document listings in production.
    """
    try:
        stats = index.describe_index_stats()
        total_count = stats.get('total_vector_count', 0)
        
        return {
            "message": "Pinecone doesn't support listing all vectors directly",
            "total_count": total_count,
            "suggestion": "Use query with filters or maintain a separate document index",
            "note": "This is a known limitation of Pinecone's architecture"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


# ============================================================================
# STEP 8: RUN THE SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
