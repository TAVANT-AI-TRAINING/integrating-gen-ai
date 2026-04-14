"""
Embedding CRUD with ChromaDB - Simple Demo

This demo shows how to:
1. Create embeddings using OpenAI
2. Store embeddings in ChromaDB
3. Read/retrieve embeddings and documents
4. Update existing documents
5. Delete documents
6. Expose all operations via FastAPI

Simple all-in-one file for easy learning!
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ============================================================================
# STEP 1: LOAD CONFIGURATION
# ============================================================================
load_dotenv()

OPENAI_API_EMBEDDING_KEY = os.getenv("OPENAI_API_EMBEDDING_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "company_policies"

if not OPENAI_API_EMBEDDING_KEY:
    raise ValueError("OPENAI_API_EMBEDDING_KEY not found in environment variables")

# ============================================================================
# STEP 2: INITIALIZE CLIENTS
# ============================================================================

# Initialize OpenAI client for embeddings
openai_client = OpenAI(api_key=OPENAI_API_EMBEDDING_KEY)

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Company policies and documents"}
)

print(f"✓ ChromaDB initialized: {CHROMA_DB_DIR}")
print(f"✓ Collection: {COLLECTION_NAME}")
print(f"✓ Documents in collection: {collection.count()}")

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
    distance: float

# ============================================================================
# STEP 5: CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="ChromaDB Embedding CRUD API",
    description="Simple API for CRUD operations with ChromaDB and OpenAI embeddings",
    version="1.0.0"
)

# ============================================================================
# STEP 6: API ENDPOINTS - CRUD OPERATIONS
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "collection": COLLECTION_NAME,
        "total_documents": collection.count()
    }


@app.post("/documents", response_model=DocumentResponse)
def create_document(doc: DocumentCreate):
    """
    CREATE: Add a new document with embeddings to ChromaDB.
    
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
        
        # Store in ChromaDB
        collection.add(
            ids=[doc.doc_id],
            embeddings=[embedding],
            documents=[doc.text],
            metadatas=[doc.metadata] if doc.metadata else None
        )
        
        return DocumentResponse(
            doc_id=doc.doc_id,
            text=doc.text,
            metadata=doc.metadata,
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
        result = collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if len(result['ids']) == 0:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        return DocumentResponse(
            doc_id=result['ids'][0],
            text=result['documents'][0],
            metadata=result['metadatas'][0] if result['metadatas'] is not None and len(result['metadatas']) > 0 else None,
            embedding_length=len(result['embeddings'][0]) if result['embeddings'] is not None and len(result['embeddings']) > 0 else 0
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(doc_id: str, doc: DocumentUpdate):
    """
    UPDATE: Update an existing document's text and/or metadata.
    
    Example: PUT /documents/policy-001
    {
        "text": "Updated policy text",
        "metadata": {"category": "security", "updated": "2024-02-01"}
    }
    """
    try:
        # Check if document exists
        existing = collection.get(ids=[doc_id])
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        # Generate new embedding for updated text
        embedding = create_embedding(doc.text)
        
        # Update in ChromaDB
        collection.update(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc.text],
            metadatas=[doc.metadata] if doc.metadata else None
        )
        
        return DocumentResponse(
            doc_id=doc_id,
            text=doc.text,
            metadata=doc.metadata,
            embedding_length=len(embedding)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    DELETE: Remove a document from ChromaDB.
    
    Example: DELETE /documents/policy-001
    """
    try:
        # Check if document exists
        existing = collection.get(ids=[doc_id])
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        # Delete from ChromaDB
        collection.delete(ids=[doc_id])
        
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
    
    Example:
    {
        "query_text": "What are the security requirements?",
        "n_results": 3
    }
    """
    try:
        # Generate embedding for query
        query_embedding = create_embedding(query.query_text)
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format response
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append(QueryResult(
                doc_id=results['ids'][0][i],
                text=results['documents'][0][i],
                metadata=results['metadatas'][0][i] if results['metadatas'] is not None and len(results['metadatas']) > 0 else None,
                distance=results['distances'][0][i]
            ))
        
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.get("/documents")
def list_all_documents():
    """
    LIST: Get all documents in the collection.
    """
    try:
        # Get all documents (limit to prevent overwhelming response)
        result = collection.get(
            include=["documents", "metadatas"],
            limit=100
        )
        
        documents = []
        for i in range(len(result['ids'])):
            documents.append({
                "doc_id": result['ids'][i],
                "text": result['documents'][i][:100] + "..." if len(result['documents'][i]) > 100 else result['documents'][i],
                "metadata": result['metadatas'][i] if result['metadatas'] is not None and len(result['metadatas']) > 0 else None
            })
        
        return {
            "total_count": collection.count(),
            "returned_count": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


# ============================================================================
# STEP 8: RUN THE SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
