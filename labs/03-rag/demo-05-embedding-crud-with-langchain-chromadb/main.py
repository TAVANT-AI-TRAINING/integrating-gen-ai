"""
Embedding CRUD with LangChain & ChromaDB - Simple Demo

This demo shows how LangChain simplifies working with embeddings and vector databases:
1. Use LangChain's OpenAIEmbeddings (no manual API calls)
2. Use LangChain's Chroma integration (high-level operations)
3. Work with Document objects (structured data)
4. Perform CRUD operations easily
5. Expose via FastAPI

LangChain makes it even simpler than direct ChromaDB usage!
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain imports - these make everything simpler!
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
# STEP 2: INITIALIZE LANGCHAIN COMPONENTS
# ============================================================================

# Initialize OpenAI Embeddings through LangChain
# LangChain handles all the API calls for us!
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_EMBEDDING_KEY,
    model=OPENAI_EMBEDDING_MODEL
)

# Initialize Chroma vector store through LangChain
# This gives us high-level operations: add, search, delete, etc.
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)

print(f"✓ LangChain initialized with ChromaDB: {CHROMA_DB_DIR}")
print(f"✓ Collection: {COLLECTION_NAME}")
print(f"✓ Embedding model: {OPENAI_EMBEDDING_MODEL}")

# ============================================================================
# STEP 3: FASTAPI MODELS
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
# STEP 4: CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="LangChain ChromaDB Embedding CRUD API",
    description="Simple API using LangChain for CRUD operations with ChromaDB",
    version="1.0.0"
)

# ============================================================================
# STEP 5: API ENDPOINTS - CRUD OPERATIONS
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "collection": COLLECTION_NAME,
        "framework": "LangChain + ChromaDB"
    }


@app.post("/documents", response_model=DocumentResponse)
def create_document(doc: DocumentCreate):
    """
    CREATE: Add a new document with embeddings using LangChain.
    
    LangChain automatically:
    - Generates embeddings
    - Stores in ChromaDB
    - Handles all the complexity
    
    Example:
    {
        "doc_id": "policy-001",
        "text": "All employees must complete annual security training.",
        "metadata": {"category": "security", "date": "2024-01-15"}
    }
    """
    try:
        # Create a LangChain Document object
        document = Document(
            page_content=doc.text,
            metadata={**(doc.metadata or {}), "doc_id": doc.doc_id}
        )
        
        # Add to vector store with specific ID
        # LangChain handles embedding generation automatically!
        vectorstore.add_documents(
            documents=[document],
            ids=[doc.doc_id]
        )
        
        return DocumentResponse(
            doc_id=doc.doc_id,
            text=doc.text,
            metadata=doc.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating document: {str(e)}")


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str):
    """
    READ: Retrieve a document by its ID using LangChain.
    
    Example: GET /documents/policy-001
    """
    try:
        # Use the underlying ChromaDB client to get by ID
        result = vectorstore.get(ids=[doc_id])
        
        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        # Extract metadata and remove doc_id if present
        metadata = result['metadatas'][0] if result['metadatas'] is not None and len(result['metadatas']) > 0 else {}
        if metadata and 'doc_id' in metadata:
            metadata = {k: v for k, v in metadata.items() if k != 'doc_id'}
        
        return DocumentResponse(
            doc_id=result['ids'][0],
            text=result['documents'][0],
            metadata=metadata if metadata else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(doc_id: str, doc: DocumentUpdate):
    """
    UPDATE: Update an existing document using LangChain.
    
    LangChain makes this easy:
    - Just delete and re-add with new content
    - Embeddings are regenerated automatically
    
    Example: PUT /documents/policy-001
    {
        "text": "Updated policy text",
        "metadata": {"category": "security", "updated": "2024-02-01"}
    }
    """
    try:
        # Check if document exists
        existing = vectorstore.get(ids=[doc_id])
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        # Delete old version
        vectorstore.delete(ids=[doc_id])
        
        # Add updated version
        document = Document(
            page_content=doc.text,
            metadata={**(doc.metadata or {}), "doc_id": doc_id}
        )
        
        vectorstore.add_documents(
            documents=[document],
            ids=[doc_id]
        )
        
        return DocumentResponse(
            doc_id=doc_id,
            text=doc.text,
            metadata=doc.metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    DELETE: Remove a document using LangChain.
    
    Example: DELETE /documents/policy-001
    """
    try:
        # Check if document exists
        existing = vectorstore.get(ids=[doc_id])
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        # Delete from vector store
        vectorstore.delete(ids=[doc_id])
        
        return {
            "message": f"Document '{doc_id}' deleted successfully",
            "doc_id": doc_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# ============================================================================
# STEP 6: ADDITIONAL OPERATIONS - SEMANTIC SEARCH
# ============================================================================

@app.post("/query", response_model=List[QueryResult])
def query_similar_documents(query: QueryRequest):
    """
    QUERY: Find similar documents using LangChain's semantic search.
    
    LangChain makes this incredibly simple:
    - Just call similarity_search_with_score()
    - It handles embedding generation and search automatically
    
    Example:
    {
        "query_text": "What are the security requirements?",
        "n_results": 3
    }
    """
    try:
        # LangChain's simple semantic search API
        # Returns documents with similarity scores
        results = vectorstore.similarity_search_with_score(
            query=query.query_text,
            k=query.n_results
        )
        
        # Format response
        formatted_results = []
        for doc, score in results:
            # Extract doc_id from metadata
            doc_id = doc.metadata.get('doc_id', 'unknown')
            
            # Clean metadata (remove doc_id)
            clean_metadata = {k: v for k, v in doc.metadata.items() if k != 'doc_id'}
            
            formatted_results.append(QueryResult(
                doc_id=doc_id,
                text=doc.page_content,
                metadata=clean_metadata if clean_metadata else None,
                score=float(score)
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
        # Get all documents
        result = vectorstore.get()
        
        documents = []
        for i, doc_id in enumerate(result['ids']):
            text = result['documents'][i] if result['documents'] is not None else ""
            metadata = result['metadatas'][i] if result['metadatas'] is not None and len(result['metadatas']) > i else {}
            
            # Clean metadata
            if metadata and 'doc_id' in metadata:
                metadata = {k: v for k, v in metadata.items() if k != 'doc_id'}
            
            documents.append({
                "doc_id": doc_id,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "metadata": metadata if metadata else None
            })
        
        return {
            "total_count": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


# ============================================================================
# STEP 7: RUN THE SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
