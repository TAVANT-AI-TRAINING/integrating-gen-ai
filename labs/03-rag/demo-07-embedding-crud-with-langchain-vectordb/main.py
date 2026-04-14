"""
Embedding CRUD with LangChain - Multi Vector Database Support

This demo shows LangChain's power: Switch between vector databases with configuration!

Supports:
- ChromaDB (local, development)
- Pinecone (cloud, production)

Same code works with both - just change VECTOR_DB in .env!

This is the power of LangChain abstractions! 🦜🔗
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ============================================================================
# STEP 1: LOAD CONFIGURATION
# ============================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_EMBEDDING_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# NEW: Choose vector database via configuration!
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb").lower()  # "chromadb" or "pinecone"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_policies")

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_EMBEDDING_KEY not found in environment variables")

print(f"✓ Selected Vector Database: {VECTOR_DB.upper()}")

# ============================================================================
# STEP 2: INITIALIZE EMBEDDINGS (Same for both databases!)
# ============================================================================

embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model=OPENAI_EMBEDDING_MODEL
)

print(f"✓ OpenAI embeddings initialized: {OPENAI_EMBEDDING_MODEL}")

# ============================================================================
# STEP 3: INITIALIZE VECTOR STORE (Based on configuration!)
# ============================================================================

vectorstore = None

if VECTOR_DB == "chromadb":
    # ChromaDB Configuration
    from langchain_chroma import Chroma
    
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"✓ ChromaDB initialized")
    print(f"  - Storage: {CHROMA_DB_DIR}")
    print(f"  - Collection: {COLLECTION_NAME}")

elif VECTOR_DB == "pinecone":
    # Pinecone Configuration
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-policies")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables (required for Pinecone)")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
    
    # Initialize LangChain Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    
    print(f"✓ Pinecone initialized")
    print(f"  - Index: {PINECONE_INDEX_NAME}")
    print(f"  - Cloud: {PINECONE_CLOUD}/{PINECONE_REGION}")

else:
    raise ValueError(f"Unsupported VECTOR_DB: {VECTOR_DB}. Use 'chromadb' or 'pinecone'")

print(f"✓ Vector store ready!")

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
    title="LangChain Multi-VectorDB CRUD API",
    description=f"API using LangChain with {VECTOR_DB.upper()} for CRUD operations",
    version="1.0.0"
)

# ============================================================================
# STEP 6: API ENDPOINTS - CRUD OPERATIONS
# ============================================================================
# 
# IMPORTANT: This code works with BOTH ChromaDB and Pinecone!
# LangChain provides a unified interface - same code, different backend!
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_db": VECTOR_DB,
        "collection": COLLECTION_NAME,
        "framework": "LangChain",
        "message": f"Running with {VECTOR_DB.upper()}!"
    }


@app.post("/documents", response_model=DocumentResponse)
def create_document(doc: DocumentCreate):
    """
    CREATE: Add a new document with embeddings.
    
    Works with both ChromaDB and Pinecone!
    LangChain handles the differences automatically.
    
    Example:
    {
        "doc_id": "policy-001",
        "text": "All employees must complete annual security training.",
        "metadata": {"category": "security", "date": "2024-01-15"}
    }
    """
    try:
        # Create a LangChain Document
        document = Document(
            page_content=doc.text,
            metadata={**(doc.metadata or {}), "doc_id": doc.doc_id}
        )
        
        # Add to vector store (works for both ChromaDB and Pinecone!)
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
    READ: Retrieve a document by its ID.
    
    Works with both ChromaDB and Pinecone!
    
    Example: GET /documents/policy-001
    """
    try:
        # Get from vector store (unified interface!)
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
    UPDATE: Update an existing document.
    
    Works with both ChromaDB and Pinecone!
    Strategy: Delete old version, add updated version with new embedding.
    
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
        
        # Delete old version (unified interface!)
        vectorstore.delete(ids=[doc_id])
        
        # Add updated version (unified interface!)
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
    DELETE: Remove a document.
    
    Works with both ChromaDB and Pinecone!
    
    Example: DELETE /documents/policy-001
    """
    try:
        # Check if document exists
        existing = vectorstore.get(ids=[doc_id])
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        # Delete from vector store (unified interface!)
        vectorstore.delete(ids=[doc_id])
        
        return {
            "message": f"Document '{doc_id}' deleted successfully",
            "doc_id": doc_id,
            "vector_db": VECTOR_DB
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# ============================================================================
# STEP 7: SEMANTIC SEARCH
# ============================================================================

@app.post("/query", response_model=List[QueryResult])
def query_similar_documents(query: QueryRequest):
    """
    QUERY: Find similar documents using semantic search.
    
    Works with both ChromaDB and Pinecone!
    LangChain provides the same API for both backends.
    
    Example:
    {
        "query_text": "What are the security requirements?",
        "n_results": 3
    }
    """
    try:
        # LangChain's unified semantic search API!
        # Works for both ChromaDB and Pinecone
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
    
    Note: Behavior differs between databases:
    - ChromaDB: Returns all documents
    - Pinecone: Limited support (use query instead)
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
            "documents": documents,
            "vector_db": VECTOR_DB,
            "note": "Full listing with ChromaDB, limited with Pinecone"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


# ============================================================================
# STEP 8: RUN THE SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
