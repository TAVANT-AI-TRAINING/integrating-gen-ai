"""
FastAPI routes for embedding and document management.
This module contains all API endpoints for the embedding service.

COMPLETE WORKFLOW (when API request is received):
Step 1: Configuration & Initialization (runs automatically when embedding_service.py is imported)
  → Sets up database and Azure OpenAI connections

Step 2: Request Validation (FastAPI automatically validates using Pydantic models)
  → Validates request body structure and data types

Step 3: Call Service Function (calls functions from embedding_service.py)
  → embed_and_store_text() - Store single document
  → store_documents() - Store multiple documents (batch)
  → get_embedding_by_id() - Retrieve embedding by document ID
  → delete_by_doc_id() - Delete document
  → query_similar_documents() - Semantic search

Step 4: Return Response (formatted JSON response with proper status codes)
  → Success responses with data
  → Error responses with appropriate HTTP status codes
"""

import logging
from uuid import uuid4
from fastapi import APIRouter, status, Query
from langchain_core.documents import Document

from main.service.embedding_service import (
    embed_and_store_text,
    store_documents,
    get_embedding_by_id,
    delete_by_doc_id,
    query_similar_documents,
)
from main.models.models import (
    DocumentRequest,
    BatchDocumentRequest,
    DeleteEmbeddingRequest,
    QueryRequest,
    StoreResponse,
    BatchStoreResponse,
    EmbeddingResponse,
    DeleteResponse,
    QueryHit,
    QueryResponse,
    HealthResponse,
)

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 2: REQUEST VALIDATION
# ============================================================================
# FastAPI automatically validates incoming requests against Pydantic models
# All models are imported from main.models.models
# ============================================================================

# ============================================================================
# STEP 3: API ENDPOINTS
# ============================================================================
# Each endpoint follows this flow:
# 1. FastAPI validates request using Pydantic models (Step 2)
# 2. Endpoint calls service function from embedding_service.py (Step 3)
# 3. Service function handles database operations
# 4. Endpoint formats and returns response (Step 4)
# ============================================================================

# Create router for API endpoints
router = APIRouter()

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    description="Health check endpoint to verify the API is running",
    response_description="Returns the health status of the API"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Status and message indicating the API is healthy
    """
    return HealthResponse(
        status="healthy",
        message="Embedding API is running successfully"
    )

@router.post(
    "/embed_and_store",
    response_model=StoreResponse,
    status_code=status.HTTP_201_CREATED,
    description="Accepts a text document, generates an embedding using OpenAI, and stores it in the ChromaDB database.",
    response_description="Returns the document ID (provided or auto-generated) and a success message."
)
async def embed_and_store(request: DocumentRequest):
    """
    Embed and store a document in the vector database.
    
    Execution flow:
    - FastAPI validates request body (automatic via Pydantic)
    - Call embed_and_store_text() from embedding_service.py
    - Service function: generates embedding → stores in database
    - Return success response with document ID
    """
    # Call service function to embed and store document
    # This calls embed_and_store_text() which handles: ID generation → embedding → storage
    doc_id = embed_and_store_text(
        text=request.text,
        doc_id=request.doc_id,
        metadata=request.metadata,
    )
    logger.info(f"✓ Stored document with ID: {doc_id}")
    return StoreResponse(message="Document embedded and stored successfully", doc_id=doc_id)

@router.post(
    "/embed_and_store_batch",
    response_model=BatchStoreResponse,
    status_code=status.HTTP_201_CREATED,
    description="Accepts multiple text documents, generates embeddings using Azure OpenAI, and stores them in the PGVector database in a single batch operation.",
    response_description="Returns the number of documents stored and a success message."
)
async def embed_and_store_batch(request: BatchDocumentRequest):
    """
    Embed and store multiple documents in the vector database (batch operation).
    
    Execution flow:
    - FastAPI validates request body (automatic via Pydantic)
    - Convert request documents to LangChain Document objects
    - Call store_documents() from embedding_service.py
    - Service function: generates embeddings → stores in database (batch operation)
    - Return success response with document count
    
    This endpoint is more efficient than calling /embed_and_store multiple times
    as it processes all documents in a single batch operation.
    """
    # Convert request documents to LangChain Document objects
    documents = []
    for doc_request in request.documents:
        # Generate document ID if not provided
        final_doc_id = doc_request.doc_id or str(uuid4())
        md = dict(doc_request.metadata or {})
        md["id"] = final_doc_id
        
        # Create Document object
        doc = Document(page_content=doc_request.text, metadata=md)
        documents.append(doc)
    
    # Call service function to store documents in batch
    # This calls store_documents() which handles: batch embedding → batch storage
    success = store_documents(
        documents=documents,
        collection_metadata=request.collection_metadata
    )
    
    if not success:
        raise ValueError("Failed to store documents in batch")
    
    count = len(documents)
    logger.info(f"✓ Stored {count} documents in batch")
    return BatchStoreResponse(
        message=f"Successfully stored {count} documents",
        count=count
    )

@router.delete(
    "/delete_embedding",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    description="Deletes a document and its embedding from the database by document ID.",
    response_description="Returns a success message and the deleted document ID."
)
async def delete_embedding(request: DeleteEmbeddingRequest):
    """
    Delete a document and its embedding from the database.
    
    Execution flow:
    - FastAPI validates request body (automatic via Pydantic)
    - Call delete_by_doc_id() from embedding_service.py
    - Function: deletes document from database
    - Return success response or 404 if not found
    """
    doc_id = request.doc_id
    # Call function to delete document
    # This calls delete_by_doc_id() which handles: database deletion
    deleted = delete_by_doc_id(doc_id)
    if not deleted:
        raise ValueError(f"Document with ID '{doc_id}' not found")
    logger.info(f"✓ Deleted document with ID: {doc_id}")
    return DeleteResponse(message=f"Document with ID '{doc_id}' deleted successfully", doc_id=doc_id)

@router.get(
    "/get_embedding",
    response_model=EmbeddingResponse,
    description="Retrieves the vector embedding, metadata, and original content for a given document ID.",
    response_description="Returns the document ID, embedding vector, metadata, and original page content."
)
async def get_embedding(
    doc_id: str = Query(
        ...,
        description="The document ID to retrieve",
        example="compliance_guide_001"
    )
):
    """
    Retrieve an embedding and document by document ID.
    
    Execution flow:
    - FastAPI validates query parameter (automatic via Query)
    - Call get_embedding_by_id() from embedding_service.py
    - Service function: fetches embedding and metadata from database
    - Return response with embedding vector, metadata, and content
    """
    # Call service function to retrieve embedding
    # This calls get_embedding_by_id() which handles: database query → parse data → return
    embedding_vector, metadata, page_content = get_embedding_by_id(doc_id)
    logger.info(f"✓ Retrieved embedding for document ID: {doc_id}")
    return EmbeddingResponse(
        doc_id=doc_id,
        embedding=embedding_vector,
        metadata=metadata,
        page_content=page_content,
    )

@router.post(
    "/query",
    response_model=QueryResponse,
    description="Performs semantic similarity search to find documents similar to the query text.",
    response_description="Returns a list of similar documents with their similarity scores, ordered by relevance."
)
async def query_documents(request: QueryRequest):
    """
    Query similar documents using semantic similarity search.
    
    Execution flow:
    - FastAPI validates request body (automatic via Pydantic)
    - Normalize filters (convert empty dicts to None)
    - Call query_similar_documents() from embedding_service.py
    - Service function: generates query embedding → searches database → returns top-k
    - Format results into response model
    - Return response with similarity scores
    """
    # Normalize filters: convert empty dicts to None to avoid PGVector errors
    filters = request.filters
    if filters is not None and isinstance(filters, dict) and len(filters) == 0:
        filters = None
    
    # Call service function to perform similarity search
    # This calls query_similar_documents() which handles: query embedding → database search → returns results
    results = query_similar_documents(
        query=request.query,
        k=request.top_k,
        filters=filters
    )
    
    # Convert results to QueryHit format for response
    query_hits = []
    for doc, score in results:
        # Extract doc_id from metadata
        doc_id = doc.metadata.get("id") if doc.metadata else None
        
        query_hits.append(QueryHit(
            doc_id=doc_id,
            score=float(score),
            metadata=doc.metadata,
            page_content=doc.page_content
        ))
    
    logger.info(f"✓ Query '{request.query}' returned {len(query_hits)} results")
    return QueryResponse(
        query=request.query,
        results=query_hits
    )