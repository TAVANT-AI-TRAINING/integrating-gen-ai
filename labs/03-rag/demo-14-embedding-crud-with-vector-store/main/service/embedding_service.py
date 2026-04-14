"""
Embedding Service Module for RAG Solution with ChromaDB

This module provides a complete embedding service that integrates OpenAI embeddings
with ChromaDB vector storage using LangChain's Chroma implementation.

COMPLETE WORKFLOW (when imported by app.py or routes.py):
Step 1: Configuration & Initialization (runs automatically on import)
  - Load environment variables from .env file
  - Configure ChromaDB persistence directory
  - Validate all required environment variables
  - Initialize OpenAI Embeddings model (reusable instance)
  - Initialize universal Chroma instance (reused for all operations)

Step 2: Service-Layer Functions (PRIMARY API - called by routes.py)
  - embed_and_store_text() - Embed and store a single text document
  - get_embedding_by_id() - Retrieve embedding, metadata, and content by document ID
  - query_similar_documents() - Perform semantic similarity search with optional filters

Step 3: Core Database Operations (implementation details)
  - store_documents() - Batch store multiple documents with optional collection metadata
  - delete_by_doc_id() - Delete document by ID (also called directly from routes.py)

ARCHITECTURE:
- Universal vectorstore instance: A single Chroma instance is created at module load
  and reused for all embedding and search operations, ensuring consistency and efficiency.
- Persistent storage: ChromaDB stores embeddings locally in the configured directory
- Centralized configuration: All environment variables are validated at startup to ensure
  the service is properly configured before handling requests.
"""

import os
import sys
import logging
from uuid import uuid4
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: CONFIGURATION & INITIALIZATION
# ============================================================================
# This step runs automatically when this module is imported by app.py or routes.py
# It sets up ChromaDB and OpenAI connections by:
# - Loading environment variables from .env file
# - Configuring ChromaDB persistence directory
# - Validating all required variables
# - Initializing OpenAI Embeddings model
# - Creating a universal vector store instance (reused for all operations)
# 
# NOTE: This happens before the FastAPI app starts serving requests
# ============================================================================

# Load environment variables from .env file
try:
    load_dotenv()
    logger.info("✓ Environment variables loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load .env file: {e}")
    sys.exit(1)

# Extract ChromaDB configuration from environment variables
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_policies")

# Load OpenAI configuration from environment variables
OPENAI_API_EMBEDDING_KEY = os.getenv("OPENAI_API_EMBEDDING_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

# Validate all required environment variables together
required_vars = {
    # OpenAI variables
    "OPENAI_API_EMBEDDING_KEY": OPENAI_API_EMBEDDING_KEY,
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file."
    logger.error(f"✗ {error_msg}")
    raise ValueError(error_msg)

# Log ChromaDB configuration
logger.info(f"✓ ChromaDB persistence directory configured: {PERSIST_DIRECTORY}")
logger.info(f"✓ Collection name configured: {COLLECTION_NAME}")

# ============================================================================
# Initialize OpenAI Embeddings model
# ============================================================================
# This creates a reusable model instance that will be used for all embedding calls
# The model is initialized with:
# - API key for authentication
# - Model name (default: text-embedding-3-small)
# ============================================================================
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_EMBEDDING_KEY,
    model=OPENAI_MODEL
)
logger.info(f"✓ OpenAI embeddings model initialized successfully (model: {OPENAI_MODEL})")

# ============================================================================
# Initialize universal vector store instance
# ============================================================================
# This creates a single reusable Chroma instance for all operations
# It handles: embedding generation, vector storage, and similarity search
# The vectorstore is initialized with:
# - Embeddings model (for generating embeddings)
# - Collection name (for organizing documents)
# - Persist directory (for local storage)
# ============================================================================
vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
)
logger.info("✓ Universal vector store initialized successfully")


# ============================================================================
# STEP 2: SERVICE-LAYER FUNCTIONS (PRIMARY API)
# ============================================================================
# These functions are the main API called by routes.py to handle API requests
# They provide a clean interface between the API layer and database operations
# 
# PRIORITY: These are the primary functions that should be used by API endpoints
# ============================================================================

def embed_and_store_text(text: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Embed and store a single text document in the vector database.
    
    This function uses the universal vectorstore instance to generate embeddings
    and store documents. It automatically generates a UUID if no doc_id is provided.
    
    Execution flow:
    1. Generate or use provided document ID (UUID if not provided)
    2. Create Document object with text content and metadata
    3. Use universal vectorstore to generate embedding and store in database
    4. Return the document ID used for storage
    
    Args:
        text: The text content to embed and store
        doc_id: Optional document ID. If not provided, a UUID will be generated
        metadata: Optional dictionary of metadata to store with the document
    
    Returns:
        str: The document ID used for storage (provided or generated)
    
    Called from: routes.py POST /embed_and_store endpoint
    
    Example:
        >>> doc_id = embed_and_store_text(
        ...     "This is a sample document",
        ...     metadata={"source": "manual", "category": "policies"}
        ... )
    """
    # Generate document ID if not provided
    final_doc_id = doc_id or str(uuid4())
    md = dict(metadata or {})
    md["id"] = final_doc_id

    # Create Document object with text content and metadata
    doc = Document(page_content=text, metadata=md)

    # ============================================================================
    # Add document to vectorstore with explicit ID
    # ============================================================================
    # This step: generates embedding → stores in ChromaDB
    # IMPORTANT: We must explicitly pass the document ID to ChromaDB,
    # otherwise ChromaDB generates its own ID and our get/delete operations
    # won't work because they rely on our custom doc_id
    # ============================================================================
    vectorstore.add_documents([doc], ids=[final_doc_id])
    
    logger.info(f"✓ Stored document with ID: {final_doc_id}")
    return final_doc_id


def get_embedding_by_id(doc_id: str) -> Tuple[List[float], Dict[str, Any], str]:
    """Fetch embedding vector, metadata, and page_content for a given document ID.
    
    This function queries ChromaDB to retrieve document data including the embedding
    vector, metadata, and original text content.
    
    Execution flow:
    1. Query ChromaDB collection by document ID
    2. Extract embedding vector from the result
    3. Extract metadata
    4. Extract page_content (original text) from the document
    5. Return tuple of (embedding_vector, metadata, page_content)
    
    Args:
        doc_id: The document ID to search for (stored in metadata->id)
    
    Returns:
        Tuple containing:
        - List[float]: The embedding vector
        - Dict[str, Any]: The document metadata
        - str: The original page content (text)
    
    Raises:
        ValueError: If document with the given ID is not found
    
    Called from: routes.py POST /get_embedding endpoint
    """
    try:
        # Get the underlying ChromaDB collection
        collection = vectorstore._collection
        
        # Query ChromaDB by document ID (stored in metadata)
        results = collection.get(
            ids=[doc_id],
            include=["embeddings", "metadatas", "documents"]
        )
        
        # Check if document was found
        if len(results['ids']) == 0:
            raise ValueError(f"Document with ID '{doc_id}' not found")
        
        # Extract the components - check if arrays have elements (use 'is not None' to avoid array truth value error)
        embedding_vector = results['embeddings'][0] if results['embeddings'] is not None and len(results['embeddings']) > 0 else []
        metadata = results['metadatas'][0] if results['metadatas'] is not None and len(results['metadatas']) > 0 else {}
        page_content = results['documents'][0] if results['documents'] is not None and len(results['documents']) > 0 else ""
        
        return embedding_vector, metadata, page_content
        
    except ValueError:
        # Re-raise ValueError (document not found)
        raise
    except Exception as e:
        logger.error(f"✗ Error retrieving embedding for doc_id '{doc_id}': {e}")
        raise ValueError(f"Error retrieving document: {str(e)}")


def query_similar_documents(query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
    """Perform semantic similarity search to find documents similar to the query.
    
    This function uses the universal vectorstore instance to generate an embedding
    for the query text and search for similar documents using cosine similarity.
    Optional metadata filters can be applied to narrow down the search results.
    
    Execution flow:
    1. Use universal vectorstore instance
    2. Generate embedding for query text using OpenAI
    3. Perform cosine similarity search in ChromaDB vector database
    4. Apply metadata filters if provided (filters must be non-empty dict)
    5. Return top-k most similar documents with similarity scores
    
    Args:
        query: The search query text to find similar documents for
        k: Number of results to return (default: 5)
        filters: Optional metadata filters for filtering results.
                 Must be None or a non-empty dictionary. Empty dicts are treated as None.
                 Example: {"category": "policies", "source": "manual"}
        
    Returns:
        List[Tuple[Document, float]]: List of tuples containing:
        - Document: The matching document object with content and metadata
        - float: The similarity score (lower is more similar in cosine distance)
    
    Called from: routes.py POST /query endpoint
    
    Note:
        Empty filter dictionaries cause errors, so they are automatically
        treated as None (no filtering). Only non-empty dictionaries are applied.
    """
    try:
        # Only apply filters if they exist, are not None, and not empty
        # Empty dicts cause errors, so treat them as no filter
        if filters is not None and isinstance(filters, dict) and len(filters) > 0:
            
            # ============================================================================
            # PERFORM SIMILARITY SEARCH WITH METADATA FILTERS
            # ============================================================================
            # This step uses the universal 'vectorstore' instance to:
            # - Generate an embedding for the query text
            # - Search the database for similar document vectors (cosine similarity)
            # - Apply metadata filters to narrow down results
            # - Return top-k most similar documents with similarity scores
            # ============================================================================
            results = vectorstore.similarity_search_with_score(query, k=k, filter=filters)
            
            logger.info(f"✓ Query with filters returned {len(results)} results")
        else:
            # ============================================================================
            # PERFORM SIMILARITY SEARCH WITHOUT FILTERS
            # ============================================================================
            # This step uses the universal 'vectorstore' instance to:
            # - Generate an embedding for the query text
            # - Search the database for similar document vectors (cosine similarity)
            # - Return top-k most similar documents with similarity scores
            # ============================================================================
            results = vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"✓ Query without filters returned {len(results)} results")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Error querying similar documents: {e}")
        raise

# ============================================================================
# STEP 3: CORE DATABASE OPERATIONS (Implementation Details)
# ============================================================================
# These functions handle lower-level database operations
# - Batch document storage with collection metadata support
# - Direct document deletion by ID
# - Can be called by service-layer functions or directly from routes.py
# ============================================================================

def store_documents(documents: List[Document], collection_metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Store multiple documents in the vector database with optional collection metadata.
    
    This function handles batch document storage using the universal vectorstore instance.
    
    Execution flow:
    1. Validate documents list (handle empty list gracefully)
    2. Add all documents directly to existing collection using universal vectorstore
    3. Each document is embedded and stored in ChromaDB
    
    Args:
        documents: List of Document objects to store (must have page_content and metadata)
        collection_metadata: Optional metadata dictionary for the collection.
                            (Note: ChromaDB handles metadata differently than PgVector, 
                            this parameter is kept for API compatibility)
        
    Returns:
        bool: True if successful, False if an error occurred
    
    Called from: Can be used programmatically for batch document storage
    
    Note:
        This function is useful for bulk importing documents. 
        For single document storage, use embed_and_store_text().
    """
    try:
        # Handle empty document list
        if not documents:
            logger.info("✓ No documents to store (empty list)")
            return True
        
        # Extract IDs from document metadata or generate new ones
        doc_ids = []
        for doc in documents:
            if doc.metadata and "id" in doc.metadata:
                doc_ids.append(doc.metadata["id"])
            else:
                # Generate new ID if not provided
                new_id = str(uuid4())
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["id"] = new_id
                doc_ids.append(new_id)
        
        # Add documents to vectorstore with explicit IDs
        # This step: generates embeddings → stores in ChromaDB
        vectorstore.add_documents(documents, ids=doc_ids)
        logger.info(f"✓ Stored {len(documents)} documents using universal vectorstore")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to store documents: {e}")
        return False

def delete_by_doc_id(doc_id: str) -> bool:
    """Delete a document and its embedding from the vector database by document ID.
    
    This function deletes a document from ChromaDB using its ID.
    
    Execution flow:
    1. Get the underlying ChromaDB collection
    2. Delete document by ID
    3. Return True if successful, False otherwise
    
    Args:
        doc_id: The document ID to delete
    
    Returns:
        bool: True if document was successfully deleted, False if not found
    
    Called from: routes.py DELETE /delete_embedding endpoint
    """
    try:
        # Get the underlying ChromaDB collection
        collection = vectorstore._collection
        
        # Check if document exists first
        results = collection.get(ids=[doc_id])
        if len(results['ids']) == 0:
            logger.warning(f"✗ Document with ID '{doc_id}' not found")
            return False
        
        # Delete the document
        collection.delete(ids=[doc_id])
        logger.info(f"✓ Deleted document with ID: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error deleting document '{doc_id}': {e}")
        return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
# This section previously contained database connection helpers for PostgreSQL.
# ChromaDB uses local persistence and doesn't require connection management.
# ============================================================================
