"""
Embedding Service
Database configuration and utilities for embedding and vector storage.
This module handles database connections, embeddings initialization, and core database operations.

NOTE: Similarity search functionality will be added in the next sprint.
Currently, this service handles document storage with embeddings, but search capabilities are not yet available.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document


# Optional imports used in helpers
try:
    import psycopg
except Exception:
    psycopg = None  # Will error at call time if missing


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Load environment variables
try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Failed to load .env file: {e}")
    sys.exit(1)


# Database configuration
try:
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "rag_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_policies")

    # Validate required environment variables
    if not DB_USER or not DB_PASSWORD or not DB_HOST or not DB_NAME:
        raise ValueError("Missing required database environment variables")

    # Connection string for SQLAlchemy/langchain_postgres (requires +psycopg driver prefix)
    CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    # Connection string for direct psycopg connections (without driver prefix)
    PSYCOPG_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    logger.info(f"Database connection string configured for host: {DB_HOST}:{DB_PORT}")
    logger.info(f"Collection name configured: {COLLECTION_NAME}")

except Exception as e:
    logger.error(f"Failed to configure database connection: {e}")
    sys.exit(1)


# Initialize embeddings model
try:
    # Load Azure OpenAI configuration from environment variables
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    # Validate required environment variables
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT
    }

    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    logger.info("Azure OpenAI embeddings model initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize embeddings model: {e}")
    sys.exit(1)


# ============================================================================
# Configuration Helper Functions
# ============================================================================

def get_connection():
    """Create a new psycopg connection using the configured DSN.
    
    Note: This creates a new connection for each call. For production use,
    consider implementing connection pooling (e.g., using psycopg.pool.ConnectionPool
    or SQLAlchemy's connection pooling) to improve performance and resource management.
    """
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    return psycopg.connect(PSYCOPG_DSN)


def get_vectorstore():
    """Create a configured PGVector instance for the current collection.

    Note: Requires langchain_postgres to be installed.
    """
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
    )


# Export constants for external use
def get_collection_name():
    """Get the collection name for the vector database."""
    return COLLECTION_NAME


# ============================================================================
# Core Storage Functions
# ============================================================================

def store_documents(documents: List[Document], collection_metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Store multiple documents using helper methods.
    
    If collection_metadata is provided, it will initialize the collection first with metadata.
    
    Args:
        documents: List of Document objects to store
        collection_metadata: Optional metadata for the collection
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if collection_metadata:
            # Initialize collection with metadata using first document or empty document
            init_doc = documents[0] if documents else Document(page_content="", metadata={})
            remaining_docs = documents[1:] if documents else []
            
            PGVector.from_documents(
                embedding=embeddings,
                documents=[init_doc],
                collection_name=COLLECTION_NAME,
                connection=CONNECTION_STRING,
                collection_metadata=collection_metadata
            )
            logger.info(f"Collection '{COLLECTION_NAME}' initialized with metadata")
            
            # Add remaining documents if any
            if remaining_docs:
                vectorstore = get_vectorstore()
                vectorstore.add_documents(remaining_docs)
                logger.info(f"Added {len(remaining_docs)} additional documents")
        else:
            # No metadata - use helper methods directly
            vectorstore = get_vectorstore()
            vectorstore.add_documents(documents)
            logger.info(f"Stored {len(documents)} documents")
        
        return True
    except Exception as e:
        logger.error(f"Failed to store documents: {e}")
        return False

