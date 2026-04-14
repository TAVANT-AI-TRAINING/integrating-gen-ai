"""
Query Service

This service handles similarity search queries against the vector database.
Provides functionality to search for relevant document chunks based on user queries.
"""

import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

from main.services.embedding_service import get_vectorstore, get_collection_name

logger = logging.getLogger(__name__)


def similarity_search(query: str, k: int = 1) -> List[Document]:
    """
    Perform similarity search on the vector database.
    
    Args:
        query: The search query string
        k: Number of results to return (default: 1)
        
    Returns:
        List of Document objects matching the query
        
    Raises:
        RuntimeError: If vectorstore is not available or search fails
    """
    try:
        logger.info(f"Performing similarity search: query='{query[:50]}...', k={k}")
        
        vectorstore = get_vectorstore()
        
        # Perform similarity search
        documents = vectorstore.similarity_search(query=query, k=k)
        
        logger.info(f"Found {len(documents)} results")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to perform similarity search: {e}")
        raise RuntimeError(f"Search failed: {e}")


def format_search_results(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Format search results into a structured dictionary format.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of dictionaries containing formatted results
    """
    results = []
    
    for idx, doc in enumerate(documents):
        result = {
            "rank": idx + 1,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "content_length": len(doc.page_content)
        }
        results.append(result)
    
    return results


def query_documents(query: str, k: int = 1) -> Dict[str, Any]:
    """
    Query documents and return formatted results.
    
    Args:
        query: The search query string
        k: Number of results to return (default: 1)
        
    Returns:
        Dictionary containing query results and metadata
    """
    try:
        documents = similarity_search(query, k)
        formatted_results = format_search_results(documents)
        
        return {
            "query": query,
            "collection": get_collection_name(),
            "num_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Failed to query documents: {e}")
        raise

