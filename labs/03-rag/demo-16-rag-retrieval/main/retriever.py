from typing import Optional, Dict, Any


def create_retriever(
    vector_store: Any,
    k: int = 4,
    search_type: str = "similarity",
    metadata_filter: Optional[Dict] = None
) -> Any:
    """
    Create a retriever from a vector store with specified configuration.
    
    Args:
        vector_store: Vector store instance (e.g., PGVector)
        k: Number of documents to retrieve
        search_type: Search type ("similarity" or "mmr")
        metadata_filter: Optional metadata filter dictionary
    
    Returns:
        Configured retriever instance
    """
    search_kwargs = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
