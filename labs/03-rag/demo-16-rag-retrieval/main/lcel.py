from typing import Optional, Dict, Any
from .retriever import create_retriever
from .utils import format_docs_runnable


def create_lcel_chain(
    vector_store: Any,
    k: int,
    metadata_filter: Optional[Dict] = None
) -> Any:
    """
    Create an LCEL retrieval chain.
    
    Args:
        vector_store: Vector store instance
        k: Number of documents to retrieve
        metadata_filter: Optional metadata filter dictionary
    
    Returns:
        LCEL chain that can be invoked with a query
    """
    retriever = create_retriever(vector_store, k=k, metadata_filter=metadata_filter)
    retrieval_chain = retriever | format_docs_runnable
    return retrieval_chain

