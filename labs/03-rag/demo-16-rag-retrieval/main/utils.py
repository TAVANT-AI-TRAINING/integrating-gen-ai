import logging
from typing import List, Any
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def format_docs(docs: List[Document]) -> str:
    """
    Format documents for display.
    
    Args:
        docs: List of document objects
    
    Returns:
        Formatted string with document information
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get('source', 'N/A')
        page = d.metadata.get('page', 'N/A')
        parts.append(f"[{i}] source={src} page={page}\n{d.page_content}")
    return "\n\n".join(parts)


format_docs_runnable = RunnableLambda(format_docs)


def test_query(retriever: Any, query: str, show_full_content: bool = False) -> List[Document]:
    """
    Test a retriever with a query and log the results.
    
    Args:
        retriever: Retriever instance
        query: Query string
        show_full_content: If True, show full document content; otherwise show preview
    
    Returns:
        List of retrieved documents
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Query: '{query}'")
    logger.info('='*80)

    docs = retriever.invoke(query)
    logger.info(f"\nRetrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs):
        logger.info(f"[Document {i+1}]")
        logger.info(f"  Source: {doc.metadata.get('source', 'N/A')}")
        logger.info(f"  Page: {doc.metadata.get('page', 'N/A')}")
        if show_full_content:
            logger.info(f"  Content:\n{doc.page_content}")
        else:
            logger.info(f"  Content Preview: {doc.page_content[:200]}...")
        logger.info("-" * 80)
    return docs

