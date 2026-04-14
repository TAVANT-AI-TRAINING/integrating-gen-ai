import logging
from typing import Optional, Dict, List, Any
from .vector_store import load_env, get_embeddings, get_connection_string, get_vector_store
from .retriever import create_retriever
from .utils import test_query
from .lcel import create_lcel_chain

logger = logging.getLogger(__name__)


def test_retriever_with_k(vector_store: Any, k: int, query: str, label: Optional[str] = None) -> Any:
    """Create a retriever with a specific k value and test it with a query."""
    if label:
        logger.info(f"\n[{label}] k={k}")
    else:
        logger.info(f"\n--- With k={k} ---")
    retriever = create_retriever(vector_store, k=k)
    test_query(retriever, query)
    return retriever


def compare_retrievers_with_k(vector_store: Any, k_values: List[int], query: str) -> None:
    """Create and compare retrievers with different k values."""
    logger.info("\n[Comparing Different K Values]")
    for k in k_values:
        test_retriever_with_k(vector_store, k, query)


def test_filtering(vector_store: Any, k: int = 4, query: str = "What is the policy?") -> None:
    """Create and test retrievers with and without metadata filtering."""
    logger.info("\n[Metadata Filtering]")
    
    logger.info("\n--- Without Filter (searches all documents) ---")
    retriever_no_filter = create_retriever(vector_store, k=k)
    test_query(retriever_no_filter, query)

    logger.info("\n--- With Source Filter (searches specific document) ---")
    retriever_filtered = create_retriever(
        vector_store,
        k=k,
        metadata_filter={"source": "hr_handbook.pdf"}
    )
    test_query(retriever_filtered, query)


def test_search_types(vector_store: Any, k: int = 4, query: str = "How do I request time off?") -> None:
    """Create and test retrievers with different search types."""
    logger.info("\n[Different Search Types]")

    logger.info("\n--- Similarity Search (default) ---")
    retriever_similarity = create_retriever(vector_store, k=k, search_type="similarity")
    test_query(retriever_similarity, query)

    logger.info("\n--- MMR Search (Maximum Marginal Relevance - diverse results) ---")
    logger.info("Note: MMR balances relevance with diversity to avoid redundant results")
    try:
        retriever_mmr = create_retriever(vector_store, k=k, search_type="mmr")
        test_query(retriever_mmr, query)
    except Exception as e:
        logger.warning(f"MMR not available: {e}")


def run_lcel_with_k(vector_store: Any, k: int, query: str, metadata_filter: Optional[Dict] = None) -> Any:
    """Run LCEL retrieval chain with a specific k value."""
    retrieval_chain = create_lcel_chain(vector_store, k=k, metadata_filter=metadata_filter)
    result = retrieval_chain.invoke(query)
    
    filter_label = f" (filtered)" if metadata_filter else ""
    logger.info(f"\nLCEL result (k={k}{filter_label}):\n")
    logger.info(result)
    return result


def run_lcel_scenarios(vector_store: Any) -> None:
    """Run LCEL retrieval chain scenarios with different configurations."""
    logger.info("\n--- LCEL with k=3 ---")
    run_lcel_with_k(vector_store, k=3, query="What are the employee benefits?")
    
    logger.info("\n--- LCEL with k=5 ---")
    run_lcel_with_k(vector_store, k=5, query="What are the employee benefits?")
    
    logger.info("\n--- LCEL with k=4 and metadata filter ---")
    run_lcel_with_k(
        vector_store, 
        k=4, 
        query="What is the policy?",
        metadata_filter={"source": "hr_handbook.pdf"}
    )


def run() -> None:
    """Main entry point that orchestrates all retrieval scenarios."""
    logger.info("\n" + "="*80)
    logger.info("Build and Test a Configurable Retriever")
    logger.info("="*80)

    # Initialize components
    env = load_env()
    embeddings = get_embeddings(env)
    connection_string = get_connection_string(env)
    vector_store = get_vector_store(connection_string, env["COLLECTION_NAME"], embeddings)

    # Run different scenarios
    logger.info("\n" + "="*80)
    logger.info("[Scenario 1] Basic Retrieval")
    logger.info("*"*80)
    test_retriever_with_k(vector_store, k=3, query="What is the vacation policy?", label="Basic Retrieval")

    logger.info("\n" + "="*80)
    logger.info("[Scenario 2] Comparing Different K Values")
    logger.info("*"*80)
    compare_retrievers_with_k(vector_store, k_values=[2, 6], query="What are the employee benefits?")

    logger.info("\n" + "="*80)
    logger.info("[Scenario 3] Metadata Filtering")
    logger.info("*"*80)
    test_filtering(vector_store)

    logger.info("\n" + "="*80)
    logger.info("[Scenario 4] Different Search Types")
    logger.info("*"*80)
    test_search_types(vector_store)

    logger.info("\n" + "="*80)
    logger.info("[Scenario 5] LCEL Retrieval Chains")
    logger.info("*"*80)
    run_lcel_scenarios(vector_store)

    logger.info("*"*80)


    logger.info("\n\n" + "="*80)
    logger.info("Complete - Key Takeaways")
    logger.info("✓ Retrievers standardize the search interface")
    logger.info("✓ K value controls context vs noise tradeoff")
    logger.info("✓ Metadata filtering enables scoped searches")
    logger.info("✓ Always use the same embedding model as ingestion")
    logger.info("*"*80)
