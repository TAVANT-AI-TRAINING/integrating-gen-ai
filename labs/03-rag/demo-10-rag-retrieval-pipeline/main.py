"""
Demo 10: RAG Retrieval Pipeline

This demo focuses PURELY on the RETRIEVAL phase of RAG by connecting to an 
EXISTING vector database and demonstrating various retrieval strategies.

Prerequisites:
    - Run demo-09 first to ingest documents into vector database
    - Ensure vector database (ChromaDB or Pinecone) has data

What this demo does:
    1. Connect to existing vector database
    2. Verify data exists
    3. Demonstrate 6 retrieval strategies with detailed output
    4. Show chunk content (100-200 chars) for each result
    5. Compare retrieval quality across different approaches

Focus: Understanding retrieval strategies and quality analysis
(For ingestion, see demo-09. For complete RAG with generation, see demo-11)

Supports two vector databases via configuration:
- ChromaDB (local, file-based)
- Pinecone (cloud-based)

Usage:
    # After running demo-09 to ingest data:
    # Set VECTOR_DB=chromadb or pinecone in .env
    uv run python main.py
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Retrieval display configuration
CHUNK_PREVIEW_LENGTH = 200  # Show 200 chars of each retrieved chunk

print("=" * 70)
print("RAG RETRIEVAL PIPELINE - CONNECT TO EXISTING VECTOR STORE")
print("=" * 70)
print(f"Vector Database: {VECTOR_DB.upper()}")
print(f"Chunk Preview Length: {CHUNK_PREVIEW_LENGTH} characters")
print("\n⚠️  Prerequisites: Run demo-09 first to ingest documents!")
print("=" * 70)

# ============================================================================
# INITIALIZE EMBEDDINGS
# ============================================================================
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
print("\n✓ OpenAI embeddings initialized: text-embedding-3-small")

# ============================================================================
# INITIALIZE VECTOR STORE (Config-Driven)
# ============================================================================
vectorstore = None

if VECTOR_DB == "chromadb":
    from langchain_chroma import Chroma
    
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_policies")
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    
    print(f"✓ ChromaDB initialized")
    print(f"  - Storage: {CHROMA_DB_DIR}")
    print(f"  - Collection: {COLLECTION_NAME}")

elif VECTOR_DB == "pinecone":
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
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        print(f"✓ Created Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    
    print(f"✓ Pinecone initialized")
    print(f"  - Index: {PINECONE_INDEX_NAME}")
    print(f"  - Cloud: {PINECONE_CLOUD}/{PINECONE_REGION}")

else:
    raise ValueError(f"Unsupported VECTOR_DB: {VECTOR_DB}. Use 'chromadb' or 'pinecone'")

print(f"✓ Vector store ready!\n")


# ============================================================================
# VERIFY VECTOR STORE HAS DATA
# ============================================================================
def verify_vector_store() -> int:
    """Verify that vector store has data and return document count."""
    print("=" * 70)
    print("VERIFYING VECTOR STORE DATA")
    print("=" * 70)
    
    try:
        # Try a test query to verify data exists
        test_results = vectorstore.similarity_search("test", k=1)
        
        if not test_results:
            print("\n❌ Vector store is EMPTY!")
            print("\n📋 Solution:")
            print("   1. Navigate to demo-09-rag-ingestion-pipeline")
            print("   2. Run: uv run python main.py")
            print("   3. This will ingest documents into the vector store")
            print("   4. Then come back and run this demo")
            return 0
        
        # Try to get a rough count
        sample_results = vectorstore.similarity_search("document", k=100)
        doc_count = len(sample_results)
        
        print(f"\n✓ Vector store has data!")
        print(f"  - Found at least {doc_count} chunks")
        print(f"  - Ready for retrieval demonstrations")
        
        # Show sample
        if sample_results:
            print(f"\n📄 Sample chunk:")
            sample = sample_results[0]
            print(f"  Source: {sample.metadata.get('source', 'Unknown')}")
            preview = sample.page_content[:CHUNK_PREVIEW_LENGTH]
            print(f"  Content: {preview}...")
        
        return doc_count
        
    except Exception as e:
        print(f"\n⚠️  Error verifying vector store: {e}")
        print("\nThis might mean:")
        print("  1. Vector store is empty (run demo-09 first)")
        print("  2. Wrong VECTOR_DB configuration in .env")
        print("  3. Collection/index name mismatch")
        return 0


# ============================================================================
# STEP 4: RETRIEVAL STRATEGIES
# ============================================================================

def similarity_search_basic(query: str, k: int = 3) -> List[Document]:
    """Basic similarity search with k results and detailed content display."""
    print(f"\n[Retrieval] Similarity Search (k={k})")
    print(f"Query: \"{query}\"")
    print("-" * 70)
    
    results = vectorstore.similarity_search(query, k=k)
    
    print(f"\n✓ Retrieved {len(results)} documents\n")
    
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] Metadata:")
        print(f"      Source: {doc.metadata.get('source', 'Unknown')}")
        if 'page' in doc.metadata:
            print(f"      Page: {doc.metadata['page']}")
        
        # Show chunk content (200 chars)
        content_preview = doc.page_content[:CHUNK_PREVIEW_LENGTH]
        content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
        print(f"      Content: {content_preview}...")
        print(f"      Length: {len(doc.page_content)} characters\n")
    
    return results


def similarity_search_with_score(query: str, k: int = 3) -> List[tuple]:
    """Similarity search with relevance scores and detailed content.
    
    Note: Scores represent distance in vector space:
    - Lower score = more similar = more relevant
    - Higher score = less similar = less relevant
    """
    print(f"\n[Retrieval] Similarity Search with Scores (k={k})")
    print(f"Query: \"{query}\"")
    print("  (Lower score = Higher relevance)")
    print("-" * 70)
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    print(f"\n✓ Retrieved {len(results)} documents with scores\n")
    
    for i, (doc, score) in enumerate(results, 1):
        # Determine relevance level (distance-based: lower = more relevant)
        if score > 0.8:
            relevance = "🔴 Low relevance (far distance)"
        elif score > 0.6:
            relevance = "🟡 Medium relevance"
        else:
            relevance = "🟢 High relevance (close distance)"
        
        print(f"  [{i}] Score: {score:.4f} {relevance}")
        print(f"      Source: {doc.metadata.get('source', 'Unknown')}")
        if 'page' in doc.metadata:
            print(f"      Page: {doc.metadata['page']}")
        
        # Show chunk content
        content_preview = doc.page_content[:CHUNK_PREVIEW_LENGTH]
        content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
        print(f"      Content: {content_preview}...")
        print(f"      Length: {len(doc.page_content)} characters\n")
    
    return results


def metadata_filtering_search(query: str, filter_dict: dict, k: int = 3) -> List[Document]:
    """Search with metadata filtering to limit results to specific sources/properties.
    
    Args:
        query: Search query
        filter_dict: Metadata filters e.g., {'source': 'specific_file.pdf'}
        k: Number of results
    """
    print(f"\n[Retrieval] Metadata Filtering Search (k={k})")
    print(f"Query: \"{query}\"")
    print(f"Filter: {filter_dict}")
    print("  (Only returns results matching metadata criteria)")
    print("-" * 70)
    
    try:
        # Different vector stores have different filtering syntax
        if VECTOR_DB == "chromadb":
            results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
        elif VECTOR_DB == "pinecone":
            results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = vectorstore.similarity_search(query, k=k)
            print("⚠️  Filtering not implemented for this vector store, showing all results")
        
        print(f"\n✓ Retrieved {len(results)} documents matching filter\n")
        
        if not results:
            print("  ⚠️  No results found matching the filter criteria")
            print("  Try: Adjusting the filter or checking available metadata")
            return results
        
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] Metadata:")
            for key, value in doc.metadata.items():
                print(f"      {key}: {value}")
            
            # Show chunk content
            content_preview = doc.page_content[:CHUNK_PREVIEW_LENGTH]
            content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
            print(f"      Content: {content_preview}...")
            print(f"      Length: {len(doc.page_content)} characters\n")
        
        return results
        
    except Exception as e:
        print(f"\n⚠️  Error during filtered search: {e}")
        print("  This might be due to:")
        print("  1. Unsupported filter format for this vector store")
        print("  2. Metadata field doesn't exist")
        print("  3. Filter syntax incompatibility")
        return []


def max_marginal_relevance_search(query: str, k: int = 3, fetch_k: int = 10) -> List[Document]:
    """
    MMR search - balances relevance with diversity.
    Fetches fetch_k candidates, returns k diverse results.
    """
    print(f"\n[Retrieval] MMR Search (k={k}, fetch_k={fetch_k})")
    print(f"Query: \"{query}\"")
    print("  (Maximizes diversity while maintaining relevance)")
    print("-" * 70)
    
    try:
        results = vectorstore.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=fetch_k
        )
        
        print(f"\n✓ Retrieved {len(results)} diverse documents\n")
        
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] Metadata:")
            print(f"      Source: {doc.metadata.get('source', 'Unknown')}")
            if 'page' in doc.metadata:
                print(f"      Page: {doc.metadata['page']}")
            
            # Show chunk content
            content_preview = doc.page_content[:CHUNK_PREVIEW_LENGTH]
            content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
            print(f"      Content: {content_preview}...")
            print(f"      Length: {len(doc.page_content)} characters\n")
        
        return results
    except Exception as e:
        print(f"\n✗ MMR not supported by this vector store: {e}")
        return []


def retriever_interface_demo(query: str):
    """Demonstrate using retriever interface with configuration."""
    print(f"\n[Retrieval] Using Retriever Interface")
    print(f"Query: \"{query}\"")
    print("-" * 70)
    
    # Create retriever with specific configuration
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    results = retriever.invoke(query)
    
    print(f"\n✓ Retriever returned {len(results)} documents\n")
    
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] Metadata:")
        print(f"      Source: {doc.metadata.get('source', 'Unknown')}")
        if 'page' in doc.metadata:
            print(f"      Page: {doc.metadata['page']}")
        
        # Show chunk content
        content_preview = doc.page_content[:CHUNK_PREVIEW_LENGTH]
        content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
        print(f"      Content: {content_preview}...")
        print(f"      Length: {len(doc.page_content)} characters\n")
    
    return results


def retriever_with_filter(query: str, metadata_filter: Dict[str, Any] = None):
    """Demonstrate retrieval with metadata filtering."""
    print(f"\n[Retrieval] Retrieval with Metadata Filter")
    print(f"Query: \"{query}\"")
    if metadata_filter:
        print(f"Filter: {metadata_filter}")
    
    try:
        if VECTOR_DB == "chromadb" and metadata_filter:
            # ChromaDB uses where clause
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 3,
                    "filter": metadata_filter
                }
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
        
        results = retriever.invoke(query)
        
        print(f"✓ Retrieved {len(results)} filtered documents")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata.get('source', 'Unknown')[:50]}...")
        
        return results
    except Exception as e:
        print(f"✗ Filtering error: {e}")
        return []


# ============================================================================
# RETRIEVAL QUALITY ANALYSIS
# ============================================================================

def analyze_retrieval_quality(query: str, k_values: List[int] = [1, 2, 3, 5]):
    """Compare retrieval quality across different k values."""
    print("\n" + "=" * 70)
    print("RETRIEVAL QUALITY ANALYSIS")
    print("=" * 70)
    print(f"\nQuery: \"{query}\"")
    print(f"Testing k values: {k_values}")
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        if results:
            avg_score = sum(score for _, score in results) / len(results)
            print(f"  Average relevance score: {avg_score:.4f}")
            print(f"  Top result score: {results[0][1]:.4f}")
            print(f"  Bottom result score: {results[-1][1]:.4f}")
        else:
            print("  No results found")


def display_document_details(doc: Document, index: int = 1):
    """Display detailed information about a retrieved document."""
    print(f"\n{'=' * 70}")
    print(f"DOCUMENT #{index} - DETAILED VIEW")
    print(f"{'=' * 70}")
    
    # Metadata
    print("\n[Metadata]")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    
    # Content statistics
    print("\n[Content Statistics]")
    print(f"  Total length: {len(doc.page_content)} characters")
    print(f"  Word count: {len(doc.page_content.split())} words")
    print(f"  Line count: {len(doc.page_content.splitlines())} lines")
    
    # Full content preview (first 400 chars for detailed view)
    print("\n[Content Preview - First 400 characters]")
    content_preview = doc.page_content[:400]
    # Format for readability
    lines = content_preview.split('\n')
    for line in lines[:10]:  # Show up to 10 lines
        if line.strip():
            print(f"  {line}")
    if len(doc.page_content) > 400:
        print("  ...")
    
    print(f"\n{'=' * 70}")


# ============================================================================
# DEMONSTRATION SCENARIOS
# ============================================================================

def demonstrate_retrieval_scenarios():
    """Run comprehensive retrieval demonstrations."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING RETRIEVAL SCENARIOS")
    print("=" * 70)
    
    # Scenario 1: Different k values
    print("\n[Scenario 1] Comparing Different K Values")
    print("-" * 70)
    
    query1 = "What are the key policies?"
    
    for k in [2, 4, 6]:
        similarity_search_basic(query1, k=k)
    
    # Scenario 2: Retrieval with scores
    print("\n\n[Scenario 2] Retrieval with Relevance Scores")
    print("-" * 70)
    
    query2 = "remote work guidelines"
    similarity_search_with_score(query2, k=3)
    
    # Scenario 3: MMR for diversity
    print("\n\n[Scenario 3] MMR Search for Diverse Results")
    print("-" * 70)
    
    query3 = "company policies"
    max_marginal_relevance_search(query3, k=4, fetch_k=10)
    
    # Scenario 4: Retriever interface
    print("\n\n[Scenario 4] Using Retriever Interface")
    print("-" * 70)
    
    query4 = "code review process"
    retriever_interface_demo(query4)
    
    # Scenario 5: Quality analysis
    print("\n\n[Scenario 5] Retrieval Quality Analysis")
    print("-" * 70)
    
    query5 = "What are the guidelines?"
    analyze_retrieval_quality(query5, k_values=[1, 2, 3, 5])
    
    # Scenario 6: Metadata filtering
    print("\n\n[Scenario 6] Metadata Filtering Search")
    print("-" * 70)
    
    query6 = "What are the guidelines?"
    # First, get a sample to see available metadata
    sample = vectorstore.similarity_search(query6, k=1)
    if sample and sample[0].metadata:
        # Try to filter by source if available
        source_value = sample[0].metadata.get('source')
        if source_value:
            print(f"\nFiltering by source: {source_value}")
            metadata_filtering_search(query6, {'source': source_value}, k=3)
        else:
            print("\n⚠️  No 'source' metadata available for filtering")
            print("Demonstrating concept with available metadata:")
            # Try with any available metadata key
            if sample[0].metadata:
                key = list(sample[0].metadata.keys())[0]
                value = sample[0].metadata[key]
                metadata_filtering_search(query6, {key: value}, k=3)
    else:
        print("\n⚠️  No metadata available for filtering demonstration")
    
    # Scenario 7: Detailed document inspection
    print("\n\n[Scenario 7] Detailed Document Inspection")
    print("-" * 70)
    
    query7 = "employee benefits"
    results = similarity_search_with_score(query7, k=2)
    
    if results:
        print("\nInspecting top result in detail:")
        display_document_details(results[0][0], index=1)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run the complete demonstration."""
    print("\n" + "=" * 70)
    print("DEMO 10: RAG RETRIEVAL PIPELINE")
    print("(Connect to Existing Vector Store & Demonstrate Retrieval)")
    print("=" * 70)
    
    # Verify vector store has data
    doc_count = verify_vector_store()
    
    if doc_count == 0:
        print("\n" + "=" * 70)
        print("❌ CANNOT PROCEED - VECTOR STORE IS EMPTY")
        print("=" * 70)
        return
    
    # Demonstrate various retrieval scenarios
    print("\n" + "=" * 70)
    print("BEGINNING RETRIEVAL DEMONSTRATIONS")
    print("=" * 70)
    
    demonstrate_retrieval_scenarios()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ RAG RETRIEVAL PIPELINE DEMONSTRATION FINISHED!")
    print("=" * 70)
    print("\n📋 Summary:")
    print(f"  1. Connected to existing {VECTOR_DB.upper()} vector store")
    print(f"  2. Verified data exists (at least {doc_count} chunks)")
    print(f"  3. Demonstrated 7 retrieval scenarios")
    print(f"  4. Showed detailed chunk content (200 chars per result)")
    print("\n🎯 Retrieval Strategies Demonstrated:")
    print("  • Basic similarity search (different k values)")
    print("  • Similarity search with relevance scores (distance-based)")
    print("  • MMR search for diverse results")
    print("  • Retriever interface configuration")
    print("  • Retrieval quality analysis")
    print("  • Metadata filtering (source-specific retrieval)")
    print("  • Detailed document inspection")
    print("\n💡 Key Learnings:")
    print("  • How similarity search finds relevant chunks")
    print("  • What relevance scores indicate")
    print("  • How k value affects retrieval quality")
    print("  • When to use MMR for diversity")
    print("\n📚 Related Demos:")
    print("  • demo-09: RAG Ingestion (run this first to prepare data)")
    print("  • demo-11: Complete RAG with LLM generation")
    print("=" * 70)


if __name__ == "__main__":
    main()
