"""
Demo 09: Complete RAG Ingestion Pipeline

This demo shows the complete RAG (Retrieval-Augmented Generation) ingestion pipeline:
1. Load documents from multiple sources (PDF, text, web)
2. Chunk documents into smaller pieces
3. Generate embeddings for each chunk
4. Store chunks with embeddings in vector database
5. Query for similar documents

Supports two vector databases via configuration:
- ChromaDB (local, file-based)
- Pinecone (cloud-based)

Usage:
    # Set VECTOR_DB=chromadb or pinecone in .env
    uv run python main.py
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
# Vector database selection
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb").lower()

# OpenAI Configuration
OPENAI_API_EMBEDDING_KEY = os.getenv("OPENAI_API_EMBEDDING_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_EMBEDDING_KEY:
    raise ValueError("OPENAI_API_EMBEDDING_KEY not found in environment variables")

# Document source configuration
DOCS_DIR = Path("Documents")
PDF_FILE = DOCS_DIR / "company_policy.pdf"
WEB_URL = "https://www.python.org/"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

print("=" * 70)
print("RAG INGESTION PIPELINE CONFIGURATION")
print("=" * 70)
print(f"Vector Database: {VECTOR_DB.upper()}")
print(f"Chunk Size: {CHUNK_SIZE} characters")
print(f"Chunk Overlap: {CHUNK_OVERLAP} characters")
print("=" * 70)

# ============================================================================
# INITIALIZE EMBEDDINGS
# ============================================================================
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_EMBEDDING_KEY,
    model=OPENAI_EMBEDDING_MODEL
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
# STEP 1: LOAD DOCUMENTS FROM MULTIPLE SOURCES
# ============================================================================
def load_documents() -> List[Document]:
    """
    Load documents from PDF, text files, and web.
    
    Returns:
        List of Document objects
    """
    print("=" * 70)
    print("STEP 1: LOADING DOCUMENTS")
    print("=" * 70)
    
    all_docs = []
    
    # Load PDF
    print("\n[1.1] Loading PDF...")
    if PDF_FILE.exists():
        try:
            pdf_loader = PyPDFLoader(str(PDF_FILE))
            pdf_docs = pdf_loader.load()
            all_docs.extend(pdf_docs)
            print(f"  ✓ Loaded {len(pdf_docs)} page(s) from PDF")
        except Exception as e:
            print(f"  ✗ Error loading PDF: {e}")
    else:
        print(f"  ⚠️  PDF not found: {PDF_FILE}")
    
    # Load text files
    print("\n[1.2] Loading text files...")
    txt_files = list(DOCS_DIR.glob("*.txt"))
    if txt_files:
        for txt_file in txt_files:
            try:
                text_loader = TextLoader(str(txt_file), encoding="utf-8")
                text_docs = text_loader.load()
                all_docs.extend(text_docs)
                print(f"  ✓ Loaded: {txt_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading {txt_file.name}: {e}")
    else:
        print("  ⚠️  No .txt files found")
    
    # # Load web page
    # print("\n[1.3] Loading web page...")
    # try:
    #     web_loader = WebBaseLoader(WEB_URL)
    #     web_docs = web_loader.load()
    #     all_docs.extend(web_docs)
    #     print(f"  ✓ Loaded web page")
    #     print(f"  ✓ Content length: {len(web_docs[0].page_content):,} characters")
    # except Exception as e:
    #     print(f"  ✗ Error loading web page: {e}")
    
    # print(f"\n✓ Total documents loaded: {len(all_docs)}")
    return all_docs


# ============================================================================
# STEP 2: CHUNK DOCUMENTS
# ============================================================================
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of chunked Document objects
    """
    print("\n" + "=" * 70)
    print("STEP 2: CHUNKING DOCUMENTS")
    print("=" * 70)
    
    if not documents:
        print("⚠️  No documents to chunk!")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Calculate statistics
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_length = min(chunk_lengths) if chunk_lengths else 0
    max_length = max(chunk_lengths) if chunk_lengths else 0
    
    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"  - Average length: {avg_length:.0f} characters")
    print(f"  - Min length: {min_length} characters")
    print(f"  - Max length: {max_length} characters")
    
    # Show sample chunk
    if chunks:
        print(f"\n📄 Sample Chunk:")
        print(f"  Source: {chunks[0].metadata.get('source', 'Unknown')}")
        print(f"  Length: {len(chunks[0].page_content)} characters")
        preview = chunks[0].page_content[:200].replace('\n', ' ')
        print(f"  Preview: {preview}...")
    
    return chunks


# ============================================================================
# STEP 3 & 4: GENERATE EMBEDDINGS AND STORE IN VECTOR DATABASE
# ============================================================================
def store_chunks(chunks: List[Document]) -> None:
    """
    Generate embeddings for chunks and store in vector database.
    
    This step combines:
    - Step 3: Generate embeddings using OpenAI
    - Step 4: Store chunks with embeddings in vector database
    
    Args:
        chunks: List of chunked Document objects
    """
    print("\n" + "=" * 70)
    print("STEPS 3 & 4: GENERATE EMBEDDINGS & STORE")
    print("=" * 70)
    
    if not chunks:
        print("⚠️  No chunks to store!")
        return
    
    print(f"\n🔄 Processing {len(chunks)} chunks...")
    print("  - Generating embeddings with OpenAI")
    print(f"  - Storing in {VECTOR_DB.upper()}")
    
    try:
        # Add documents (automatically generates embeddings and stores)
        vectorstore.add_documents(chunks)
        print(f"\n✓ Successfully stored {len(chunks)} chunks with embeddings!")
    except Exception as e:
        print(f"\n✗ Error storing chunks: {e}")
        raise


# ============================================================================
# STEP 5: QUERY FOR SIMILAR DOCUMENTS
# ============================================================================
def query_documents(query: str, k: int = 3) -> None:
  """
  Query the vector database for similar documents.
  
  Args:
    query: Search query string
    k: Number of results to return
  """
  print("\n" + "=" * 70)
  print("STEP 5: QUERYING FOR SIMILAR DOCUMENTS")
  print("=" * 70)
  print(f"\nQuery: \"{query}\"")
  print(f"Top {k} results:\n")
  
  try:
    # Perform similarity search with scores
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    if not results:
      print("⚠️  No results found!")
      return
    
    # Display results with scores and rank
    for rank, (doc, score) in enumerate(results, 1):
      print(f"--- Rank {rank} (Score: {score:.4f}) ---")
      print(f"Source: {doc.metadata.get('source', 'Unknown')}")
      if 'page' in doc.metadata:
        print(f"Page: {doc.metadata['page']}")
      print(f"Length: {len(doc.page_content)} characters")
      preview = doc.page_content[:300].replace('\n', ' ')
      print(f"Content: {preview}...")
      print()
      
    
  except Exception as e:
    print(f"✗ Error querying documents: {e}")
    raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run the complete RAG ingestion pipeline."""
    print("\n" + "=" * 70)
    print("DEMO 09: COMPLETE RAG INGESTION PIPELINE")
    print("=" * 70)
    
    # Step 1: Load documents
    documents = load_documents()
    
    if not documents:
        print("\n⚠️  No documents loaded. Please add documents to the Documents/ folder.")
        return
    
    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("\n⚠️  No chunks created.")
        return
    
    # Steps 3 & 4: Generate embeddings and store
    store_chunks(chunks)
    
    # Step 5: Query for similar documents
    print("\n" + "=" * 70)
    print("TESTING SIMILARITY SEARCH")
    print("=" * 70)
    
    # Example queries
    queries = [
        "What is the remote work policy?",
        "Code review guidelines",
        "Python programming"
    ]
    
    for query in queries:
        query_documents(query, k=2)
    
    # Summary
    print("=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\n✓ Summary:")
    print(f"  1. Loaded {len(documents)} documents")
    print(f"  2. Created {len(chunks)} chunks")
    print(f"  3. Generated embeddings with OpenAI")
    print(f"  4. Stored in {VECTOR_DB.upper()}")
    print(f"  5. Demonstrated similarity search")
    print("\n✓ RAG ingestion pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()