"""
Demo 10: Complete RAG Pipeline

This demo shows the COMPLETE RAG (Retrieval-Augmented Generation) flow:
1. Load documents from multiple sources
2. Chunk documents
3. Generate embeddings and store in vector database
4. Retrieve relevant chunks based on query
5. Generate answer using LLM with retrieved context

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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Document source configuration
DOCS_DIR = Path("Documents")
PDF_FILE = DOCS_DIR / "company_policy.pdf"
WEB_URL = "https://www.python.org/"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

print("=" * 70)
print("COMPLETE RAG PIPELINE CONFIGURATION")
print("=" * 70)
print(f"Vector Database: {VECTOR_DB.upper()}")
print(f"LLM Model: {OPENAI_MODEL}")
print(f"Chunk Size: {CHUNK_SIZE} characters")
print(f"Chunk Overlap: {CHUNK_OVERLAP} characters")
print("=" * 70)

# ============================================================================
# INITIALIZE EMBEDDINGS
# ============================================================================
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
print("\n✓ OpenAI embeddings initialized: text-embedding-3-small")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0
)
print(f"✓ OpenAI LLM initialized: {OPENAI_MODEL}")

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
# STEP 1: LOAD DOCUMENTS
# ============================================================================
def load_documents() -> List[Document]:
    """Load documents from PDF, text files, and web."""
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
    
    # Load web page
    print("\n[1.3] Loading web page...")
    try:
        web_loader = WebBaseLoader(WEB_URL)
        web_docs = web_loader.load()
        all_docs.extend(web_docs)
        print(f"  ✓ Loaded web page")
        print(f"  ✓ Content length: {len(web_docs[0].page_content):,} characters")
    except Exception as e:
        print(f"  ✗ Error loading web page: {e}")
    
    print(f"\n✓ Total documents loaded: {len(all_docs)}")
    return all_docs


# ============================================================================
# STEP 2: CHUNK DOCUMENTS
# ============================================================================
def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
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
    
    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"  - Average length: {avg_length:.0f} characters")
    print(f"  - Min length: {min(chunk_lengths) if chunk_lengths else 0} characters")
    print(f"  - Max length: {max(chunk_lengths) if chunk_lengths else 0} characters")
    
    return chunks


# ============================================================================
# STEP 3: STORE CHUNKS IN VECTOR DATABASE
# ============================================================================
def store_chunks(chunks: List[Document]) -> None:
    """Generate embeddings and store chunks in vector database."""
    print("\n" + "=" * 70)
    print("STEP 3: STORE CHUNKS WITH EMBEDDINGS")
    print("=" * 70)
    
    if not chunks:
        print("⚠️  No chunks to store!")
        return
    
    print(f"\n🔄 Processing {len(chunks)} chunks...")
    print("  - Generating embeddings with OpenAI")
    print(f"  - Storing in {VECTOR_DB.upper()}")
    
    try:
        vectorstore.add_documents(chunks)
        print(f"\n✓ Successfully stored {len(chunks)} chunks with embeddings!")
    except Exception as e:
        print(f"\n✗ Error storing chunks: {e}")
        raise


# ============================================================================
# STEP 4: RETRIEVE RELEVANT DOCUMENTS
# ============================================================================
def retrieve_documents(query: str, k: int = 3) -> List[Document]:
    """
    Retrieve relevant documents based on query.
    
    Args:
        query: Search query string
        k: Number of documents to retrieve
        
    Returns:
        List of relevant Document objects
    """
    print("\n" + "=" * 70)
    print("STEP 4: RETRIEVE RELEVANT DOCUMENTS")
    print("=" * 70)
    print(f"\nQuery: \"{query}\"")
    print(f"Retrieving top {k} most relevant chunks...")
    
    try:
        # Perform similarity search
        results = vectorstore.similarity_search(query, k=k)
        
        if not results:
            print("⚠️  No results found!")
            return []
        
        print(f"\n✓ Retrieved {len(results)} relevant chunks:")
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] Source: {doc.metadata.get('source', 'Unknown')}")
            if 'page' in doc.metadata:
                print(f"    Page: {doc.metadata['page']}")
            print(f"    Length: {len(doc.page_content)} characters")
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"    Preview: {preview}...")
        
        return results
        
    except Exception as e:
        print(f"✗ Error retrieving documents: {e}")
        raise


# ============================================================================
# STEP 5: GENERATE ANSWER USING LLM
# ============================================================================
def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents as context for LLM."""
    return "\n\n".join([doc.page_content for doc in docs])


def generate_answer(query: str, retrieved_docs: List[Document]) -> str:
    """
    Generate answer using LLM with retrieved context.
    
    Args:
        query: User's question
        retrieved_docs: Retrieved relevant documents
        
    Returns:
        Generated answer string
    """
    print("\n" + "=" * 70)
    print("STEP 5: GENERATE ANSWER WITH LLM")
    print("=" * 70)
    
    if not retrieved_docs:
        return "I don't have enough information to answer that question."
    
    # Create RAG prompt template
    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: Provide a clear, concise answer based solely on the context above. If the context doesn't contain relevant information, say so."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {
            "context": lambda x: format_docs(retrieved_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n🤖 Generating answer with LLM...")
    print(f"   Using {len(retrieved_docs)} retrieved chunks as context")
    
    # Generate answer
    answer = rag_chain.invoke(query)
    
    print(f"\n✓ Answer generated!")
    return answer


# ============================================================================
# COMPLETE RAG PIPELINE
# ============================================================================
def run_rag_pipeline(query: str, k: int = 3) -> str:
    """
    Run the complete RAG pipeline for a single query.
    
    Args:
        query: User's question
        k: Number of documents to retrieve
        
    Returns:
        Generated answer
    """
    print("\n" + "=" * 70)
    print("RUNNING COMPLETE RAG PIPELINE")
    print("=" * 70)
    print(f"\n📝 Question: \"{query}\"")
    
    # Step 4: Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, k=k)
    
    # Step 5: Generate answer
    answer = generate_answer(query, retrieved_docs)
    
    print("\n" + "=" * 70)
    print("RAG PIPELINE RESULT")
    print("=" * 70)
    print(f"\n❓ Question: {query}")
    print(f"\n💡 Answer:\n{answer}")
    print("\n" + "=" * 70)
    
    return answer


# ============================================================================
# DEMONSTRATE DIFFERENT RETRIEVAL SCENARIOS
# ============================================================================
def demonstrate_retrieval_scenarios():
    """Demonstrate different retrieval configurations."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING RETRIEVAL SCENARIOS")
    print("=" * 70)
    
    # Scenario 1: Different k values
    print("\n[Scenario 1] Comparing Different K Values")
    print("-" * 70)
    
    query = "What are the key policies?"
    
    for k in [2, 4]:
        print(f"\n--- With k={k} ---")
        docs = vectorstore.similarity_search(query, k=k)
        print(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] {doc.metadata.get('source', 'Unknown')}")
    
    # Scenario 2: Retriever with configuration
    print("\n\n[Scenario 2] Using Retriever Interface")
    print("-" * 70)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    query2 = "What are the guidelines?"
    print(f"\nQuery: \"{query2}\"")
    results = retriever.invoke(query2)
    print(f"✓ Retrieved {len(results)} documents using retriever interface")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run the complete demonstration."""
    print("\n" + "=" * 70)
    print("DEMO 10: COMPLETE RAG PIPELINE")
    print("=" * 70)
    
    # Steps 1-3: Ingestion (Load, Chunk, Store)
    documents = load_documents()
    
    if not documents:
        print("\n⚠️  No documents loaded. Please add documents to the Documents/ folder.")
        return
    
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("\n⚠️  No chunks created.")
        return
    
    store_chunks(chunks)
    
    # Demonstrate retrieval scenarios
    demonstrate_retrieval_scenarios()
    
    # Steps 4-5: Complete RAG (Retrieve + Generate)
    print("\n\n" + "=" * 70)
    print("TESTING COMPLETE RAG PIPELINE")
    print("=" * 70)
    
    # Test queries
    queries = [
        "What is the remote work policy?",
        "What are the code review guidelines?",
        "Tell me about Python programming"
    ]
    
    for query in queries:
        print("\n")
        run_rag_pipeline(query, k=3)
        print("\n" + "-" * 70)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ COMPLETE RAG PIPELINE DEMONSTRATION FINISHED!")
    print("=" * 70)
    print("\n📋 Summary:")
    print(f"  1. Loaded {len(documents)} documents")
    print(f"  2. Created {len(chunks)} chunks")
    print(f"  3. Stored chunks with embeddings in {VECTOR_DB.upper()}")
    print(f"  4. Demonstrated retrieval with different k values")
    print(f"  5. Generated answers using {OPENAI_MODEL}")
    print("\n🎯 Key Concepts Demonstrated:")
    print("  • Document loading from multiple sources")
    print("  • Chunking with RecursiveCharacterTextSplitter")
    print("  • Vector embedding and storage")
    print("  • Similarity search and retrieval")
    print("  • RAG chain with LCEL")
    print("  • Answer generation with LLM")
    print("  • Config-driven vector database selection")
    print("=" * 70)


if __name__ == "__main__":
    main()
