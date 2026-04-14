"""
Demo 12: RAG Pipeline - FastAPI Service

This demo exposes a complete RAG (Retrieval-Augmented Generation) pipeline via REST API:
- Ingestion endpoints: Load and store documents (demo-09 functionality)
- Retrieval endpoints: Search and retrieve relevant chunks (demo-10 functionality)
- Generation endpoints: Generate answers using LLM with retrieved context (demo-11 functionality)

Supports two vector databases via configuration:
- ChromaDB (local, file-based)
- Pinecone (cloud-based)

Usage:
    # Start the server
    uvicorn main:app --reload --port 8000
    
    # Or use uv
    uv run uvicorn main:app --reload --port 8000
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import tempfile
import shutil

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_PREVIEW_LENGTH = 200

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0
)

# Initialize Vector Store
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
    
elif VECTOR_DB == "pinecone":
    from langchain_pinecone import PineconeVectorStore
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-policies-demo")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY required when VECTOR_DB=pinecone")
    
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

else:
    raise ValueError(f"Unsupported VECTOR_DB: {VECTOR_DB}")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="RAG Pipeline API",
    description="Complete RAG pipeline with ingestion, retrieval, and generation endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    vector_db: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int

class IngestTextRequest(BaseModel):
    text: str = Field(..., description="Text content to ingest", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Optional metadata")

class IngestTextResponse(BaseModel):
    status: str
    chunks_created: int
    message: str

class RetrievalRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    k: int = Field(default=4, ge=1, le=20, description="Number of results to retrieve")
    include_scores: bool = Field(default=False, description="Include relevance scores")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")

class RetrievalResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    count: int

class MMRRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    k: int = Field(default=4, ge=1, le=20, description="Number of diverse results")
    fetch_k: int = Field(default=20, ge=1, le=100, description="Number of candidates to fetch")

class GenerationRequest(BaseModel):
    query: str = Field(..., description="User question", min_length=1)
    k: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve")
    include_sources: bool = Field(default=True, description="Include source chunks in response")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature (0=deterministic, 2=creative)")

class GenerationResponse(BaseModel):
    query: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    context_count: int

class VerifyStoreResponse(BaseModel):
    status: str
    has_data: bool
    chunk_count: int
    sample_chunk: Optional[Dict[str, Any]] = None

class FileIngestResponse(BaseModel):
    status: str
    filename: str
    documents_loaded: int
    chunks_created: int
    message: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "RAG Pipeline API",
        "version": "1.0.0",
        "description": "FastAPI service combining demo-09 (ingestion), demo-10 (retrieval), and demo-11 (generation)",
        "endpoints": {
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            },
            "health": "GET /health",
            "ingestion": {
                "text": "POST /ingest/text",
                "file": "POST /ingest/file"
            },
            "retrieval": {
                "verify": "GET /retrieve/verify",
                "similarity": "POST /retrieve/similarity",
                "mmr": "POST /retrieve/mmr"
            },
            "generation": {
                "rag": "POST /generate/rag"
            }
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and configuration"""
    return HealthResponse(
        status="healthy",
        vector_db=VECTOR_DB.upper(),
        llm_model=OPENAI_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

# ============================================================================
# INGESTION ENDPOINTS (Demo-09 Functionality)
# ============================================================================

@app.post("/ingest/text", response_model=IngestTextResponse, tags=["Ingestion"])
async def ingest_text(request: IngestTextRequest):
    """
    Ingest text content into the vector store.
    
    **Demo-09 Functionality**: Load and store text as chunks with embeddings.
    
    - **text**: Raw text content to ingest
    - **metadata**: Optional metadata (source, type, author, etc.)
    
    Returns:
    - Number of chunks created
    - Status message
    """
    try:
        # Create document
        doc = Document(
            page_content=request.text,
            metadata=request.metadata or {"source": "api_text_input"}
        )
        
        # Chunk the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents([doc])
        
        # Add to vector store
        vectorstore.add_documents(chunks)
        
        return IngestTextResponse(
            status="success",
            chunks_created=len(chunks),
            message=f"Successfully ingested {len(chunks)} chunks into {VECTOR_DB.upper()}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/ingest/file", response_model=FileIngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(..., description="File to upload (PDF or TXT)")):
    """
    Ingest a file (PDF or TXT) into the vector store.
    
    **Demo-09 Functionality**: Load documents from files, chunk, and store with embeddings.
    
    - **file**: File to upload (PDF or TXT format)
    
    Returns:
    - Number of documents loaded
    - Number of chunks created
    - Status message
    """
    temp_file_path = None
    try:
        # Validate file type
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only PDF and TXT files are supported."
            )
        
        # Create temp file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        # Load document based on file type
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        else:  # .txt
            loader = TextLoader(temp_file_path)
        
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = file.filename
        
        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add to vector store
        vectorstore.add_documents(chunks)
        
        return FileIngestResponse(
            status="success",
            filename=file.filename,
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            message=f"Successfully ingested {file.filename} into {VECTOR_DB.upper()}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# ============================================================================
# RETRIEVAL ENDPOINTS (Demo-10 Functionality)
# ============================================================================

@app.get("/retrieve/verify", response_model=VerifyStoreResponse, tags=["Retrieval"])
async def verify_vector_store():
    """
    Verify that the vector store has data.
    
    **Demo-10 Functionality**: Check if vector store is populated before retrieval.
    
    Returns:
    - Data availability status
    - Approximate chunk count
    - Sample chunk (if available)
    """
    try:
        # Test query to check data
        test_results = vectorstore.similarity_search("test", k=1)
        
        if not test_results:
            return VerifyStoreResponse(
                status="empty",
                has_data=False,
                chunk_count=0,
                sample_chunk=None
            )
        
        # Get approximate count
        sample_results = vectorstore.similarity_search("document", k=100)
        doc_count = len(sample_results)
        
        # Prepare sample
        sample = None
        if sample_results:
            sample_doc = sample_results[0]
            sample = {
                "source": sample_doc.metadata.get('source', 'Unknown'),
                "content_preview": sample_doc.page_content[:CHUNK_PREVIEW_LENGTH],
                "metadata": sample_doc.metadata
            }
        
        return VerifyStoreResponse(
            status="ready",
            has_data=True,
            chunk_count=doc_count,
            sample_chunk=sample
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.post("/retrieve/similarity", response_model=RetrievalResponse, tags=["Retrieval"])
async def retrieve_similarity(request: RetrievalRequest):
    """
    Perform similarity search to retrieve relevant chunks.
    
    **Demo-10 Functionality**: Find the most relevant chunks using vector similarity.
    
    - **query**: Search query
    - **k**: Number of results (1-20)
    - **include_scores**: Include relevance scores (lower score = higher relevance)
    - **filter**: Optional metadata filter (e.g., {"source": "file.pdf"})
    
    Returns:
    - List of relevant chunks with content and metadata
    - Relevance scores (if requested)
    """
    try:
        if request.include_scores:
            results = vectorstore.similarity_search_with_score(
                request.query,
                k=request.k,
                filter=request.filter
            )
            
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH],
                    "metadata": doc.metadata,
                    "score": float(score),
                    "relevance": "high" if score < 0.6 else "medium" if score < 0.8 else "low"
                }
                formatted_results.append(result)
        else:
            results = vectorstore.similarity_search(
                request.query,
                k=request.k,
                filter=request.filter
            )
            
            formatted_results = [
                {
                    "content": doc.page_content,
                    "content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH],
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        
        return RetrievalResponse(
            query=request.query,
            results=formatted_results,
            count=len(formatted_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.post("/retrieve/mmr", response_model=RetrievalResponse, tags=["Retrieval"])
async def retrieve_mmr(request: MMRRequest):
    """
    Perform MMR (Maximum Marginal Relevance) search for diverse results.
    
    **Demo-10 Functionality**: Balance relevance with diversity to avoid redundant results.
    
    - **query**: Search query
    - **k**: Number of diverse results to return
    - **fetch_k**: Number of candidates to consider (higher = more diverse)
    
    Returns:
    - List of diverse relevant chunks
    """
    try:
        results = vectorstore.max_marginal_relevance_search(
            request.query,
            k=request.k,
            fetch_k=request.fetch_k
        )
        
        formatted_results = [
            {
                "content": doc.page_content,
                "content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH],
                "metadata": doc.metadata
            }
            for doc in results
        ]
        
        return RetrievalResponse(
            query=request.query,
            results=formatted_results,
            count=len(formatted_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MMR retrieval failed: {str(e)}")

# ============================================================================
# GENERATION ENDPOINTS (Demo-11 Functionality)
# ============================================================================

@app.post("/generate/rag", response_model=GenerationResponse, tags=["Generation"])
async def generate_rag_answer(request: GenerationRequest):
    """
    Generate an answer using RAG (Retrieval-Augmented Generation).
    
    **Demo-11 Functionality**: Complete RAG flow combining retrieval and generation.
    
    Steps:
    1. Retrieve relevant chunks from vector store (demo-10)
    2. Use chunks as context for LLM
    3. Generate answer based on context (demo-11)
    
    - **query**: User's question
    - **k**: Number of context chunks to retrieve (1-20)
    - **include_sources**: Include source chunks in response
    - **temperature**: LLM creativity (0=deterministic, 2=creative)
    
    Returns:
    - Generated answer
    - Source chunks used (if requested)
    - Context count
    """
    try:
        # Step 1: Retrieve relevant context
        retrieved_docs = vectorstore.similarity_search(request.query, k=request.k)
        
        if not retrieved_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please ingest documents first using /ingest endpoints."
            )
        
        # Step 2: Prepare context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Step 3: Create prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Guidelines:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- Cite specific parts of the context when relevant
- If the question cannot be answered from context, explain what information is missing"""),
            ("human", "{question}")
        ])
        
        # Step 4: Generate answer with specified temperature
        llm_with_temp = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=request.temperature
        )
        
        chain = prompt_template | llm_with_temp | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": request.query
        })
        
        # Step 5: Prepare sources
        sources = None
        if request.include_sources:
            sources = [
                {
                    "content": doc.page_content,
                    "content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH],
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ]
        
        return GenerationResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            context_count=len(retrieved_docs)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Print startup information"""
    print("\n" + "=" * 70)
    print("RAG PIPELINE FASTAPI SERVICE - STARTUP")
    print("=" * 70)
    print(f"Vector Database: {VECTOR_DB.upper()}")
    print(f"LLM Model: {OPENAI_MODEL}")
    print(f"Chunk Size: {CHUNK_SIZE} characters")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} characters")
    print("=" * 70)
    print("\n✓ API Server Ready!")
    print(f"✓ Interactive API docs: http://localhost:8000/docs")
    print(f"✓ Alternative docs: http://localhost:8000/redoc")
    print(f"✓ Health check: http://localhost:8000/health")
    print("=" * 70)
    print("\n📚 Demo Integration:")
    print("  • Ingestion: demo-09 functionality")
    print("  • Retrieval: demo-10 functionality")
    print("  • Generation: demo-11 functionality")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
