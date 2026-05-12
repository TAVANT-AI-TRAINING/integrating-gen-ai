"""
Demo 13: Chunking Strategies Showcase

Demonstrates 4 chunking strategies side-by-side so students can see concretely
how splitting decisions affect chunk count, size, and coherence:

  1. Fixed-size    — CharacterTextSplitter: hard cuts at N chars, simple and fast
  2. Recursive     — RecursiveCharacterTextSplitter: tries paragraph/sentence/word
                     boundaries before hard-cutting (LangChain default)
  3. Semantic      — SemanticChunker: groups sentences by topic using embedding
                     similarity; splits where topic shifts (slower, more coherent)
  4. Parent-Child  — Small child chunks indexed for precise retrieval; larger
                     parent chunks returned for richer LLM context

Key learning: The /chunk/compare endpoint shows all 4 strategies on the same
input text simultaneously, including processing time — so students can see the
cost/quality trade-off directly.

Usage:
    uv run uvicorn main:app --reload --port 8001
"""

import os
import time
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chunking_demo")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SEMANTIC_BREAKPOINT_TYPE = os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_TYPE", "percentile")
SEMANTIC_BREAKPOINT_AMOUNT = float(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT", "90"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHUNK_PREVIEW_LENGTH = 200

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

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

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR,
)

# Parent-child retriever uses its own child collection + in-memory docstore
child_vectorstore = Chroma(
    collection_name=f"{COLLECTION_NAME}_children",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR,
)
docstore = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=50,
    ),
    parent_splitter=RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=0,
    ),
)

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Chunking Strategies API",
    description="Compare 4 RAG chunking strategies: Fixed, Recursive, Semantic, Parent-Child",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

StrategyName = Literal["fixed", "recursive", "semantic", "parent_child"]


class ChunkStats(BaseModel):
    strategy: str
    description: str
    chunk_count: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    processing_time_ms: float
    chunks: List[str] = []


class ParentChildStats(BaseModel):
    strategy: str
    description: str
    parent_chunk_count: int
    child_chunk_count: int
    avg_parent_size: float
    avg_child_size: float
    processing_time_ms: float
    note: str


class CompareRequest(BaseModel):
    text: str = Field(..., description="Text to chunk", min_length=10)
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    include_content: bool = Field(default=False, description="Include full chunk text in response")


class CompareResponse(BaseModel):
    input_text_length: int
    chunk_size_used: int
    chunk_overlap_used: int
    fixed: ChunkStats
    recursive: ChunkStats
    semantic: ChunkStats
    parent_child: ParentChildStats
    comparison_summary: Dict[str, str]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to chunk", min_length=10)
    strategy: StrategyName = Field(default="recursive")
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    include_full_content: bool = Field(default=False)


class IngestTextRequest(BaseModel):
    text: str = Field(..., description="Text to ingest", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default={})
    strategy: StrategyName = Field(default="recursive", description="Chunking strategy to use")
    chunk_size: int = Field(default=CHUNK_SIZE, ge=100, le=4000)
    chunk_overlap: int = Field(default=CHUNK_OVERLAP, ge=0, le=500)


class IngestResponse(BaseModel):
    status: str
    strategy: str
    chunks_created: int
    message: str


class FileIngestResponse(BaseModel):
    status: str
    filename: str
    strategy: str
    documents_loaded: int
    chunks_created: int
    message: str


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    include_scores: bool = Field(default=False)


class GenerationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    include_sources: bool = Field(default=True)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


# ============================================================================
# HELPERS
# ============================================================================

def _chunk_stats(strategy: str, description: str, chunks: List[Document],
                 elapsed_ms: float, include_content: bool) -> ChunkStats:
    sizes = [len(c.page_content) for c in chunks]
    return ChunkStats(
        strategy=strategy,
        description=description,
        chunk_count=len(chunks),
        avg_chunk_size=round(sum(sizes) / len(sizes), 1) if sizes else 0,
        min_chunk_size=min(sizes) if sizes else 0,
        max_chunk_size=max(sizes) if sizes else 0,
        processing_time_ms=round(elapsed_ms, 1),
        chunks=[c.page_content for c in chunks] if include_content else [],
    )


def _apply_strategy(text: str, strategy: StrategyName,
                    chunk_size: int, chunk_overlap: int) -> List[Document]:
    doc = Document(page_content=text, metadata={"source": "api_input"})

    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents([doc])

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents([doc])

    if strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
        )
        return splitter.split_documents([doc])

    raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Chunking Strategies API",
        "version": "1.0.0",
        "description": "Compare 4 RAG chunking strategies side-by-side",
        "strategies": {
            "fixed": "CharacterTextSplitter — hard cuts at N chars",
            "recursive": "RecursiveCharacterTextSplitter — respects paragraph/sentence/word boundaries",
            "semantic": "SemanticChunker — splits at topic shifts detected via embeddings",
            "parent_child": "ParentDocumentRetriever — small child chunks for retrieval, large parents for context",
        },
        "endpoints": {
            "docs": "/docs",
            "chunk": {
                "compare": "POST /chunk/compare — all 4 strategies on same text",
                "analyze": "POST /chunk/analyze — deep stats for one strategy",
            },
            "ingest": {
                "text": "POST /ingest/text",
                "file": "POST /ingest/file",
            },
            "retrieve": {
                "verify": "GET /retrieve/verify",
                "similarity": "POST /retrieve/similarity",
            },
            "generate": "POST /generate/rag",
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    try:
        count = len(vectorstore.get()["ids"])
    except Exception:
        count = 0
    return {
        "status": "healthy",
        "llm_model": OPENAI_MODEL,
        "embedding_model": "text-embedding-3-small",
        "vector_store_chunks": count,
        "default_chunk_size": CHUNK_SIZE,
        "default_chunk_overlap": CHUNK_OVERLAP,
        "semantic_breakpoint_type": SEMANTIC_BREAKPOINT_TYPE,
        "semantic_breakpoint_amount": SEMANTIC_BREAKPOINT_AMOUNT,
        "parent_chunk_size": PARENT_CHUNK_SIZE,
        "child_chunk_size": CHILD_CHUNK_SIZE,
        "strategies_available": ["fixed", "recursive", "semantic", "parent_child"],
    }


# ----------------------------------------------------------------------------
# CHUNK COMPARISON ENDPOINTS
# ----------------------------------------------------------------------------

@app.post("/chunk/compare", response_model=CompareResponse, tags=["Chunking"])
async def compare_chunking_strategies(request: CompareRequest):
    """
    Run all 4 chunking strategies on the same input text and return side-by-side stats.

    This is the key teaching endpoint. Notice:
    - **fixed** and **recursive** are nearly instant (no API calls)
    - **semantic** is slower — it embeds every sentence to detect topic shifts
    - processing_time_ms per strategy makes this cost visible
    - chunk_count varies significantly: semantic produces fewer, more coherent chunks

    Set `include_content=true` to also return the actual chunk text.
    """
    text = request.text
    cs = request.chunk_size
    co = request.chunk_overlap
    doc = Document(page_content=text, metadata={"source": "compare_input"})

    # --- Fixed ---
    t0 = time.perf_counter()
    fixed_chunks = CharacterTextSplitter(
        separator="\n\n", chunk_size=cs, chunk_overlap=co
    ).split_documents([doc])
    fixed_ms = (time.perf_counter() - t0) * 1000

    # --- Recursive ---
    t0 = time.perf_counter()
    recursive_chunks = RecursiveCharacterTextSplitter(
        chunk_size=cs, chunk_overlap=co
    ).split_documents([doc])
    recursive_ms = (time.perf_counter() - t0) * 1000

    # --- Semantic (makes embedding API calls) ---
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        t0 = time.perf_counter()
        semantic_chunks = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
        ).split_documents([doc])
        semantic_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic chunking failed: {str(e)}")

    # --- Parent-Child (preview only — no indexing, just size simulation) ---
    t0 = time.perf_counter()
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=50)
    parents = parent_splitter.split_documents([doc])
    children = []
    for p in parents:
        children.extend(child_splitter.split_documents([p]))
    pc_ms = (time.perf_counter() - t0) * 1000

    parent_sizes = [len(p.page_content) for p in parents]
    child_sizes = [len(c.page_content) for c in children]

    include = request.include_content
    fixed_stat = _chunk_stats("fixed", "CharacterTextSplitter — splits at '\\n\\n', then hard-cuts at chunk_size", fixed_chunks, fixed_ms, include)
    recursive_stat = _chunk_stats("recursive", "RecursiveCharacterTextSplitter — tries \\n\\n → \\n → sentence → word boundaries", recursive_chunks, recursive_ms, include)
    semantic_stat = _chunk_stats("semantic", f"SemanticChunker ({SEMANTIC_BREAKPOINT_TYPE} {SEMANTIC_BREAKPOINT_AMOUNT}) — splits where embedding similarity drops (topic shift)", semantic_chunks, semantic_ms, include)

    pc_stat = ParentChildStats(
        strategy="parent_child",
        description=f"ParentDocumentRetriever — child chunks ({CHILD_CHUNK_SIZE} chars) indexed for retrieval; parent chunks ({PARENT_CHUNK_SIZE} chars) returned to LLM",
        parent_chunk_count=len(parents),
        child_chunk_count=len(children),
        avg_parent_size=round(sum(parent_sizes) / len(parent_sizes), 1) if parent_sizes else 0,
        avg_child_size=round(sum(child_sizes) / len(child_sizes), 1) if child_sizes else 0,
        processing_time_ms=round(pc_ms, 1),
        note="This is a size preview only. Use POST /ingest/text or /ingest/file with strategy='parent_child' to build the full index.",
    )

    counts = {
        "fixed": fixed_stat.chunk_count,
        "recursive": recursive_stat.chunk_count,
        "semantic": semantic_stat.chunk_count,
    }
    summary = {
        "most_chunks": max(counts, key=counts.get),
        "fewest_chunks": min(counts, key=counts.get),
        "most_uniform": "fixed",
        "most_coherent_estimated": "semantic",
        "fastest": "fixed" if fixed_ms <= recursive_ms else "recursive",
        "slowest": "semantic (embedding API calls per sentence)",
        "observation": (
            f"Semantic chunking produced {semantic_stat.chunk_count} coherent chunks vs "
            f"{fixed_stat.chunk_count} fixed chunks from the same text. "
            f"It took {semantic_ms:.0f} ms vs {fixed_ms:.0f} ms for fixed-size."
        ),
    }

    return CompareResponse(
        input_text_length=len(text),
        chunk_size_used=cs,
        chunk_overlap_used=co,
        fixed=fixed_stat,
        recursive=recursive_stat,
        semantic=semantic_stat,
        parent_child=pc_stat,
        comparison_summary=summary,
    )


@app.post("/chunk/analyze", tags=["Chunking"])
async def analyze_single_strategy(request: AnalyzeRequest):
    """
    Deep analysis of one chosen chunking strategy.

    Returns per-chunk sizes, a histogram bucket, and optionally the full chunk text.
    Useful for tuning chunk_size and chunk_overlap before committing to a strategy.
    """
    try:
        t0 = time.perf_counter()
        if request.strategy == "parent_child":
            doc = Document(page_content=request.text, metadata={"source": "analyze_input"})
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=0)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=50)
            parents = parent_splitter.split_documents([doc])
            children_all = []
            for p in parents:
                children_all.extend(child_splitter.split_documents([p]))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "strategy": "parent_child",
                "processing_time_ms": round(elapsed_ms, 1),
                "parents": {
                    "count": len(parents),
                    "sizes": [len(p.page_content) for p in parents],
                    "chunks": [p.page_content for p in parents] if request.include_full_content else [],
                },
                "children": {
                    "count": len(children_all),
                    "sizes": [len(c.page_content) for c in children_all],
                    "chunks": [c.page_content for c in children_all] if request.include_full_content else [],
                },
            }

        chunks = _apply_strategy(request.text, request.strategy, request.chunk_size, request.chunk_overlap)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        sizes = [len(c.page_content) for c in chunks]

        return {
            "strategy": request.strategy,
            "chunk_size_param": request.chunk_size,
            "chunk_overlap_param": request.chunk_overlap,
            "processing_time_ms": round(elapsed_ms, 1),
            "chunk_count": len(chunks),
            "avg_size": round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "size_distribution": sizes,
            "chunks": [c.page_content for c in chunks] if request.include_full_content else [c.page_content[:CHUNK_PREVIEW_LENGTH] + "..." for c in chunks],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ----------------------------------------------------------------------------
# INGESTION ENDPOINTS
# ----------------------------------------------------------------------------

@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestTextRequest):
    """
    Ingest text using a chosen chunking strategy.

    - **strategy**: fixed | recursive | semantic | parent_child
    - **chunk_size / chunk_overlap**: override defaults from .env
    """
    try:
        doc = Document(
            page_content=request.text,
            metadata=request.metadata or {"source": "api_text_input"}
        )

        if request.strategy == "parent_child":
            parent_retriever.add_documents([doc])
            child_count = len(child_vectorstore.get()["ids"])
            return IngestResponse(
                status="success",
                strategy="parent_child",
                chunks_created=child_count,
                message=f"Ingested with parent-child strategy. Child chunks in index: {child_count}",
            )

        chunks = _apply_strategy(request.text, request.strategy, request.chunk_size, request.chunk_overlap)
        vectorstore.add_documents(chunks)

        return IngestResponse(
            status="success",
            strategy=request.strategy,
            chunks_created=len(chunks),
            message=f"Ingested {len(chunks)} chunks using '{request.strategy}' strategy",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/file", response_model=FileIngestResponse, tags=["Ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    strategy: StrategyName = "recursive",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """
    Upload a PDF or TXT file and ingest with a chosen chunking strategy.

    Use the **strategy** query parameter: fixed | recursive | semantic | parent_child
    """
    temp_path = None
    try:
        if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())

        loader = PyPDFLoader(temp_path) if file.filename.endswith(".pdf") else TextLoader(temp_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = file.filename

        if strategy == "parent_child":
            parent_retriever.add_documents(documents)
            child_count = len(child_vectorstore.get()["ids"])
            return FileIngestResponse(
                status="success",
                filename=file.filename,
                strategy="parent_child",
                documents_loaded=len(documents),
                chunks_created=child_count,
                message=f"Ingested '{file.filename}' with parent-child strategy. Child chunks: {child_count}",
            )

        all_chunks = []
        for doc in documents:
            all_chunks.extend(_apply_strategy(doc.page_content, strategy, chunk_size, chunk_overlap))
        vectorstore.add_documents(all_chunks)

        return FileIngestResponse(
            status="success",
            filename=file.filename,
            strategy=strategy,
            documents_loaded=len(documents),
            chunks_created=len(all_chunks),
            message=f"Ingested '{file.filename}': {len(all_chunks)} chunks via '{strategy}' strategy",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ----------------------------------------------------------------------------
# RETRIEVAL ENDPOINTS
# ----------------------------------------------------------------------------

@app.get("/retrieve/verify", tags=["Retrieval"])
async def verify_store():
    """Check that the vector store contains data."""
    try:
        data = vectorstore.get()
        count = len(data["ids"])
        child_data = child_vectorstore.get()
        child_count = len(child_data["ids"])

        sample = None
        if count > 0:
            sample = {
                "content_preview": data["documents"][0][:CHUNK_PREVIEW_LENGTH],
                "metadata": data["metadatas"][0],
            }

        return {
            "status": "ready" if count > 0 else "empty",
            "has_data": count > 0,
            "chunk_count": count,
            "child_chunk_count": child_count,
            "sample_chunk": sample,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/retrieve/similarity", tags=["Retrieval"])
async def retrieve_similarity(request: RetrievalRequest):
    """Standard dense similarity search against ingested chunks."""
    try:
        if request.include_scores:
            results = vectorstore.similarity_search_with_score(request.query, k=request.k)
            docs = [
                {
                    "rank": i + 1,
                    "content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH],
                    "metadata": doc.metadata,
                    "score": float(score),
                }
                for i, (doc, score) in enumerate(results)
            ]
        else:
            results = vectorstore.similarity_search(request.query, k=request.k)
            docs = [
                {
                    "rank": i + 1,
                    "content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH],
                    "metadata": doc.metadata,
                }
                for i, doc in enumerate(results)
            ]
        return {"query": request.query, "count": len(docs), "results": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


# ----------------------------------------------------------------------------
# GENERATION ENDPOINT
# ----------------------------------------------------------------------------

@app.post("/generate/rag", tags=["Generation"])
async def generate_rag(request: GenerationRequest):
    """Full RAG answer using the ingested chunks."""
    try:
        retrieved = vectorstore.similarity_search(request.query, k=request.k)
        if not retrieved:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Ingest documents first via /ingest/text or /ingest/file."
            )

        context = "\n\n".join(doc.page_content for doc in retrieved)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Answer the question using ONLY the provided context.

Context:
{context}

Rules:
- Answer based ONLY on the context above
- If the context lacks the information, say so clearly
- Be concise and cite relevant parts of the context"""),
            ("human", "{question}"),
        ])

        chain = prompt | ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=request.temperature,
        ) | StrOutputParser()

        answer = chain.invoke({"context": context, "question": request.query})

        sources = None
        if request.include_sources:
            sources = [
                {"content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH], "metadata": doc.metadata}
                for doc in retrieved
            ]

        return {
            "query": request.query,
            "answer": answer,
            "context_count": len(retrieved),
            "sources": sources,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 70)
    print("DEMO-13: CHUNKING STRATEGIES SHOWCASE — STARTUP")
    print("=" * 70)
    print(f"LLM Model:           {OPENAI_MODEL}")
    print(f"Embedding Model:     text-embedding-3-small")
    print(f"Default Chunk Size:  {CHUNK_SIZE} chars")
    print(f"Default Overlap:     {CHUNK_OVERLAP} chars")
    print(f"Semantic Threshold:  {SEMANTIC_BREAKPOINT_TYPE} @ {SEMANTIC_BREAKPOINT_AMOUNT}")
    print(f"Parent/Child Sizes:  {PARENT_CHUNK_SIZE} / {CHILD_CHUNK_SIZE} chars")
    print("=" * 70)
    print("\n✓ API Server Ready!")
    print("✓ Interactive docs: http://localhost:8001/docs")
    print("✓ Health check:     http://localhost:8001/health")
    print("=" * 70)
    print("\n📚 Key Endpoint: POST /chunk/compare")
    print("   → Send any text and see all 4 strategies side-by-side")
    print("   → Notice: semantic strategy is slower (embedding API calls)")
    print("   → Compare chunk_count and processing_time_ms across strategies")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
