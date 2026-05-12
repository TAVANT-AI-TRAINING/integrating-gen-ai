"""
Demo 14: Hybrid Search — BM25 + Dense Vector Retrieval

Pure dense (vector) search is great for semantic similarity but fails on:
  - Exact terms: product codes, error messages, names, acronyms
  - Rare words not well-represented in embedding training data
  - Short queries with little semantic signal (e.g. "GDPR Article 17")

BM25 (sparse / keyword search) fixes those cases but fails on paraphrasing
and synonyms ("heart attack" vs "myocardial infarction").

Hybrid search fuses both using Reciprocal Rank Fusion (RRF):
  RRF_score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d))

The key teaching endpoint is POST /retrieve/compare — it runs the same query
through all three modes side-by-side and shows which chunks each retriever
found that the others missed (overlap_analysis).

Usage:
    uv run uvicorn main:app --reload --port 8002
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hybrid_search_demo")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_K = int(os.getenv("DEFAULT_K", "4"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.4"))
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

# BM25 index is in-memory. Rehydrated from ChromaDB on startup so it survives
# server restarts as long as ChromaDB data persists.
all_chunks: List[Document] = []


def _rehydrate_bm25():
    """Rebuild the in-memory BM25 chunk list from the persisted ChromaDB store."""
    global all_chunks
    data = vectorstore.get(include=["documents", "metadatas"])
    all_chunks = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]


def _build_bm25_retriever(k: int) -> BM25Retriever:
    if not all_chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed. Ingest documents first via /ingest/text or /ingest/file."
        )
    retriever = BM25Retriever.from_documents(all_chunks)
    retriever.k = k
    return retriever


def _doc_fingerprint(doc: Document) -> str:
    """Short fingerprint for overlap analysis."""
    return doc.page_content[:80]


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Hybrid Search API",
    description="Compare dense, sparse (BM25), and hybrid retrieval side-by-side",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

RetrievalMode = Literal["dense", "sparse", "hybrid"]


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default={})
    chunk_size: int = Field(default=CHUNK_SIZE, ge=100, le=4000)
    chunk_overlap: int = Field(default=CHUNK_OVERLAP, ge=0, le=500)


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    bm25_total_docs: int
    message: str


class FileIngestResponse(BaseModel):
    status: str
    filename: str
    documents_loaded: int
    chunks_created: int
    bm25_total_docs: int
    message: str


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    include_content: bool = Field(default=True)


class HybridRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    dense_weight: float = Field(default=DENSE_WEIGHT, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=SPARSE_WEIGHT, ge=0.0, le=1.0)
    include_content: bool = Field(default=True)


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    dense_weight: float = Field(default=DENSE_WEIGHT, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=SPARSE_WEIGHT, ge=0.0, le=1.0)
    include_content: bool = Field(default=True)


class GenerationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    retrieval_mode: RetrievalMode = Field(default="hybrid")
    dense_weight: float = Field(default=DENSE_WEIGHT, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=SPARSE_WEIGHT, ge=0.0, le=1.0)
    include_sources: bool = Field(default=True)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


# ============================================================================
# HELPERS
# ============================================================================

def _format_docs(docs: List[Document], include_content: bool) -> List[Dict[str, Any]]:
    return [
        {
            "rank": i + 1,
            "content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH],
            "content": doc.page_content if include_content else None,
            "metadata": doc.metadata,
        }
        for i, doc in enumerate(docs)
    ]


def _overlap_analysis(
    dense: List[Document],
    sparse: List[Document],
    hybrid: List[Document],
) -> Dict[str, Any]:
    def fp_set(docs):
        return {_doc_fingerprint(d) for d in docs}

    d = fp_set(dense)
    s = fp_set(sparse)
    h = fp_set(hybrid)

    return {
        "dense_sparse_overlap_count": len(d & s),
        "dense_hybrid_overlap_count": len(d & h),
        "sparse_hybrid_overlap_count": len(s & h),
        "all_three_overlap_count": len(d & s & h),
        "unique_to_dense_count": len(d - s - h),
        "unique_to_sparse_count": len(s - d - h),
        "unique_to_hybrid_count": len(h - d - s),
        "insight": (
            "Hybrid retrieved chunks that neither dense nor sparse found alone — "
            "this is RRF fusion filling keyword + semantic gaps."
            if (h - d - s)
            else "All hybrid results also appeared in dense or sparse results — "
                 "RRF reranked them rather than adding new ones."
        ),
    }


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Hybrid Search API",
        "version": "1.0.0",
        "description": "Compare dense, sparse (BM25), and hybrid (RRF) retrieval for the same query",
        "how_it_works": {
            "dense": "OpenAI text-embedding-3-small — captures semantic meaning",
            "sparse": "BM25 (Okapi BM25 via rank-bm25) — captures exact keyword matches",
            "hybrid": "EnsembleRetriever (RRF fusion) — combines both ranked lists",
            "rrf_formula": "score(d) = 1/(k+rank_dense) + 1/(k+rank_bm25), k=60",
        },
        "endpoints": {
            "docs": "/docs",
            "ingest": {"text": "POST /ingest/text", "file": "POST /ingest/file"},
            "retrieve": {
                "verify": "GET /retrieve/verify",
                "dense": "POST /retrieve/dense",
                "sparse": "POST /retrieve/sparse",
                "hybrid": "POST /retrieve/hybrid",
                "compare": "POST /retrieve/compare  ← KEY endpoint",
            },
            "generate": "POST /generate/rag",
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    try:
        chroma_count = len(vectorstore.get()["ids"])
    except Exception:
        chroma_count = 0
    return {
        "status": "healthy",
        "llm_model": OPENAI_MODEL,
        "embedding_model": "text-embedding-3-small",
        "chroma_chunk_count": chroma_count,
        "bm25_chunk_count": len(all_chunks),
        "default_dense_weight": DENSE_WEIGHT,
        "default_sparse_weight": SPARSE_WEIGHT,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


# ----------------------------------------------------------------------------
# INGESTION
# ----------------------------------------------------------------------------

@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestTextRequest):
    """
    Ingest text into both ChromaDB (dense) and the in-memory BM25 index (sparse).

    Both indexes are updated atomically so that all three retrieval modes
    always query the same document set.
    """
    try:
        doc = Document(
            page_content=request.text,
            metadata=request.metadata or {"source": "api_text_input"}
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        chunks = splitter.split_documents([doc])

        vectorstore.add_documents(chunks)
        all_chunks.extend(chunks)

        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            bm25_total_docs=len(all_chunks),
            message=f"Ingested {len(chunks)} chunks. BM25 index now has {len(all_chunks)} docs.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/file", response_model=FileIngestResponse, tags=["Ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """Upload and ingest a PDF or TXT file into both dense and BM25 indexes."""
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

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)

        vectorstore.add_documents(chunks)
        all_chunks.extend(chunks)

        return FileIngestResponse(
            status="success",
            filename=file.filename,
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            bm25_total_docs=len(all_chunks),
            message=f"Ingested '{file.filename}': {len(chunks)} chunks added.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ----------------------------------------------------------------------------
# RETRIEVAL
# ----------------------------------------------------------------------------

@app.get("/retrieve/verify", tags=["Retrieval"])
async def verify_store():
    """Verify both ChromaDB and BM25 indexes are populated."""
    try:
        data = vectorstore.get()
        chroma_count = len(data["ids"])
        sample = None
        if chroma_count > 0:
            sample = {
                "content_preview": data["documents"][0][:CHUNK_PREVIEW_LENGTH],
                "metadata": data["metadatas"][0],
            }
        return {
            "status": "ready" if chroma_count > 0 else "empty",
            "has_data": chroma_count > 0,
            "chroma_chunk_count": chroma_count,
            "bm25_chunk_count": len(all_chunks),
            "bm25_in_sync": chroma_count == len(all_chunks),
            "sample_chunk": sample,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/retrieve/dense", tags=["Retrieval"])
async def retrieve_dense(request: RetrievalRequest):
    """Dense-only retrieval using vector similarity (OpenAI embeddings)."""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.k})
        docs = retriever.invoke(request.query)
        return {
            "mode": "dense",
            "description": "Vector similarity search — captures semantic meaning",
            "query": request.query,
            "count": len(docs),
            "results": _format_docs(docs, request.include_content),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dense retrieval failed: {str(e)}")


@app.post("/retrieve/sparse", tags=["Retrieval"])
async def retrieve_sparse(request: RetrievalRequest):
    """Sparse-only retrieval using BM25 keyword matching."""
    try:
        retriever = _build_bm25_retriever(request.k)
        docs = retriever.invoke(request.query)
        return {
            "mode": "sparse",
            "description": "BM25 keyword search — captures exact term matches",
            "query": request.query,
            "count": len(docs),
            "results": _format_docs(docs, request.include_content),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sparse retrieval failed: {str(e)}")


@app.post("/retrieve/hybrid", tags=["Retrieval"])
async def retrieve_hybrid(request: HybridRetrievalRequest):
    """
    Hybrid retrieval: EnsembleRetriever (RRF fusion of dense + BM25).

    Tune dense_weight and sparse_weight to balance semantic vs keyword precision.
    Both weights must sum to 1.0.
    """
    try:
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": request.k})
        bm25_retriever = _build_bm25_retriever(request.k)
        hybrid = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[request.dense_weight, request.sparse_weight],
        )
        docs = hybrid.invoke(request.query)[:request.k]
        return {
            "mode": "hybrid",
            "description": f"EnsembleRetriever RRF fusion — dense weight={request.dense_weight}, sparse weight={request.sparse_weight}",
            "query": request.query,
            "count": len(docs),
            "results": _format_docs(docs, request.include_content),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid retrieval failed: {str(e)}")


@app.post("/retrieve/compare", tags=["Retrieval"])
async def compare_retrieval_modes(request: CompareRequest):
    """
    Run the same query through dense, sparse (BM25), and hybrid retrieval modes.

    The **overlap_analysis** section is the key teaching element — it shows:
    - Which chunks appeared in all three modes (high confidence)
    - Which chunks only dense found (semantic-only matches)
    - Which chunks only BM25 found (keyword-only matches)
    - Which chunks only hybrid found (RRF surfaced, others missed)

    Try these query types to see the difference:
    - Semantic query: "What are the consequences of missing a project deadline?"
      → Dense wins; BM25 struggles on paraphrasing
    - Keyword query: "two-factor authentication"
      → BM25 wins; dense may miss exact term
    - Mixed query: "TFA security requirements minimum characters"
      → Hybrid wins
    """
    try:
        k = request.k

        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        dense_docs = dense_retriever.invoke(request.query)

        bm25_retriever = _build_bm25_retriever(k)
        sparse_docs = bm25_retriever.invoke(request.query)

        hybrid = EnsembleRetriever(
            retrievers=[
                vectorstore.as_retriever(search_kwargs={"k": k}),
                _build_bm25_retriever(k),
            ],
            weights=[request.dense_weight, request.sparse_weight],
        )
        hybrid_docs = hybrid.invoke(request.query)[:k]

        return {
            "query": request.query,
            "k": k,
            "dense_weight": request.dense_weight,
            "sparse_weight": request.sparse_weight,
            "results": {
                "dense": {
                    "mode": "dense",
                    "description": "Vector similarity (OpenAI text-embedding-3-small)",
                    "count": len(dense_docs),
                    "documents": _format_docs(dense_docs, request.include_content),
                },
                "sparse": {
                    "mode": "sparse",
                    "description": "BM25 keyword search (Okapi BM25 via rank-bm25)",
                    "count": len(sparse_docs),
                    "documents": _format_docs(sparse_docs, request.include_content),
                },
                "hybrid": {
                    "mode": "hybrid",
                    "description": f"EnsembleRetriever RRF (dense {request.dense_weight} + sparse {request.sparse_weight})",
                    "count": len(hybrid_docs),
                    "documents": _format_docs(hybrid_docs, request.include_content),
                },
            },
            "overlap_analysis": _overlap_analysis(dense_docs, sparse_docs, hybrid_docs),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compare retrieval failed: {str(e)}")


# ----------------------------------------------------------------------------
# GENERATION
# ----------------------------------------------------------------------------

@app.post("/generate/rag", tags=["Generation"])
async def generate_rag(request: GenerationRequest):
    """
    RAG answer using the chosen retrieval mode.

    - **retrieval_mode**: dense | sparse | hybrid (default: hybrid)
    - Compare answer quality across modes for the same question
    """
    try:
        if request.retrieval_mode == "dense":
            docs = vectorstore.as_retriever(search_kwargs={"k": request.k}).invoke(request.query)
        elif request.retrieval_mode == "sparse":
            docs = _build_bm25_retriever(request.k).invoke(request.query)
        else:
            hybrid = EnsembleRetriever(
                retrievers=[
                    vectorstore.as_retriever(search_kwargs={"k": request.k}),
                    _build_bm25_retriever(request.k),
                ],
                weights=[request.dense_weight, request.sparse_weight],
            )
            docs = hybrid.invoke(request.query)[:request.k]

        if not docs:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Ingest documents first."
            )

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Answer using ONLY the provided context.

Context:
{context}

Rules:
- Answer based ONLY on the context above
- If context lacks the information, say so clearly
- Be concise and cite relevant parts"""),
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
                for doc in docs
            ]

        return {
            "query": request.query,
            "retrieval_mode": request.retrieval_mode,
            "context_count": len(docs),
            "answer": answer,
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
    _rehydrate_bm25()
    print("\n" + "=" * 70)
    print("DEMO-14: HYBRID SEARCH — STARTUP")
    print("=" * 70)
    print(f"LLM Model:           {OPENAI_MODEL}")
    print(f"Embedding Model:     text-embedding-3-small")
    print(f"Dense Weight:        {DENSE_WEIGHT}")
    print(f"Sparse Weight:       {SPARSE_WEIGHT}")
    print(f"BM25 docs reloaded:  {len(all_chunks)}")
    print("=" * 70)
    print("\n✓ API Server Ready!")
    print("✓ Interactive docs: http://localhost:8002/docs")
    print("✓ Health check:     http://localhost:8002/health")
    print("=" * 70)
    print("\n📚 Key Endpoint: POST /retrieve/compare")
    print("   → Same query through dense + BM25 + hybrid side-by-side")
    print("   → overlap_analysis shows what each mode found uniquely")
    print("   → Try keyword queries (BM25 wins) vs semantic queries (dense wins)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
