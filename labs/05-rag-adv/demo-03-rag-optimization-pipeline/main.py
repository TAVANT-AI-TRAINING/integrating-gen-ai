"""
Demo 15: Full RAG Optimization Pipeline

Demonstrates the complete "optimization ladder" from the course guide:

  Stage 1 (baseline)        → Fixed-size chunking + dense-only retrieval
  Stage 2 (semantic)        → SemanticChunker + dense-only retrieval
  Stage 3 (hybrid)          → SemanticChunker + EnsembleRetriever (BM25 + dense)
  Stage 4 (reranked)        → SemanticChunker + Hybrid (fetch 20) + CrossEncoderReranker (top 4)

The key endpoint POST /optimize/evaluate runs an embedded 8-query golden dataset
through all 4 stages and returns Recall@K and MRR per stage — making the
quality improvement at each rung measurable and concrete.

Architecture:
  - Dual ChromaDB collections: pipeline_baseline (fixed chunks) + pipeline_semantic (semantic chunks)
  - BM25 is rebuilt from ChromaDB on startup (in-memory, survives restarts as long as ChromaDB persists)
  - Cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) downloads from HuggingFace
    on first use (~80MB); cached under ~/.cache/huggingface/

Usage:
    uv run uvicorn main:app --reload --port 8003

Note: First startup will download the cross-encoder model. Ensure you have a
      network connection for the initial run.
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
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_BASELINE = os.getenv("COLLECTION_NAME_BASELINE", "pipeline_baseline")
COLLECTION_SEMANTIC = os.getenv("COLLECTION_NAME_SEMANTIC", "pipeline_semantic")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SEMANTIC_BREAKPOINT_TYPE = os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_TYPE", "percentile")
SEMANTIC_BREAKPOINT_AMOUNT = float(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT", "90"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.4"))
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
INITIAL_FETCH_K = int(os.getenv("INITIAL_FETCH_K", "20"))
DEFAULT_K = int(os.getenv("DEFAULT_K", "4"))
CHUNK_PREVIEW_LENGTH = 200

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# ============================================================================
# GOLDEN DATASET
# Built from the content of Documents/guidelines.txt and Documents/policy.txt
# Relevance is checked via keyword overlap (at least 2 of the listed keywords
# must appear in the retrieved chunk). This is transparent in all API responses.
# ============================================================================
GOLDEN_DATASET = [
    {
        "id": "q1",
        "query": "How many days per week can employees work remotely?",
        "relevant_keywords": ["remote", "3 days", "remotely"],
        "hint": "Section 1 of guidelines.txt",
    },
    {
        "id": "q2",
        "query": "What health benefits does the company provide?",
        "relevant_keywords": ["health insurance", "medical", "dental", "vision"],
        "hint": "Employee Benefits in policy.txt",
    },
    {
        "id": "q3",
        "query": "How many vacation days do employees receive per year?",
        "relevant_keywords": ["vacation", "15 days", "per year"],
        "hint": "Leave Policy in policy.txt",
    },
    {
        "id": "q4",
        "query": "What must happen before code changes can be merged?",
        "relevant_keywords": ["code review", "peer review", "merge"],
        "hint": "Section 2 of guidelines.txt",
    },
    {
        "id": "q5",
        "query": "What are the password and authentication security requirements?",
        "relevant_keywords": ["password", "12 characters", "two-factor"],
        "hint": "Section 3 of guidelines.txt",
    },
    {
        "id": "q6",
        "query": "How long is parental leave and is it paid?",
        "relevant_keywords": ["parental leave", "12 weeks", "paid"],
        "hint": "Leave Policy in policy.txt",
    },
    {
        "id": "q7",
        "query": "What is the annual training and development budget per employee?",
        "relevant_keywords": ["training budget", "2,000", "annual"],
        "hint": "Professional Development in policy.txt",
    },
    {
        "id": "q8",
        "query": "What are the guidelines for running effective meetings?",
        "relevant_keywords": ["meeting", "30 minutes", "agenda"],
        "hint": "Section 4 of guidelines.txt",
    },
]


def _is_relevant(chunk_text: str, keywords: List[str]) -> bool:
    """A chunk is relevant if it contains at least 2 of the query's keywords."""
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches >= 2


def _recall_at_k(docs: List[Document], keywords: List[str], k: int) -> float:
    top_k = docs[:k]
    found = any(_is_relevant(doc.page_content, keywords) for doc in top_k)
    return 1.0 if found else 0.0


def _mrr(docs: List[Document], keywords: List[str]) -> float:
    for rank, doc in enumerate(docs, start=1):
        if _is_relevant(doc.page_content, keywords):
            return 1.0 / rank
    return 0.0


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

# Dual ChromaDB collections
baseline_vectorstore = Chroma(
    collection_name=COLLECTION_BASELINE,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR,
)

semantic_vectorstore = Chroma(
    collection_name=COLLECTION_SEMANTIC,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR,
)

# Cross-encoder (downloads ~80MB model on first use, cached in ~/.cache/huggingface)
print("Loading cross-encoder model (downloads on first run)...")
_cross_encoder_model = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
cross_encoder_compressor = CrossEncoderReranker(
    model=_cross_encoder_model,
    top_n=RERANK_TOP_N,
)

# BM25 in-memory chunk lists (rehydrated from ChromaDB on startup)
baseline_chunks: List[Document] = []
semantic_chunks: List[Document] = []


def _rehydrate_chunks():
    global baseline_chunks, semantic_chunks
    for store, bucket in [(baseline_vectorstore, "baseline"), (semantic_vectorstore, "semantic")]:
        data = store.get(include=["documents", "metadatas"])
        chunks = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
        if bucket == "baseline":
            baseline_chunks[:] = chunks
        else:
            semantic_chunks[:] = chunks


def _build_bm25(chunks: List[Document], k: int) -> BM25Retriever:
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed in this collection. Ingest documents first."
        )
    r = BM25Retriever.from_documents(chunks)
    r.k = k
    return r


# ============================================================================
# PIPELINE STAGES
# ============================================================================

def _run_stage_baseline(query: str, k: int) -> List[Document]:
    return baseline_vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)


def _run_stage_semantic(query: str, k: int) -> List[Document]:
    return semantic_vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)


def _run_stage_hybrid(query: str, k: int, dense_w: float, sparse_w: float) -> List[Document]:
    hybrid = EnsembleRetriever(
        retrievers=[
            semantic_vectorstore.as_retriever(search_kwargs={"k": k}),
            _build_bm25(semantic_chunks, k),
        ],
        weights=[dense_w, sparse_w],
    )
    return hybrid.invoke(query)[:k]


def _run_stage_reranked(query: str, fetch_k: int, top_n: int, dense_w: float, sparse_w: float) -> List[Document]:
    base_hybrid = EnsembleRetriever(
        retrievers=[
            semantic_vectorstore.as_retriever(search_kwargs={"k": fetch_k}),
            _build_bm25(semantic_chunks, fetch_k),
        ],
        weights=[dense_w, sparse_w],
    )
    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=cross_encoder_compressor,
        base_retriever=base_hybrid,
    )
    return reranking_retriever.invoke(query)


def _format_docs(docs: List[Document], include_content: bool = False) -> List[Dict[str, Any]]:
    return [
        {
            "rank": i + 1,
            "content_preview": doc.page_content[:CHUNK_PREVIEW_LENGTH],
            "content": doc.page_content if include_content else None,
            "metadata": doc.metadata,
        }
        for i, doc in enumerate(docs)
    ]


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="RAG Optimization Pipeline API",
    description="Full optimization ladder: fixed chunking → semantic → hybrid → re-ranking with Recall@K and MRR evaluation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

PipelineStage = Literal["baseline", "semantic_chunking", "hybrid_search", "reranked"]


class IngestResponse(BaseModel):
    status: str
    collection: str
    strategy: str
    chunks_created: int
    message: str


class FileIngestResponse(BaseModel):
    status: str
    filename: str
    baseline_chunks: int
    semantic_chunks_count: int
    message: str


class EvaluateRequest(BaseModel):
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    dense_weight: float = Field(default=DENSE_WEIGHT, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=SPARSE_WEIGHT, ge=0.0, le=1.0)
    initial_fetch_k: int = Field(default=INITIAL_FETCH_K, ge=5, le=50)
    rerank_top_n: int = Field(default=RERANK_TOP_N, ge=1, le=20)


class PipelineRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    dense_weight: float = Field(default=DENSE_WEIGHT, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=SPARSE_WEIGHT, ge=0.0, le=1.0)
    initial_fetch_k: int = Field(default=INITIAL_FETCH_K, ge=5, le=50)
    include_content: bool = Field(default=False)


class CustomQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    dense_weight: float = Field(default=DENSE_WEIGHT, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=SPARSE_WEIGHT, ge=0.0, le=1.0)
    initial_fetch_k: int = Field(default=INITIAL_FETCH_K, ge=5, le=50)
    relevant_keywords: Optional[List[str]] = Field(
        default=None,
        description="If provided, compute Recall@K and MRR using keyword overlap matching"
    )
    include_content: bool = Field(default=False)


class GenerationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    pipeline_stage: PipelineStage = Field(default="reranked")
    dense_weight: float = Field(default=DENSE_WEIGHT, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=SPARSE_WEIGHT, ge=0.0, le=1.0)
    include_sources: bool = Field(default=True)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "RAG Optimization Pipeline API",
        "version": "1.0.0",
        "optimization_ladder": {
            "stage_1_baseline": "CharacterTextSplitter(1000) + dense-only retrieval",
            "stage_2_semantic": "SemanticChunker(percentile 90) + dense-only retrieval",
            "stage_3_hybrid": "SemanticChunker + EnsembleRetriever (dense 0.6 + BM25 0.4)",
            "stage_4_reranked": "SemanticChunker + Hybrid(fetch 20) + CrossEncoderReranker(top 4)",
        },
        "endpoints": {
            "docs": "/docs",
            "ingest": {
                "baseline": "POST /ingest/baseline",
                "optimized": "POST /ingest/optimized",
                "file": "POST /ingest/file  ← ingests into BOTH collections",
            },
            "retrieve": {
                "verify": "GET /retrieve/verify",
                "pipeline": "POST /retrieve/pipeline  ← one query through all 4 stages",
                "rerank": "POST /retrieve/rerank",
            },
            "generate": "POST /generate/rag",
            "optimize": {
                "evaluate": "POST /optimize/evaluate  ← KEY endpoint: golden dataset → Recall@K + MRR",
                "custom_query": "POST /optimize/custom-query",
                "golden_dataset": "GET /optimize/golden-dataset",
            },
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    try:
        baseline_count = len(baseline_vectorstore.get()["ids"])
    except Exception:
        baseline_count = 0
    try:
        semantic_count = len(semantic_vectorstore.get()["ids"])
    except Exception:
        semantic_count = 0

    return {
        "status": "healthy",
        "llm_model": OPENAI_MODEL,
        "embedding_model": "text-embedding-3-small",
        "cross_encoder_model": CROSS_ENCODER_MODEL,
        "cross_encoder_loaded": True,
        "baseline_collection": {
            "name": COLLECTION_BASELINE,
            "chunk_count": baseline_count,
            "bm25_chunk_count": len(baseline_chunks),
        },
        "semantic_collection": {
            "name": COLLECTION_SEMANTIC,
            "chunk_count": semantic_count,
            "bm25_chunk_count": len(semantic_chunks),
        },
        "pipeline_config": {
            "dense_weight": DENSE_WEIGHT,
            "sparse_weight": SPARSE_WEIGHT,
            "initial_fetch_k": INITIAL_FETCH_K,
            "rerank_top_n": RERANK_TOP_N,
        },
        "golden_dataset_queries": len(GOLDEN_DATASET),
    }


# ----------------------------------------------------------------------------
# INGESTION
# ----------------------------------------------------------------------------

@app.post("/ingest/baseline", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_baseline(
    file: UploadFile = File(...),
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """
    Ingest a file into the BASELINE collection using fixed-size CharacterTextSplitter.
    This represents Stage 1 of the optimization ladder.
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

        splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        baseline_vectorstore.add_documents(chunks)
        baseline_chunks.extend(chunks)

        return IngestResponse(
            status="success",
            collection=COLLECTION_BASELINE,
            strategy=f"fixed-size (CharacterTextSplitter, size={chunk_size})",
            chunks_created=len(chunks),
            message=f"Ingested '{file.filename}' into baseline collection: {len(chunks)} chunks.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline ingestion failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/ingest/optimized", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_optimized(file: UploadFile = File(...)):
    """
    Ingest a file into the SEMANTIC collection using SemanticChunker.
    This represents Stage 2–4 of the optimization ladder.

    Note: SemanticChunker makes embedding API calls per sentence — it is slower
    than fixed-size chunking but produces topically coherent chunks.
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

        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
        )
        chunks = splitter.split_documents(documents)
        semantic_vectorstore.add_documents(chunks)
        semantic_chunks.extend(chunks)

        return IngestResponse(
            status="success",
            collection=COLLECTION_SEMANTIC,
            strategy=f"semantic (SemanticChunker, {SEMANTIC_BREAKPOINT_TYPE}={SEMANTIC_BREAKPOINT_AMOUNT})",
            chunks_created=len(chunks),
            message=f"Ingested '{file.filename}' into semantic collection: {len(chunks)} chunks.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic ingestion failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/ingest/file", response_model=FileIngestResponse, tags=["Ingestion"])
async def ingest_file_both(file: UploadFile = File(...)):
    """
    Ingest a file into BOTH collections simultaneously.

    - Baseline collection: fixed-size CharacterTextSplitter
    - Semantic collection: SemanticChunker

    This is the recommended starting point — ingest once, then compare all 4
    pipeline stages against the same source documents.
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

        # Baseline
        baseline_splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        b_chunks = baseline_splitter.split_documents(documents)
        baseline_vectorstore.add_documents(b_chunks)
        baseline_chunks.extend(b_chunks)

        # Semantic
        from langchain_experimental.text_splitter import SemanticChunker
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
        )
        s_chunks = semantic_splitter.split_documents(documents)
        semantic_vectorstore.add_documents(s_chunks)
        semantic_chunks.extend(s_chunks)

        return FileIngestResponse(
            status="success",
            filename=file.filename,
            baseline_chunks=len(b_chunks),
            semantic_chunks_count=len(s_chunks),
            message=(
                f"Ingested '{file.filename}' into both collections. "
                f"Baseline: {len(b_chunks)} fixed-size chunks, "
                f"Semantic: {len(s_chunks)} topic-coherent chunks."
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dual ingestion failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ----------------------------------------------------------------------------
# RETRIEVAL
# ----------------------------------------------------------------------------

@app.get("/retrieve/verify", tags=["Retrieval"])
async def verify_stores():
    """Check both collections are populated and report chunk counts."""
    try:
        b_data = baseline_vectorstore.get()
        s_data = semantic_vectorstore.get()
        b_count = len(b_data["ids"])
        s_count = len(s_data["ids"])
        return {
            "status": "ready" if (b_count > 0 and s_count > 0) else "partial" if (b_count > 0 or s_count > 0) else "empty",
            "baseline_collection": {"chunks": b_count, "bm25_chunks": len(baseline_chunks)},
            "semantic_collection": {"chunks": s_count, "bm25_chunks": len(semantic_chunks)},
            "ready_for_evaluation": b_count > 0 and s_count > 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/retrieve/pipeline", tags=["Retrieval"])
async def retrieve_pipeline(request: PipelineRequest):
    """
    Run a single query through all 4 pipeline stages and return each stage's results.

    Use this to visually compare what each stage retrieves for the same query,
    before running the full evaluation (POST /optimize/evaluate).
    """
    try:
        k = request.k
        dw = request.dense_weight
        sw = request.sparse_weight
        fk = request.initial_fetch_k

        b_docs = _run_stage_baseline(request.query, k)
        s_docs = _run_stage_semantic(request.query, k)
        h_docs = _run_stage_hybrid(request.query, k, dw, sw)
        r_docs = _run_stage_reranked(request.query, fk, RERANK_TOP_N, dw, sw)

        return {
            "query": request.query,
            "k": k,
            "stages": {
                "baseline": {
                    "description": f"Fixed-size chunking ({CHUNK_SIZE} chars) + dense-only",
                    "count": len(b_docs),
                    "documents": _format_docs(b_docs, request.include_content),
                },
                "semantic_chunking": {
                    "description": f"SemanticChunker ({SEMANTIC_BREAKPOINT_TYPE} {SEMANTIC_BREAKPOINT_AMOUNT}) + dense-only",
                    "count": len(s_docs),
                    "documents": _format_docs(s_docs, request.include_content),
                },
                "hybrid_search": {
                    "description": f"SemanticChunker + EnsembleRetriever (dense={dw}, BM25={sw})",
                    "count": len(h_docs),
                    "documents": _format_docs(h_docs, request.include_content),
                },
                "reranked": {
                    "description": f"SemanticChunker + Hybrid(fetch {fk}) + CrossEncoderReranker(top {RERANK_TOP_N})",
                    "count": len(r_docs),
                    "documents": _format_docs(r_docs, request.include_content),
                },
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline retrieval failed: {str(e)}")


@app.post("/retrieve/rerank", tags=["Retrieval"])
async def retrieve_rerank(request: PipelineRequest):
    """
    Hybrid retrieval (fetch initial_fetch_k) followed by cross-encoder re-ranking (return top rerank_top_n).

    This is Stage 4 alone. The cross-encoder sees query + document together
    (unlike bi-encoder embeddings), producing much more accurate relevance scores.
    """
    try:
        docs = _run_stage_reranked(
            request.query, request.initial_fetch_k, RERANK_TOP_N,
            request.dense_weight, request.sparse_weight
        )
        return {
            "query": request.query,
            "mode": "hybrid + cross-encoder reranking",
            "fetched": request.initial_fetch_k,
            "returned": len(docs),
            "documents": _format_docs(docs, request.include_content),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-ranking retrieval failed: {str(e)}")


# ----------------------------------------------------------------------------
# OPTIMIZATION / EVALUATION
# ----------------------------------------------------------------------------

@app.get("/optimize/golden-dataset", tags=["Optimization"])
async def get_golden_dataset():
    """
    View the embedded golden dataset used by POST /optimize/evaluate.

    8 queries derived from Documents/guidelines.txt and Documents/policy.txt.
    Relevance is determined by keyword overlap (≥2 keywords must appear in a chunk).
    """
    return {
        "sample_count": len(GOLDEN_DATASET),
        "matching_method": "keyword_overlap",
        "matching_rule": "A retrieved chunk is relevant if it contains ≥2 of the listed keywords",
        "samples": GOLDEN_DATASET,
    }


@app.post("/optimize/evaluate", tags=["Optimization"])
async def evaluate_pipeline(request: EvaluateRequest):
    """
    Run the golden dataset through all 4 pipeline stages and return Recall@K and MRR per stage.

    This is the key teaching endpoint — it mirrors the Lab in guide Section 9
    and makes the quality improvement at each rung of the optimization ladder
    concrete and measurable.

    Expected output (approximate, depends on your documents and chunking):
      Stage 1 (baseline):          Recall@K ~0.50, MRR ~0.40
      Stage 2 (semantic chunking): Recall@K ~0.65, MRR ~0.55
      Stage 3 (hybrid search):     Recall@K ~0.75, MRR ~0.65
      Stage 4 (reranked):          Recall@K ~0.85, MRR ~0.80

    Note: First call may be slower as the cross-encoder processes all queries.
    """
    try:
        k = request.k
        dw = request.dense_weight
        sw = request.sparse_weight
        fk = request.initial_fetch_k
        top_n = request.rerank_top_n

        stage_results = {}

        for stage_name, run_fn in [
            ("baseline", lambda q: _run_stage_baseline(q, k)),
            ("semantic_chunking", lambda q: _run_stage_semantic(q, k)),
            ("hybrid_search", lambda q: _run_stage_hybrid(q, k, dw, sw)),
            ("reranked", lambda q: _run_stage_reranked(q, fk, top_n, dw, sw)),
        ]:
            per_query = []
            recalls = []
            mrrs = []

            for sample in GOLDEN_DATASET:
                docs = run_fn(sample["query"])
                recall = _recall_at_k(docs, sample["relevant_keywords"], k)
                mrr = _mrr(docs, sample["relevant_keywords"])
                recalls.append(recall)
                mrrs.append(mrr)
                per_query.append({
                    "query_id": sample["id"],
                    "query": sample["query"],
                    "recall": round(recall, 3),
                    "reciprocal_rank": round(mrr, 3),
                    "relevant_keywords": sample["relevant_keywords"],
                    "top_retrieved_preview": docs[0].page_content[:100] if docs else "",
                })

            mean_recall = round(sum(recalls) / len(recalls), 3)
            mean_mrr = round(sum(mrrs) / len(mrrs), 3)

            stage_results[stage_name] = {
                "stage": stage_name,
                "description": {
                    "baseline": f"Fixed-size chunking ({CHUNK_SIZE} chars) + dense-only retrieval",
                    "semantic_chunking": f"SemanticChunker ({SEMANTIC_BREAKPOINT_TYPE} {SEMANTIC_BREAKPOINT_AMOUNT}) + dense-only",
                    "hybrid_search": f"SemanticChunker + EnsembleRetriever (dense={dw}, BM25={sw})",
                    "reranked": f"SemanticChunker + Hybrid(fetch {fk}) + CrossEncoderReranker(top {top_n})",
                }[stage_name],
                "recall_at_k": mean_recall,
                "mrr": mean_mrr,
                "per_query": per_query,
            }

        # Improvement summary
        b = stage_results["baseline"]
        r = stage_results["reranked"]
        recall_delta = round(r["recall_at_k"] - b["recall_at_k"], 3)
        mrr_delta = round(r["mrr"] - b["mrr"], 3)

        return {
            "k": k,
            "sample_count": len(GOLDEN_DATASET),
            "matching_method": "keyword_overlap",
            "matching_rule": "A chunk is relevant if it contains ≥2 of the query's keywords",
            "pipeline_stages": stage_results,
            "improvement_summary": {
                "recall_baseline": b["recall_at_k"],
                "recall_reranked": r["recall_at_k"],
                "recall_delta": f"{'+' if recall_delta >= 0 else ''}{recall_delta}",
                "mrr_baseline": b["mrr"],
                "mrr_reranked": r["mrr"],
                "mrr_delta": f"{'+' if mrr_delta >= 0 else ''}{mrr_delta}",
                "best_recall_stage": max(stage_results, key=lambda s: stage_results[s]["recall_at_k"]),
                "best_mrr_stage": max(stage_results, key=lambda s: stage_results[s]["mrr"]),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/optimize/custom-query", tags=["Optimization"])
async def custom_query_evaluation(request: CustomQueryRequest):
    """
    Run any query through all 4 stages.

    Optionally provide `relevant_keywords` to get Recall@K and MRR computed
    on the fly. If omitted, only retrieval results are returned (no metrics).
    """
    try:
        k = request.k
        dw = request.dense_weight
        sw = request.sparse_weight
        fk = request.initial_fetch_k

        stage_docs = {
            "baseline": _run_stage_baseline(request.query, k),
            "semantic_chunking": _run_stage_semantic(request.query, k),
            "hybrid_search": _run_stage_hybrid(request.query, k, dw, sw),
            "reranked": _run_stage_reranked(request.query, fk, RERANK_TOP_N, dw, sw),
        }

        stages_output = {}
        for name, docs in stage_docs.items():
            entry = {
                "count": len(docs),
                "documents": _format_docs(docs, request.include_content),
            }
            if request.relevant_keywords:
                entry["recall_at_k"] = round(_recall_at_k(docs, request.relevant_keywords, k), 3)
                entry["mrr"] = round(_mrr(docs, request.relevant_keywords), 3)
            stages_output[name] = entry

        return {
            "query": request.query,
            "k": k,
            "metrics_computed": request.relevant_keywords is not None,
            "relevant_keywords": request.relevant_keywords,
            "stages": stages_output,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom query failed: {str(e)}")


# ----------------------------------------------------------------------------
# GENERATION
# ----------------------------------------------------------------------------

@app.post("/generate/rag", tags=["Generation"])
async def generate_rag(request: GenerationRequest):
    """
    RAG answer using a chosen pipeline stage.

    - **pipeline_stage**: baseline | semantic_chunking | hybrid_search | reranked (default)
    - Compare answer quality across stages for the same question
    """
    try:
        stage = request.pipeline_stage
        k = request.k
        dw = request.dense_weight
        sw = request.sparse_weight

        if stage == "baseline":
            docs = _run_stage_baseline(request.query, k)
        elif stage == "semantic_chunking":
            docs = _run_stage_semantic(request.query, k)
        elif stage == "hybrid_search":
            docs = _run_stage_hybrid(request.query, k, dw, sw)
        else:
            docs = _run_stage_reranked(request.query, INITIAL_FETCH_K, RERANK_TOP_N, dw, sw)

        if not docs:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Ingest documents first via /ingest/file."
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
            "pipeline_stage": stage,
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
    _rehydrate_chunks()
    print("\n" + "=" * 70)
    print("DEMO-15: FULL RAG OPTIMIZATION PIPELINE — STARTUP")
    print("=" * 70)
    print(f"LLM Model:           {OPENAI_MODEL}")
    print(f"Cross-encoder:       {CROSS_ENCODER_MODEL}")
    print(f"Dense/Sparse:        {DENSE_WEIGHT}/{SPARSE_WEIGHT}")
    print(f"Initial fetch K:     {INITIAL_FETCH_K}  →  rerank to top {RERANK_TOP_N}")
    print(f"Baseline chunks:     {len(baseline_chunks)} (reloaded from ChromaDB)")
    print(f"Semantic chunks:     {len(semantic_chunks)} (reloaded from ChromaDB)")
    print("=" * 70)
    print("\n✓ API Server Ready!")
    print("✓ Interactive docs: http://localhost:8003/docs")
    print("✓ Health check:     http://localhost:8003/health")
    print("=" * 70)
    print("\n📚 Workflow:")
    print("  1. POST /ingest/file          → ingest docs into both collections")
    print("  2. POST /retrieve/pipeline    → see one query across all 4 stages")
    print("  3. POST /optimize/evaluate    → golden dataset → Recall@K + MRR per stage")
    print("=" * 70)
    print("\n⚠  Re-ranking adds ~100–500ms latency (no API cost, runs locally)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
