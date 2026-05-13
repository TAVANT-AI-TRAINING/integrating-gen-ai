"""
Demo 02: Multi-Modal RAG FastAPI Service

Extends demo-12 (text-only RAG service) with multi-modal PDF support:
- Ingest PDFs as text chunks, table representations, and GPT-4o image captions
- Retrieve with optional modality filter (text / table / image / all)
- Generate RAG answers citing which modality each source came from

Endpoints:
    POST /ingest/file       — upload a PDF; returns per-modality chunk counts
    POST /retrieve/search   — semantic search with optional element_type filter
    POST /generate/rag      — full RAG answer from multi-modal context
    GET  /stats             — document counts in the vector store
    GET  /health            — service status
    GET  /docs              — Swagger UI

Run:
    uv run uvicorn main:app --reload --port 8000
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — package: pymupdf
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "./chroma_multimodal")
COLLECTION     = os.getenv("COLLECTION_NAME", "multimodal_docs")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "150"))
IMAGE_MIN_SIZE = 5000   # bytes — skip decorative icons

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

# ── LangChain components ──────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)      # image captioning
text_llm   = ChatOpenAI(model=OPENAI_MODEL, temperature=0)  # RAG answer generation

vectorstore = Chroma(
    collection_name=COLLECTION,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Modal RAG API",
    description=(
        "RAG pipeline that indexes PDF text, tables, and images into a single "
        "vector store. Retrieval and generation work across all modalities."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:        str            = Field(..., min_length=1, description="Search query")
    k:            int            = Field(5, ge=1, le=20, description="Results to return")
    element_type: Optional[str] = Field(
        None, description="Modality filter: text | table | image | null (all)"
    )

class SearchResult(BaseModel):
    element_type: str
    page:         int
    source:       str
    content:      str           # first 400 chars of matched chunk
    score:        float

class SearchResponse(BaseModel):
    query:   str
    results: list[SearchResult]
    count:   int

class GenerateRequest(BaseModel):
    query:        str            = Field(..., min_length=1)
    k:            int            = Field(6, ge=1, le=20, description="Context chunks to retrieve")
    element_type: Optional[str] = Field(
        None, description="Modality filter applied during retrieval"
    )

class SourceDoc(BaseModel):
    element_type: str
    page:         int
    source:       str
    excerpt:      str

class GenerateResponse(BaseModel):
    query:            str
    answer:           str
    sources:          list[SourceDoc]
    modalities_used:  list[str]
    context_chunks:   int

class IngestResponse(BaseModel):
    filename:     str
    text_chunks:  int
    table_chunks: int
    image_chunks: int
    total:        int

class StatsResponse(BaseModel):
    collection:   str
    sample_count: int  # approximate — based on k=100 probe
    has_data:     bool

class HealthResponse(BaseModel):
    status:         str
    text_llm:       str
    vision_llm:     str
    collection:     str
    chunk_size:     int
    chunk_overlap:  int

# ── PDF parsing helpers ───────────────────────────────────────────────────────

def _parse_text(pdf_path: str) -> list[Document]:
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) > 50:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source":       Path(pdf_path).name,
                        "page":         page_num,
                        "element_type": "text",
                    },
                ))
    return splitter.split_documents(docs)


def _table_to_text(rows: list, source_name: str, page: int) -> str:
    lines = []
    for i, row in enumerate(rows):
        cells = [str(c or "").strip() for c in row]
        lines.append(" | ".join(cells))
        if i == 0:
            lines.append("-" * (sum(len(c) + 3 for c in cells)))
    return (
        f"TABLE from '{source_name}' (page {page}):\n\n"
        + "\n".join(lines)
    )


def _parse_tables(pdf_path: str) -> list[Document]:
    docs        = []
    source_name = Path(pdf_path).name
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for raw_table in page.extract_tables() or []:
                rows = [r for r in raw_table if any(c for c in r)]
                if len(rows) < 2:
                    continue
                docs.append(Document(
                    page_content=_table_to_text(rows, source_name, page_num),
                    metadata={
                        "source":       source_name,
                        "page":         page_num,
                        "element_type": "table",
                    },
                ))
    return docs


def _caption_image(img_bytes: bytes, mime: str = "png") -> str:
    b64 = base64.standard_b64encode(img_bytes).decode()
    msg = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url":    f"data:image/{mime};base64,{b64}",
                "detail": "high",
            },
        },
        {
            "type": "text",
            "text": (
                "Describe this image from a business document. Include:\n"
                "1. Content type (chart, diagram, photo, screenshot, etc.)\n"
                "2. Key data points, labels, axis values, or text visible\n"
                "3. Business context or purpose\n"
                "Be specific and factual."
            ),
        },
    ])
    return vision_llm.invoke([msg]).content


def _parse_images(pdf_path: str) -> list[Document]:
    docs        = []
    source_name = Path(pdf_path).name
    pdf         = fitz.open(pdf_path)

    for page_num, page in enumerate(pdf, start=1):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                img_data  = pdf.extract_image(xref)
                img_bytes = img_data["image"]
                mime      = img_data.get("ext", "png")
                if len(img_bytes) < IMAGE_MIN_SIZE:
                    continue
                caption = _caption_image(img_bytes, mime)
                docs.append(Document(
                    page_content=(
                        f"IMAGE from '{source_name}' (page {page_num}):\n\n{caption}"
                    ),
                    metadata={
                        "source":       source_name,
                        "page":         page_num,
                        "element_type": "image",
                    },
                ))
            except Exception:
                pass

    pdf.close()
    return docs


def parse_and_index(pdf_path: str) -> dict:
    """Parse all three modalities from a PDF and add to the vector store."""
    text_docs  = _parse_text(pdf_path)
    table_docs = _parse_tables(pdf_path)
    image_docs = _parse_images(pdf_path)

    all_docs = text_docs + table_docs + image_docs
    if all_docs:
        vectorstore.add_documents(all_docs)

    return {
        "text":   len(text_docs),
        "tables": len(table_docs),
        "images": len(image_docs),
        "total":  len(all_docs),
    }


# ── RAG answer generation ─────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a helpful assistant answering questions from company documents.
Answer ONLY using the provided context. If the context is insufficient, say so explicitly.
Cite the source type (TEXT, TABLE, IMAGE) and page number for each piece of information.

Context:
{context}"""


def _build_context(docs: list[Document]) -> str:
    """Format retrieved documents into a labelled context block."""
    parts = []
    for doc in docs:
        etype = doc.metadata.get("element_type", "text").upper()
        page  = doc.metadata.get("page", "?")
        src   = doc.metadata.get("source", "")
        parts.append(f"[{etype} | {src} | page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, docs: list[Document]) -> str:
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        | text_llm
        | StrOutputParser()
    )
    return chain.invoke({
        "context":  _build_context(docs),
        "question": query,
    })


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "service":   "Multi-Modal RAG API",
        "version":   "1.0.0",
        "docs":      "/docs",
        "endpoints": {
            "ingestion":  "POST /ingest/file",
            "retrieval":  "POST /retrieve/search",
            "generation": "POST /generate/rag",
            "stats":      "GET  /stats",
            "health":     "GET  /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="healthy",
        text_llm=OPENAI_MODEL,
        vision_llm="gpt-4o",
        collection=COLLECTION,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
def stats():
    """Return approximate document counts from the vector store."""
    try:
        sample = vectorstore.similarity_search("document", k=100)
        return StatsResponse(
            collection=COLLECTION,
            sample_count=len(sample),
            has_data=len(sample) > 0,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a PDF and index it as text chunks, table representations,
    and GPT-4o Vision image captions — all in a single ChromaDB collection.

    Response shows how many chunks were created per modality.
    Image captioning calls GPT-4o Vision once per qualifying image (>5 KB).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        counts = parse_and_index(tmp_path)

        return IngestResponse(
            filename=file.filename,
            text_chunks=counts["text"],
            table_chunks=counts["tables"],
            image_chunks=counts["images"],
            total=counts["total"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/retrieve/search", response_model=SearchResponse, tags=["Retrieval"])
def search_documents(req: SearchRequest):
    """
    Semantic search across the vector store.

    - Leave `element_type` empty to search all modalities simultaneously.
    - Set `element_type` to `text`, `table`, or `image` to search a single modality.

    Returns chunks with relevance scores (lower = more similar in L2 distance).
    """
    try:
        filter_dict = {"element_type": req.element_type} if req.element_type else None
        raw = vectorstore.similarity_search_with_score(
            req.query, k=req.k, filter=filter_dict
        )
        results = [
            SearchResult(
                element_type=doc.metadata.get("element_type", "text"),
                page=doc.metadata.get("page", 0),
                source=doc.metadata.get("source", ""),
                content=doc.page_content[:400],
                score=round(float(score), 4),
            )
            for doc, score in raw
        ]
        return SearchResponse(query=req.query, results=results, count=len(results))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate/rag", response_model=GenerateResponse, tags=["Generation"])
def generate_rag(req: GenerateRequest):
    """
    Full RAG pipeline over multi-modal context.

    Steps:
    1. Retrieve top-k chunks from the vector store (filtered by modality if specified).
    2. Format context with modality labels (TEXT / TABLE / IMAGE) and page numbers.
    3. Generate an answer using the text LLM (image captions are already text).

    The `element_type` filter is useful when you know the answer is in a chart
    (image) or a data table — it narrows retrieval to that modality only.
    """
    try:
        filter_dict = {"element_type": req.element_type} if req.element_type else None
        docs = vectorstore.similarity_search(req.query, k=req.k, filter=filter_dict)

        if not docs:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No documents found. Upload a PDF first via POST /ingest/file. "
                    + (f"(filter: element_type='{req.element_type}')" if req.element_type else "")
                ),
            )

        answer          = generate_answer(req.query, docs)
        modalities_used = sorted({d.metadata.get("element_type", "text") for d in docs})

        sources = [
            SourceDoc(
                element_type=d.metadata.get("element_type", "text"),
                page=d.metadata.get("page", 0),
                source=d.metadata.get("source", ""),
                excerpt=d.page_content[:200] + "...",
            )
            for d in docs
        ]

        return GenerateResponse(
            query=req.query,
            answer=answer,
            sources=sources,
            modalities_used=modalities_used,
            context_chunks=len(docs),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    print("\n" + "=" * 60)
    print("MULTI-MODAL RAG API — STARTED")
    print("=" * 60)
    print(f"  Text LLM    : {OPENAI_MODEL}")
    print(f"  Vision LLM  : gpt-4o  (image captioning at ingest)")
    print(f"  Embeddings  : text-embedding-3-small")
    print(f"  Collection  : {COLLECTION}")
    print(f"  Chunk size  : {CHUNK_SIZE} / overlap {CHUNK_OVERLAP}")
    print(f"\n  Swagger UI  : http://localhost:8000/docs")
    print(f"  Health      : http://localhost:8000/health")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
