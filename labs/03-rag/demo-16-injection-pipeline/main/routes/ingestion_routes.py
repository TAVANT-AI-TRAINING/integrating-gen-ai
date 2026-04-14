"""
FastAPI Ingestion Routes

API endpoints for document ingestion pipeline and query functionality.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile

from main.services import injection_pipeline_service as pipeline
from main.services import query_service

app = FastAPI(title="RAG Ingestion API", version="1.0.0")


@app.post("/api/v1/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(1000, ge=1, le=8000),
    chunk_overlap: int = Query(100, ge=0, le=4000),
):
    """
    Upload a single document (PDF/TXT/MD/HTML) and run the ingestion pipeline:
    Load -> Chunk -> Embed -> Store in PgVector
    """
    # Persist upload to a temp file with correct suffix
    suffix = Path(file.filename).suffix or ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)

        # ============================================================================
        # DOCUMENT INGESTION PIPELINE
        # ============================================================================
        # This step runs the complete ingestion pipeline:
        # - Loads document using LangChain loaders (PDF/Text/HTML)
        # - Chunks document using RecursiveCharacterTextSplitter
        # - Generates embeddings using Azure OpenAI
        # - Stores chunks with embeddings in PostgreSQL vector database
        # ============================================================================
        result = pipeline.process_and_store_file(
            temp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return JSONResponse(result)
        # ============================================================================

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        # Cleanup temp file
        try:
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


@app.post("/api/v1/query")
async def query_documents(
    query: str = Body(..., embed=True, description="The search query string")
):
    """
    Query the vector database for the most relevant document chunk.
    
    Performs similarity search using the query string and returns the top 1
    most relevant document chunk from the stored embeddings.
    
    Args:
        query: The search query string
        
    Returns:
        JSON response with the top matching document chunk including content and metadata
    """
    try:
        # ============================================================================
        # SIMILARITY SEARCH QUERY
        # ============================================================================
        # This step performs semantic similarity search in the vector database:
        # - Generates embedding for the query using Azure OpenAI
        # - Searches for most similar document chunks using cosine similarity
        # - Returns top-k (k=1) most relevant document chunk with metadata
        # ============================================================================
        result = query_service.query_documents(
            query=query,
            k=1
        )
        return JSONResponse(result)
        # ============================================================================
        
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

