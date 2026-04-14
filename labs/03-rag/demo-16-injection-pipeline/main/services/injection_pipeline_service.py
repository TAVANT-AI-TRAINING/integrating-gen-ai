"""
Injection Pipeline Service

This service orchestrates the complete document ingestion pipeline:
Load -> Chunk -> Inject Metadata -> Embed -> Store in PgVector

Uses modular services for each step of the pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List

from main.services import document_processing_service as processing
from main.services import embedding_service as storage
from main.services import FileTypeDetector, ChunkingService


# Initialize services
_chunking_service = ChunkingService()


def process_and_store_file(
    file_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    """
    Process a document file through the complete ingestion pipeline:
    Load -> Chunk -> Inject Metadata -> Embed -> Store in PgVector
    
    Note: Similarity search functionality will be added in the next sprint.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Dictionary with processing results including chunks and statistics
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file type is unsupported
        RuntimeError: If storage fails
    """
    # Step 1-2: Validate file existence and type
    if not file_path or not file_path.exists():
        raise FileNotFoundError("Input file does not exist")
    elif not FileTypeDetector.is_supported(file_path):
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    file_type = FileTypeDetector.get_file_type(file_path)
    
    # Step 3: Categorize file for processing
    categorized = FileTypeDetector.categorize_files([file_path])
    
    # Step 4: Load and process document
    # ============================================================================
    # DOCUMENT LOADING AND PROCESSING
    # ============================================================================
    # This step loads the document using LangChain loaders and processes it:
    # - Loads document based on file type (PDF, text, HTML)
    # - Chunks the document using RecursiveCharacterTextSplitter
    # - Returns chunks with statistics (count, lengths, processing time)
    # ============================================================================
    result = processing.load_and_process_multi_source(
        pdf_files=categorized.get("pdf", []),
        text_files=categorized.get("text", []),
        html_files=categorized.get("html", []),
        web_urls=None,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # ============================================================================
    
    chunks = result["chunks"]
    
    # Step 5: Inject metadata (chunk IDs)
    # ============================================================================
    # CHUNK ID INJECTION
    # ============================================================================
    # This step injects unique chunk IDs into document metadata:
    # - Generates unique identifiers for each chunk
    # - Adds chunk IDs to document metadata for tracking and retrieval
    # - Associates chunks with source file path
    # ============================================================================
    chunk_ids = _chunking_service.inject_chunk_ids(
        chunks,
        file_path=file_path
    )
    # ============================================================================
    
    # Step 6: Store documents in vector database
    # ============================================================================
    # VECTOR DATABASE STORAGE
    # ============================================================================
    # This step stores document chunks in PostgreSQL vector database:
    # - Generates embeddings for each chunk using Azure OpenAI
    # - Stores chunks with embeddings in PGVector collection
    # - Raises error if storage fails
    # ============================================================================
    ok = storage.store_documents(chunks)
    if not ok:
        raise RuntimeError("Failed to store documents in vector database")
    # ============================================================================
    
    # Step 7: Prepare response
    out: Dict[str, Any] = {
        "status": "success",
        "filename": file_path.name,
        "file_type": file_type,
        "total_chunks": result["total_chunks"],
        "average_chunk_length": result["average_chunk_length"],
        "min_chunk_length": result["min_chunk_length"],
        "max_chunk_length": result["max_chunk_length"],
        "processing_time_seconds": result["processing_time_seconds"],
        "collection_name": storage.get_collection_name(),
        "chunk_ids": chunk_ids,
        "source_documents": result.get("source_documents", len(chunks)),
    }
    return out


