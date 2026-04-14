"""
Pydantic models for request validation and response formatting.
This module contains all data models used by the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


# ============================================================================
# REQUEST MODELS
# ============================================================================

class DocumentRequest(BaseModel):
    """Request model for embedding and storing a document."""
    text: str = Field(
        ...,
        description="The document content to embed and store",
        example="All employees must complete the annual security training by December 31st."
    )
    doc_id: Optional[str] = Field(
        None,
        description="Optional document ID (auto-generated if not provided)",
        example="compliance_guide_001"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata for the document",
        example={"source": "compliance_guide", "page": 1, "category": "security"}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "All employees must complete the annual security training by December 31st.",
                "doc_id": "compliance_guide_001",
                "metadata": {
                    "source": "compliance_guide",
                    "page": 1,
                    "category": "security"
                }
            }
        }

class GetEmbeddingRequest(BaseModel):
    """Request model for retrieving an embedding by document ID."""
    doc_id: str = Field(
        ...,
        description="Required document ID to retrieve",
        example="compliance_guide_001"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "compliance_guide_001"
            }
        }

class DeleteEmbeddingRequest(BaseModel):
    """Request model for deleting an embedding by document ID."""
    doc_id: str = Field(
        ...,
        description="Required document ID to delete",
        example="compliance_guide_001"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "compliance_guide_001"
            }
        }

class QueryRequest(BaseModel):
    """Request model for querying similar documents."""
    query: str = Field(
        ...,
        description="The search query text",
        example="What is the company's vacation policy?"
    )
    top_k: int = Field(
        5,
        description="Number of results to return (default: 5)",
        example=5,
        ge=1,
        le=100
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filters for filtering results",
        example={"source": "hr_manual"}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the company's vacation policy?",
                "top_k": 5,
                "filters": {
                    "source": "hr_manual"
                }
            },
            "examples": [
                {
                    "query": "What is the company's vacation policy?",
                    "top_k": 5
                },
                {
                    "query": "security training requirements",
                    "top_k": 3,
                    "filters": {
                        "source": "compliance_guide",
                        "category": "security"
                    }
                }
            ]
        }

class BatchDocumentRequest(BaseModel):
    """Request model for batch embedding and storing multiple documents."""
    documents: list[DocumentRequest] = Field(
        ...,
        description="List of documents to embed and store",
        min_items=1,
        example=[
            {
                "text": "All employees must complete the annual security training by December 31st.",
                "doc_id": "compliance_guide_001",
                "metadata": {"source": "compliance_guide", "page": 1}
            },
            {
                "text": "Our company's vacation policy allows for 20 paid days off per year.",
                "doc_id": "hr_manual_002",
                "metadata": {"source": "hr_manual", "page": 2}
            }
        ]
    )
    collection_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata for the collection (only used if collection doesn't exist)",
        example={"version": "1.0", "created_by": "admin"}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "text": "All employees must complete the annual security training by December 31st.",
                        "doc_id": "compliance_guide_001",
                        "metadata": {"source": "compliance_guide", "page": 1}
                    },
                    {
                        "text": "Our company's vacation policy allows for 20 paid days off per year.",
                        "doc_id": "hr_manual_002",
                        "metadata": {"source": "hr_manual", "page": 2}
                    }
                ],
                "collection_metadata": {
                    "version": "1.0",
                    "created_by": "admin"
                }
            }
        }


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class StoreResponse(BaseModel):
    """Response model for successful document storage."""
    message: str = Field(
        ...,
        description="Success message",
        example="Document embedded and stored successfully"
    )
    doc_id: str = Field(
        ...,
        description="The document ID (provided or generated)",
        example="compliance_guide_001"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Document embedded and stored successfully",
                "doc_id": "compliance_guide_001"
            }
        }

class EmbeddingResponse(BaseModel):
    """Response model for retrieving an embedding."""
    doc_id: str = Field(
        ...,
        description="The document ID",
        example="compliance_guide_001"
    )
    embedding: list[float] = Field(
        ...,
        description="The vector embedding (typically 1536 dimensions for OpenAI embeddings)",
        example=[0.0123, -0.0456, 0.0789, 0.0123, -0.0456, 0.0789]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Document metadata",
        example={"source": "compliance_guide", "page": 1, "id": "compliance_guide_001"}
    )
    page_content: str = Field(
        ...,
        description="The original document content",
        example="All employees must complete the annual security training by December 31st."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "compliance_guide_001",
                "embedding": [0.0123, -0.0456, 0.0789, 0.0123, -0.0456, 0.0789],
                "metadata": {
                    "source": "compliance_guide",
                    "page": 1,
                    "id": "compliance_guide_001"
                },
                "page_content": "All employees must complete the annual security training by December 31st."
            }
        }

class DeleteResponse(BaseModel):
    """Response model for successful document deletion."""
    message: str = Field(
        ...,
        description="Success message",
        example="Document with ID 'compliance_guide_001' deleted successfully"
    )
    doc_id: str = Field(
        ...,
        description="The deleted document ID",
        example="compliance_guide_001"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Document with ID 'compliance_guide_001' deleted successfully",
                "doc_id": "compliance_guide_001"
            }
        }

class QueryHit(BaseModel):
    """Model representing a single query result."""
    doc_id: Optional[str] = Field(
        None,
        description="The document ID",
        example="hr_manual_002"
    )
    score: float = Field(
        ...,
        description="Similarity score (higher = more similar, typically between 0 and 1)",
        example=0.8567
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Document metadata",
        example={"source": "hr_manual", "page": 2, "id": "hr_manual_002"}
    )
    page_content: str = Field(
        ...,
        description="The document content",
        example="Our company's vacation policy allows for 20 paid days off per year."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "hr_manual_002",
                "score": 0.8567,
                "metadata": {
                    "source": "hr_manual",
                    "page": 2,
                    "id": "hr_manual_002"
                },
                "page_content": "Our company's vacation policy allows for 20 paid days off per year."
            }
        }

class QueryResponse(BaseModel):
    """Response model for document query results."""
    query: str = Field(
        ...,
        description="The original query text",
        example="What is the company's vacation policy?"
    )
    results: list[QueryHit] = Field(
        ...,
        description="List of similar documents with similarity scores"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the company's vacation policy?",
                "results": [
                    {
                        "doc_id": "hr_manual_002",
                        "score": 0.8567,
                        "metadata": {
                            "source": "hr_manual",
                            "page": 2,
                            "id": "hr_manual_002"
                        },
                        "page_content": "Our company's vacation policy allows for 20 paid days off per year."
                    },
                    {
                        "doc_id": "hr_manual_003",
                        "score": 0.7234,
                        "metadata": {
                            "source": "hr_manual",
                            "page": 3,
                            "id": "hr_manual_003"
                        },
                        "page_content": "The work-from-home policy requires manager approval for remote work."
                    }
                ]
            }
        }

class BatchStoreResponse(BaseModel):
    """Response model for successful batch document storage."""
    message: str = Field(
        ...,
        description="Success message",
        example="Successfully stored 2 documents"
    )
    count: int = Field(
        ...,
        description="Number of documents stored",
        example=2
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Successfully stored 2 documents",
                "count": 2
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(
        ...,
        description="Health status of the API",
        example="healthy"
    )
    message: str = Field(
        ...,
        description="Detailed health message",
        example="Embedding API is running successfully"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Embedding API is running successfully"
            }
        }
