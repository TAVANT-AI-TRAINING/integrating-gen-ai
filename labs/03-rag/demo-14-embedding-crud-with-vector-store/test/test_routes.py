"""
Comprehensive test suite for routes.py (FastAPI endpoints)

Tests cover:
- POST /embed_and_store endpoint
- POST /get_embedding endpoint
- DELETE /delete_embedding endpoint
- GET /health endpoint
- Request/response validation
- Error handling and status codes
"""
import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock, Mock
from fastapi.testclient import TestClient
from fastapi import status

# Add parent directory to path to import from main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create a test client for FastAPI app"""
    from main.app import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_doc_id():
    """Sample document ID for testing"""
    return "test-doc-id-12345"


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing"""
    import numpy as np
    np.random.seed(42)
    return np.random.rand(1024).tolist()


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {"source": "test", "page": 1}


@pytest.fixture
def sample_text():
    """Sample text content for testing"""
    return "This is a test document for embedding and storage."


class TestHealthEndpoint:
    """Test suite for GET /health endpoint"""

    def test_health_endpoint_success(self, client):
        """Test successful health check"""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
        assert "Embedding API is running" in data["message"]


class TestEmbedAndStoreEndpoint:
    """Test suite for POST /embed_and_store endpoint"""

    @patch('main.routes.routes.embed_and_store_text')
    def test_embed_and_store_success(self, mock_store, client, sample_text, sample_doc_id):
        """Test successful embedding and storage"""
        mock_store.return_value = sample_doc_id
        
        response = client.post(
            "/embed_and_store",
            json={
                "text": sample_text,
                "doc_id": sample_doc_id,
                "metadata": {"source": "test"}
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["message"] == "Document embedded and stored successfully"
        assert data["doc_id"] == sample_doc_id
        mock_store.assert_called_once()

    @patch('main.routes.routes.embed_and_store_text')
    def test_embed_and_store_auto_generated_id(self, mock_store, client, sample_text):
        """Test embedding and storage with auto-generated document ID"""
        generated_id = "auto-generated-id-123"
        mock_store.return_value = generated_id
        
        response = client.post(
            "/embed_and_store",
            json={"text": sample_text}
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["doc_id"] == generated_id
        # Verify doc_id was None (auto-generated)
        call_kwargs = mock_store.call_args[1]
        assert call_kwargs.get("doc_id") is None

    @patch('main.routes.routes.embed_and_store_text')
    def test_embed_and_store_with_metadata(self, mock_store, client, sample_text, sample_metadata, sample_doc_id):
        """Test embedding and storage with metadata"""
        mock_store.return_value = sample_doc_id
        
        response = client.post(
            "/embed_and_store",
            json={
                "text": sample_text,
                "metadata": sample_metadata
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_store.call_args[1]
        assert call_kwargs["metadata"] == sample_metadata

    def test_embed_and_store_missing_text(self, client):
        """Test that missing text field raises validation error"""
        response = client.post(
            "/embed_and_store",
            json={}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('main.routes.routes.embed_and_store_text')
    def test_embed_and_store_empty_text(self, mock_store, client, sample_doc_id):
        """Test embedding and storage with empty text"""
        mock_store.return_value = sample_doc_id
        
        response = client.post(
            "/embed_and_store",
            json={"text": ""}
        )
        
        # Empty text might be accepted (validation depends on implementation)
        # If it's accepted, status should be 201
        assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_422_UNPROCESSABLE_ENTITY]

    @patch('main.routes.routes.embed_and_store_text')
    def test_embed_and_store_database_error(self, mock_store, client, sample_text):
        """Test error handling when database operation fails"""
        mock_store.side_effect = Exception("Database connection failed")
        
        response = client.post(
            "/embed_and_store",
            json={"text": sample_text}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "Internal server error" in data["detail"]


class TestGetEmbeddingEndpoint:
    """Test suite for GET /get_embedding endpoint"""

    @patch('main.routes.routes.get_embedding_by_id')
    def test_get_embedding_success(self, mock_get, client, sample_doc_id, sample_embedding, sample_metadata):
        """Test successful retrieval of embedding"""
        mock_get.return_value = (sample_embedding, sample_metadata, "Test content")
        
        response = client.get(
            f"/get_embedding?doc_id={sample_doc_id}"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["doc_id"] == sample_doc_id
        assert data["embedding"] == sample_embedding
        assert data["metadata"] == sample_metadata
        assert data["page_content"] == "Test content"
        mock_get.assert_called_once_with(sample_doc_id)

    @patch('main.routes.routes.get_embedding_by_id')
    def test_get_embedding_not_found(self, mock_get, client, sample_doc_id):
        """Test retrieval when document is not found"""
        mock_get.side_effect = ValueError(f"Document with ID '{sample_doc_id}' not found")
        
        response = client.get(
            f"/get_embedding?doc_id={sample_doc_id}"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert sample_doc_id in data["detail"]

    def test_get_embedding_missing_doc_id(self, client):
        """Test that missing doc_id field raises validation error"""
        response = client.get(
            "/get_embedding"
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('main.routes.routes.get_embedding_by_id')
    def test_get_embedding_empty_doc_id(self, mock_get, client):
        """Test retrieval with empty doc_id"""
        mock_get.side_effect = ValueError("Document with ID '' not found")
        response = client.get(
            "/get_embedding?doc_id="
        )
        
        # Empty doc_id might be accepted but should result in 404
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_ENTITY]

    @patch('main.routes.routes.get_embedding_by_id')
    def test_get_embedding_error(self, mock_get, client, sample_doc_id):
        """Test error handling when service operation fails"""
        mock_get.side_effect = Exception("Service error")
        
        response = client.get(
            f"/get_embedding?doc_id={sample_doc_id}"
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "Internal server error" in data["detail"]


class TestDeleteEmbeddingEndpoint:
    """Test suite for DELETE /delete_embedding endpoint"""

    @patch('main.routes.routes.delete_by_doc_id')
    def test_delete_embedding_success(self, mock_delete, client, sample_doc_id):
        """Test successful deletion of embedding"""
        mock_delete.return_value = True
        
        response = client.request(
            "DELETE",
            "/delete_embedding",
            json={"doc_id": sample_doc_id}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == f"Document with ID '{sample_doc_id}' deleted successfully"
        assert data["doc_id"] == sample_doc_id
        mock_delete.assert_called_once_with(sample_doc_id)

    @patch('main.routes.routes.delete_by_doc_id')
    def test_delete_embedding_not_found(self, mock_delete, client, sample_doc_id):
        """Test deletion when document is not found"""
        mock_delete.return_value = False
        
        response = client.request(
            "DELETE",
            "/delete_embedding",
            json={"doc_id": sample_doc_id}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert f"Document with ID '{sample_doc_id}' not found" in data["detail"]

    def test_delete_embedding_missing_doc_id(self, client):
        """Test that missing doc_id field raises validation error"""
        response = client.request(
            "DELETE",
            "/delete_embedding",
            json={}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('main.routes.routes.delete_by_doc_id')
    def test_delete_embedding_service_error(self, mock_delete, client, sample_doc_id):
        """Test error handling when service operation fails"""
        mock_delete.side_effect = Exception("Database error")
        
        response = client.request(
            "DELETE",
            "/delete_embedding",
            json={"doc_id": sample_doc_id}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "Internal server error" in data["detail"]


class TestRequestValidation:
    """Test suite for request validation and Pydantic models"""

    def test_document_request_model(self):
        """Test DocumentRequest Pydantic model"""
        from main.models.models import DocumentRequest
        
        # Valid request
        request = DocumentRequest(text="test", doc_id="123", metadata={"key": "value"})
        assert request.text == "test"
        assert request.doc_id == "123"
        assert request.metadata == {"key": "value"}
        
        # Request with defaults
        request_default = DocumentRequest(text="test")
        assert request_default.text == "test"
        assert request_default.doc_id is None
        assert request_default.metadata is None

    def test_get_embedding_request_model(self):
        """Test GetEmbeddingRequest Pydantic model"""
        from main.models.models import GetEmbeddingRequest
        
        request = GetEmbeddingRequest(doc_id="123")
        assert request.doc_id == "123"

    def test_delete_embedding_request_model(self):
        """Test DeleteEmbeddingRequest Pydantic model"""
        from main.models.models import DeleteEmbeddingRequest
        
        request = DeleteEmbeddingRequest(doc_id="123")
        assert request.doc_id == "123"


class TestResponseModels:
    """Test suite for response Pydantic models"""

    def test_store_response_model(self):
        """Test StoreResponse model"""
        from main.models.models import StoreResponse
        
        response = StoreResponse(message="Success", doc_id="123")
        assert response.message == "Success"
        assert response.doc_id == "123"

    def test_embedding_response_model(self, sample_embedding, sample_metadata):
        """Test EmbeddingResponse model"""
        from main.models.models import EmbeddingResponse
        
        response = EmbeddingResponse(
            doc_id="123",
            embedding=sample_embedding,
            metadata=sample_metadata,
            page_content="content"
        )
        assert response.doc_id == "123"
        assert response.embedding == sample_embedding
        assert response.metadata == sample_metadata
        assert response.page_content == "content"

    def test_delete_response_model(self):
        """Test DeleteResponse model"""
        from main.models.models import DeleteResponse
        
        response = DeleteResponse(message="Deleted", doc_id="123")
        assert response.message == "Deleted"
        assert response.doc_id == "123"


class TestIntegrationScenarios:
    """Integration test scenarios covering full workflows"""

    @patch('main.routes.routes.embed_and_store_text')
    @patch('main.routes.routes.get_embedding_by_id')
    @patch('main.routes.routes.delete_by_doc_id')
    def test_full_crud_workflow(self, mock_delete, mock_get, mock_store, client, sample_text, sample_doc_id, sample_embedding, sample_metadata):
        """Test complete CRUD workflow: create, read, delete"""
        # Create
        mock_store.return_value = sample_doc_id
        create_response = client.post(
            "/embed_and_store",
            json={"text": sample_text, "doc_id": sample_doc_id}
        )
        assert create_response.status_code == status.HTTP_201_CREATED
        
        # Read
        mock_get.return_value = (sample_embedding, sample_metadata, sample_text)
        read_response = client.get(
            f"/get_embedding?doc_id={sample_doc_id}"
        )
        assert read_response.status_code == status.HTTP_200_OK
        
        # Delete
        mock_delete.return_value = True
        delete_response = client.request(
            "DELETE",
            "/delete_embedding",
            json={"doc_id": sample_doc_id}
        )
        assert delete_response.status_code == status.HTTP_200_OK

    @patch('main.routes.routes.embed_and_store_text')
    @patch('main.routes.routes.get_embedding_by_id')
    def test_store_and_retrieve_workflow(self, mock_get, mock_store, client, sample_text, sample_doc_id, sample_embedding, sample_metadata):
        """Test store and retrieve workflow"""
        # Store
        mock_store.return_value = sample_doc_id
        store_response = client.post(
            "/embed_and_store",
            json={"text": sample_text, "doc_id": sample_doc_id}
        )
        assert store_response.status_code == status.HTTP_201_CREATED
        
        # Retrieve
        mock_get.return_value = (sample_embedding, sample_metadata, sample_text)
        get_response = client.get(
            f"/get_embedding?doc_id={sample_doc_id}"
        )
        assert get_response.status_code == status.HTTP_200_OK
        assert get_response.json()["doc_id"] == sample_doc_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

