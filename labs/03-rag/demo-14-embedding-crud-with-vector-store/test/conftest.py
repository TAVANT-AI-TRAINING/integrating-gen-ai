"""
Pytest configuration and shared fixtures for embedding API tests.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any

# Set up environment variables before any imports to allow module initialization
os.environ["COLLECTION_NAME"] = "test_collection"
os.environ["PERSIST_DIRECTORY"] = "./test_chroma_db"
os.environ["OPENAI_API_EMBEDDING_KEY"] = "test_openai_api_key"
os.environ["OPENAI_MODEL"] = "text-embedding-3-small"

# Patch OpenAIEmbeddings class BEFORE imports to prevent module-level initialization errors
# This must be done before main.embedding_service is imported
_mock_embeddings_instance = MagicMock()
_embeddings_patcher = patch('langchain_openai.OpenAIEmbeddings', return_value=_mock_embeddings_instance)

# Create a mock Chroma vectorstore
_mock_vectorstore = MagicMock()
_mock_collection = MagicMock()
_mock_vectorstore._collection = _mock_collection

# Create a mock langchain_chroma module and inject it into sys.modules before imports
_mock_langchain_chroma_module = MagicMock()
_mock_chroma_class = MagicMock(return_value=_mock_vectorstore)
_mock_langchain_chroma_module.Chroma = _mock_chroma_class
sys.modules['langchain_chroma'] = _mock_langchain_chroma_module

@pytest.fixture(scope="session", autouse=True)
def setup_module_mocks():
    """Set up module-level mocks before any tests run."""
    # Start all patchers - must be done before any module imports
    _embeddings_patcher.start()
    
    # langchain_chroma is already injected into sys.modules above,
    # so main.embedding_service will import our mock when it tries to import it
    
    try:
        yield
    finally:
        # Stop all patchers
        _embeddings_patcher.stop()
        # Clean up sys.modules if needed
        if 'langchain_chroma' in sys.modules and isinstance(sys.modules['langchain_chroma'], MagicMock):
            del sys.modules['langchain_chroma']


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Mock environment variables for all tests to prevent real API calls."""
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("PERSIST_DIRECTORY", "./test_chroma_db")
    monkeypatch.setenv("OPENAI_API_EMBEDDING_KEY", "test_openai_api_key")
    monkeypatch.setenv("OPENAI_MODEL", "text-embedding-3-small")


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test to ensure test isolation."""
    # Reset top-level mocks
    _mock_embeddings_instance.reset_mock()
    _mock_vectorstore.reset_mock()
    _mock_collection.reset_mock()

    # Reset add_documents to ensure clean state
    _mock_vectorstore.add_documents = Mock()
    _mock_vectorstore.similarity_search_with_score = Mock(return_value=[])
    
    # Reset collection methods
    _mock_collection.get = Mock(return_value={'ids': [], 'embeddings': [], 'metadatas': [], 'documents': []})
    _mock_collection.delete = Mock()
    
    yield


@pytest.fixture
def mock_embeddings_model():
    """Fixture providing access to the mock embeddings model."""
    return _mock_embeddings_instance


@pytest.fixture
def mock_collection():
    """Fixture providing access to the mock ChromaDB collection."""
    return _mock_collection


@pytest.fixture
def mock_vectorstore():
    """Fixture providing access to the mock vectorstore."""
    return _mock_vectorstore


@pytest.fixture
def sample_embedding():
    """Fixture providing a sample embedding vector."""
    import numpy as np
    np.random.seed(42)
    return np.random.rand(1536).tolist()  # OpenAI embeddings are typically 1536-dim


@pytest.fixture
def sample_doc_id():
    """Fixture providing a sample document ID."""
    return "test-doc-id-123"


@pytest.fixture
def sample_metadata():
    """Fixture providing sample metadata."""
    return {"source": "test", "page": 1}


@pytest.fixture
def sample_text():
    """Fixture providing sample text content."""
    return "This is a test document for embedding."
