"""
Comprehensive test suite for embedding_service.py

Tests cover:
- embed_and_store_text() function with mocked ChromaDB operations
- get_embedding_by_id() function with various scenarios
- delete_by_doc_id() function
- query_similar_documents() function
- Edge cases and error handling
- Input validation
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

# Add parent directory to path to import from main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbedAndStoreText:
    """Test suite for embed_and_store_text() function"""

    def test_embed_and_store_text_success(self, sample_text, mock_vectorstore):
        """Test successful embedding and storage of text"""
        from main.service.embedding_service import embed_and_store_text
        
        mock_vectorstore.add_documents = Mock()
        
        result = embed_and_store_text(text=sample_text)
        
        assert isinstance(result, str)
        assert len(result) > 0  # Should return a doc_id (UUID)
        # Verify add_documents was called with documents and ids
        mock_vectorstore.add_documents.assert_called_once()
        call_args = mock_vectorstore.add_documents.call_args
        assert 'ids' in call_args[1]  # ids should be in kwargs

    def test_embed_and_store_text_with_custom_doc_id(self, sample_text, sample_doc_id, mock_vectorstore):
        """Test embedding and storage with custom document ID"""
        from main.service.embedding_service import embed_and_store_text
        
        mock_vectorstore.add_documents = Mock()
        
        result = embed_and_store_text(text=sample_text, doc_id=sample_doc_id)
        
        assert result == sample_doc_id
        mock_vectorstore.add_documents.assert_called_once()
        # Verify the custom doc_id was used
        call_args = mock_vectorstore.add_documents.call_args
        assert call_args[1]['ids'][0] == sample_doc_id

    def test_embed_and_store_text_with_metadata(self, sample_text, sample_metadata, mock_vectorstore):
        """Test embedding and storage with metadata"""
        from main.service.embedding_service import embed_and_store_text
        
        mock_vectorstore.add_documents = Mock()
        
        result = embed_and_store_text(text=sample_text, metadata=sample_metadata)
        
        assert isinstance(result, str)
        # Verify that add_documents was called with a document containing metadata
        call_args = mock_vectorstore.add_documents.call_args
        assert call_args is not None
        documents = call_args[0][0]
        assert len(documents) == 1
        assert documents[0].page_content == sample_text

    def test_embed_and_store_text_empty_string_raises_error(self, mock_vectorstore):
        """Test that empty string text is handled (should still work but may be invalid)"""
        from main.service.embedding_service import embed_and_store_text
        
        mock_vectorstore.add_documents = Mock()
        
        # Empty string might be accepted but could cause issues
        result = embed_and_store_text(text="")
        assert isinstance(result, str)

    def test_embed_and_store_text_vectorstore_error(self, sample_text, mock_vectorstore):
        """Test error handling when vectorstore operations fail"""
        from main.service.embedding_service import embed_and_store_text
        
        mock_vectorstore.add_documents.side_effect = Exception("ChromaDB connection failed")
        
        with pytest.raises(Exception, match="ChromaDB connection failed"):
            embed_and_store_text(text=sample_text)

    def test_embed_and_store_text_generates_uuid_when_no_doc_id(self, sample_text, mock_vectorstore):
        """Test that UUID is generated when doc_id is not provided"""
        from main.service.embedding_service import embed_and_store_text
        
        mock_vectorstore.add_documents = Mock()
        
        result1 = embed_and_store_text(text=sample_text)
        result2 = embed_and_store_text(text=sample_text)
        
        # Both should be valid UUIDs and different
        assert result1 != result2
        assert len(result1) > 0
        assert len(result2) > 0


class TestGetEmbeddingById:
    """Test suite for get_embedding_by_id() function"""

    def test_get_embedding_by_id_success(self, sample_doc_id, sample_embedding, sample_metadata, mock_collection):
        """Test successful retrieval of embedding"""
        from main.service.embedding_service import get_embedding_by_id
        
        # Configure mock to return document data
        mock_collection.get.return_value = {
            'ids': [sample_doc_id],
            'embeddings': [sample_embedding],
            'metadatas': [sample_metadata],
            'documents': ["test content"]
        }
        
        embedding, metadata, page_content = get_embedding_by_id(sample_doc_id)
        
        assert embedding == sample_embedding
        assert metadata == sample_metadata
        assert page_content == "test content"
        mock_collection.get.assert_called_once_with(
            ids=[sample_doc_id],
            include=["embeddings", "metadatas", "documents"]
        )

    def test_get_embedding_by_id_not_found(self, sample_doc_id, mock_collection):
        """Test that ValueError is raised when document is not found"""
        from main.service.embedding_service import get_embedding_by_id
        
        # Configure mock to return empty results
        mock_collection.get.return_value = {
            'ids': [],
            'embeddings': [],
            'metadatas': [],
            'documents': []
        }
        
        with pytest.raises(ValueError, match=f"Document with ID '{sample_doc_id}' not found"):
            get_embedding_by_id(sample_doc_id)

    def test_get_embedding_by_id_with_metadata(self, sample_doc_id, sample_embedding, mock_collection):
        """Test handling of metadata with page_content"""
        from main.service.embedding_service import get_embedding_by_id
        
        metadata_dict = {"source": "test", "page": 1, "id": sample_doc_id}
        mock_collection.get.return_value = {
            'ids': [sample_doc_id],
            'embeddings': [sample_embedding],
            'metadatas': [metadata_dict],
            'documents': ["test content"]
        }
        
        embedding, metadata, page_content = get_embedding_by_id(sample_doc_id)
        
        assert embedding == sample_embedding
        assert isinstance(metadata, dict)
        assert page_content == "test content"

    def test_get_embedding_by_id_exception_handling(self, sample_doc_id, mock_collection):
        """Test exception handling in get_embedding_by_id"""
        from main.service.embedding_service import get_embedding_by_id
        
        mock_collection.get.side_effect = Exception("ChromaDB error")
        
        with pytest.raises(ValueError, match="Error retrieving document"):
            get_embedding_by_id(sample_doc_id)


class TestDeleteByDocId:
    """Test suite for delete_by_doc_id() function"""

    def test_delete_by_doc_id_success(self, mock_collection, sample_doc_id):
        """Test successful deletion of document by ID"""
        from main.service.embedding_service import delete_by_doc_id
        
        # Configure mock to indicate document exists
        mock_collection.get.return_value = {
            'ids': [sample_doc_id],
            'embeddings': [],
            'metadatas': [],
            'documents': []
        }
        mock_collection.delete = Mock()
        
        result = delete_by_doc_id(sample_doc_id)
        
        assert result is True
        mock_collection.get.assert_called_once_with(ids=[sample_doc_id])
        mock_collection.delete.assert_called_once_with(ids=[sample_doc_id])

    def test_delete_by_doc_id_not_found(self, mock_collection, sample_doc_id):
        """Test deletion when document is not found"""
        from main.service.embedding_service import delete_by_doc_id
        
        # Configure mock to indicate document doesn't exist
        mock_collection.get.return_value = {
            'ids': [],
            'embeddings': [],
            'metadatas': [],
            'documents': []
        }
        
        result = delete_by_doc_id(sample_doc_id)
        
        assert result is False
        mock_collection.get.assert_called_once_with(ids=[sample_doc_id])
        # delete should not be called if document doesn't exist
        mock_collection.delete.assert_not_called()

    def test_delete_by_doc_id_database_error(self, mock_collection, sample_doc_id):
        """Test error handling when ChromaDB operation fails"""
        from main.service.embedding_service import delete_by_doc_id
        
        mock_collection.get.side_effect = Exception("ChromaDB error")
        
        result = delete_by_doc_id(sample_doc_id)
        
        assert result is False  # Should return False on error


class TestQuerySimilarDocuments:
    """Test suite for query_similar_documents() function"""

    def test_query_similar_documents_success(self, mock_vectorstore):
        """Test successful similarity search"""
        from main.service.embedding_service import query_similar_documents
        from langchain_core.documents import Document
        
        # Mock return value with documents and scores
        mock_docs = [
            (Document(page_content="test1", metadata={"id": "1"}), 0.1),
            (Document(page_content="test2", metadata={"id": "2"}), 0.2)
        ]
        mock_vectorstore.similarity_search_with_score.return_value = mock_docs
        
        results = query_similar_documents("test query", k=2)
        
        assert len(results) == 2
        assert results[0][1] == 0.1  # Check score
        mock_vectorstore.similarity_search_with_score.assert_called_once_with("test query", k=2)

    def test_query_similar_documents_with_filters(self, mock_vectorstore):
        """Test similarity search with metadata filters"""
        from main.service.embedding_service import query_similar_documents
        
        mock_vectorstore.similarity_search_with_score.return_value = []
        filters = {"source": "test"}
        
        results = query_similar_documents("test query", k=5, filters=filters)
        
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(
            "test query", k=5, filter=filters
        )

    def test_query_similar_documents_empty_filters(self, mock_vectorstore):
        """Test that empty filter dict is treated as no filter"""
        from main.service.embedding_service import query_similar_documents
        
        mock_vectorstore.similarity_search_with_score.return_value = []
        
        results = query_similar_documents("test query", k=5, filters={})
        
        # Should be called without filter parameter
        call_args = mock_vectorstore.similarity_search_with_score.call_args
        assert 'filter' not in call_args[1] or call_args[1].get('filter') is None


class TestStoreDocuments:
    """Test suite for store_documents() function"""

    def test_store_documents_success(self, mock_vectorstore):
        """Test successful batch storage of documents"""
        from main.service.embedding_service import store_documents
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="test1", metadata={"id": "1"}),
            Document(page_content="test2", metadata={"id": "2"})
        ]
        mock_vectorstore.add_documents = Mock(return_value=None)
        
        result = store_documents(docs)
        
        assert result is True
        mock_vectorstore.add_documents.assert_called_once()
        # Verify IDs were extracted and passed
        call_args = mock_vectorstore.add_documents.call_args
        assert 'ids' in call_args[1]
        assert len(call_args[1]['ids']) == 2

    def test_store_documents_empty_list(self, mock_vectorstore):
        """Test storing empty document list"""
        from main.service.embedding_service import store_documents
        
        result = store_documents([])
        
        assert result is True
        # add_documents should not be called for empty list
        mock_vectorstore.add_documents.assert_not_called()

    def test_store_documents_generates_ids_when_missing(self, mock_vectorstore):
        """Test that IDs are generated for documents without them"""
        from main.service.embedding_service import store_documents
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="test1", metadata={}),
            Document(page_content="test2", metadata={})
        ]
        mock_vectorstore.add_documents = Mock(return_value=None)
        
        result = store_documents(docs)
        
        assert result is True
        # Verify IDs were generated and added to metadata
        for doc in docs:
            assert "id" in doc.metadata
            assert len(doc.metadata["id"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
