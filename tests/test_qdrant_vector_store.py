import pytest
from unittest.mock import Mock, patch
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk


class TestQdrantVectorStore:
    @pytest.fixture
    def mock_qdrant_client(self):
        with patch('qdrant_client.QdrantClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def sample_chunks(self):
        return [
            DocumentChunk(
                chunk_id="chunk1",
                source_name="test_source",
                title="Test Title 1",
                url="http://example.com/1",
                text="This is test content 1"
            ),
            DocumentChunk(
                chunk_id="chunk2",
                source_name="test_source",
                title="Test Title 2",
                url="http://example.com/2",
                text="This is test content 2"
            )
        ]

    @pytest.fixture
    def sample_embeddings(self):
        return [
            [0.1] * 384,
            [0.2] * 384
        ]

    def test_init_creates_client_and_collection(self, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = False
        store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
        assert store._url == "http://localhost:6333"
        assert store._collection_name == "test"
        mock_qdrant_client.collection_exists.assert_called_once_with("test")
        mock_qdrant_client.recreate_collection.assert_called_once()

    def test_init_skips_creation_if_exists(self, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = True
        store = QdrantVectorStore(url="http://localhost:6333", collection_name="existing")
        mock_qdrant_client.collection_exists.assert_called_once_with("existing")
        mock_qdrant_client.recreate_collection.assert_not_called()

    def test_upsert_chunks_success(self, mock_qdrant_client, sample_chunks, sample_embeddings):
        mock_qdrant_client.collection_exists.return_value = False
        store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
        store.upsert_chunks(sample_chunks, sample_embeddings)
        mock_qdrant_client.upsert.assert_called_once()

    def test_query_success(self, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = False
        mock_hit = Mock()
        mock_hit.id = "chunk1"
        mock_hit.score = 0.8
        mock_hit.payload = {
            "source_name": "test_source",
            "title": "Test Title",
            "url": "http://example.com/1",
            "text": "Test content"
        }
        mock_qdrant_client.search.return_value = [mock_hit]
        
        store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
        results = store.query([0.1] * 384, top_k=1)
        
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chunk1"
        assert results[0].confidence == 0.2  # 1 - 0.8

    def test_count_success(self, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = False
        mock_info = Mock()
        mock_info.points_count = 42
        mock_qdrant_client.get_collection.return_value = mock_info
        
        store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
        assert store.count() == 42