import pytest
from unittest.mock import patch, MagicMock

from data_engineering_copilot.services.rag import RagAnswerService, ProductionRagService
from data_engineering_copilot.domain.models import Answer, RetrievedChunk, DocumentChunk
from data_engineering_copilot.config.settings import settings

class TestRagAnswerService:
    def test_answer_success(self):
        mock_vector_store = MagicMock()
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()
        
        chunk = DocumentChunk(chunk_id="1", source_name="test", title="title", url="url", text="chunk text")
        mock_vector_store.query.return_value = [
            RetrievedChunk(chunk=chunk, distance=0.1, confidence=0.9)
        ]
        mock_ollama.generate.return_value = "Ollama Answer"
        
        service = RagAnswerService(mock_vector_store, mock_ollama, mock_embedder)
        answer = service.answer("What is this?")
        
        assert answer.text == "Ollama Answer"
        assert answer.confidence == 0.9
        assert answer.sources == (chunk,)

    def test_answer_low_confidence(self):
        mock_vector_store = MagicMock()
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()
        
        chunk = DocumentChunk(chunk_id="1", source_name="test", title="title", url="url", text="chunk text")
        mock_vector_store.query.return_value = [
            RetrievedChunk(chunk=chunk, distance=0.9, confidence=0.1)
        ]
        
        service = RagAnswerService(mock_vector_store, mock_ollama, mock_embedder)
        answer = service.answer("What is this?")
        
        assert "outside my knowledge repository" in answer.text
        assert answer.confidence == 0.0
        assert len(answer.sources) == 0
        
    def test_answer_no_chunks(self):
        mock_vector_store = MagicMock()
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()
        
        mock_vector_store.query.return_value = []
        
        service = RagAnswerService(mock_vector_store, mock_ollama, mock_embedder)
        answer = service.answer("What is this?")
        
        assert "outside my knowledge repository" in answer.text

class TestProductionRagService:
    @patch("data_engineering_copilot.services.rag.Langfuse")
    @patch("data_engineering_copilot.services.rag.requests.post")
    @patch("data_engineering_copilot.services.rag.QdrantVectorStore")
    @patch("data_engineering_copilot.services.rag.SentenceTransformerEmbeddings")
    @patch("data_engineering_copilot.services.rag.OllamaClient")
    def test_answer_question_success(self, mock_ollama_client, mock_embedder_class, mock_qdrant_class, mock_post, mock_langfuse_class):
        mock_langfuse = mock_langfuse_class.return_value
        mock_trace = MagicMock()
        mock_langfuse.trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        mock_qdrant = mock_qdrant_class.return_value
        chunk = DocumentChunk(chunk_id="1", source_name="test_src", title="t", url="u", text="ctx")
        mock_qdrant.query.return_value = [
            RetrievedChunk(chunk=chunk, distance=0.2, confidence=0.8)
        ]
        
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "The answer"}
        mock_post.return_value = mock_resp
        
        service = ProductionRagService()
        result = service.answer_question("user1", "session1", "query?", 4)
        
        assert result["answer"] == "The answer"
        assert result["confidence"] == 0.8
        assert "test_src" in result["sources"]
        
    @patch("data_engineering_copilot.services.rag.Langfuse")
    @patch("data_engineering_copilot.services.rag.requests.post")
    @patch("data_engineering_copilot.services.rag.QdrantVectorStore")
    @patch("data_engineering_copilot.services.rag.SentenceTransformerEmbeddings")
    @patch("data_engineering_copilot.services.rag.OllamaClient")
    def test_answer_question_empty(self, mock_ollama_client, mock_embedder_class, mock_qdrant_class, mock_post, mock_langfuse_class):
        mock_langfuse = mock_langfuse_class.return_value
        mock_trace = MagicMock()
        mock_langfuse.trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        mock_qdrant = mock_qdrant_class.return_value
        mock_qdrant.query.return_value = []
        
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "The answer"}
        mock_post.return_value = mock_resp
        
        service = ProductionRagService()
        result = service.answer_question("user1", "session1", "query?", 4)
        assert result["confidence"] == 0.0

    @patch("data_engineering_copilot.services.rag.Langfuse")
    @patch("data_engineering_copilot.services.rag.QdrantVectorStore")
    @patch("data_engineering_copilot.services.rag.SentenceTransformerEmbeddings")
    @patch("data_engineering_copilot.services.rag.OllamaClient")
    def test_answer_question_retrieval_fail(self, mock_ollama, mock_embed, mock_qdrant_class, mock_langfuse_class):
        mock_langfuse = mock_langfuse_class.return_value
        mock_trace = MagicMock()
        mock_langfuse.trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        mock_qdrant = mock_qdrant_class.return_value
        mock_qdrant.query.side_effect = Exception("DB error")
        
        service = ProductionRagService()
        with pytest.raises(Exception, match="DB error"):
            service.answer_question("u", "s", "q", 4)

    @patch("data_engineering_copilot.services.rag.Langfuse")
    @patch("data_engineering_copilot.services.rag.requests.post")
    @patch("data_engineering_copilot.services.rag.QdrantVectorStore")
    @patch("data_engineering_copilot.services.rag.SentenceTransformerEmbeddings")
    @patch("data_engineering_copilot.services.rag.OllamaClient")
    def test_answer_question_generation_fail(self, mock_ollama, mock_embed, mock_qdrant, mock_post, mock_langfuse_class):
        mock_langfuse = mock_langfuse_class.return_value
        mock_trace = MagicMock()
        mock_langfuse.trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        mock_post.side_effect = Exception("Network error")
        
        service = ProductionRagService()
        with pytest.raises(Exception, match="Network error"):
            service.answer_question("u", "s", "q", 4)
