from unittest.mock import MagicMock

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.rag import RagAnswerService


class TestRagAnswerService:
    def test_answer_success(self):
        mock_vector_store = MagicMock()
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()

        chunk = DocumentChunk(chunk_id="1", source_name="test", title="title", url="url", text="chunk text")
        mock_vector_store.query.return_value = [RetrievedChunk(chunk=chunk, distance=0.1, confidence=0.9)]
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
        mock_vector_store.query.return_value = [RetrievedChunk(chunk=chunk, distance=0.9, confidence=0.1)]

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
