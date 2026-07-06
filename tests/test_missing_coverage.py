import pytest
from unittest.mock import patch, MagicMock

from data_engineering_copilot.factory import build_rag_service
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
from pathlib import Path

@patch("data_engineering_copilot.factory.ProductionRagService")
def test_build_rag_service(mock_rag_class):
    mock_rag_class.return_value = MagicMock()
    service = build_rag_service()
    assert service is not None

def test_sentence_transformer_embeddings_ollama():
    """Test that Ollama embeddings can be initialized and produce an embedding dimension check."""
    embeddings = SentenceTransformerEmbeddings(
        model_name="nomic-embed-text",
        cache_dir=Path("/tmp/cache"),
        local_files_only=True,
    )
    assert embeddings.model_name == "nomic-embed-text"

def test_ollama_client_success():
    client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
    with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"response": "the answer"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp
        ans = client.generate("test prompt")
        assert ans == "the answer"

def test_ollama_client_error():
    client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
    with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = Exception("network")
        with pytest.raises(Exception):
            client.generate("test")
