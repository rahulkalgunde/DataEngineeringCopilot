"""Test suite for embeddings module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings


class TestSentenceTransformerEmbeddingsOllama:
    """Test Ollama embedding provider."""

    @pytest.fixture
    def ollama_embeddings(self):
        """Create an Ollama embeddings instance."""
        return SentenceTransformerEmbeddings(
            model_name="nomic-embed-text",
            cache_dir=Path("/tmp/cache"),
            local_files_only=True,
        )

    def test_ollama_initialization(self, ollama_embeddings):
        """Test that Ollama embeddings initializes correctly."""
        assert ollama_embeddings.model_name == "nomic-embed-text"
        assert ollama_embeddings.ollama_base_url == settings.ollama_base_url.rstrip("/")

    @patch("urllib.request.urlopen")
    def test_ollama_embed_single_text(self, mock_urlopen, ollama_embeddings):
        """Test embedding a single text via Ollama /api/embed endpoint."""
        # Mock the response from Ollama /api/embed endpoint
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        embedding_vector = [0.1] * 768  # nomic-embed-text produces 768-dim vectors
        response_data = {"embeddings": [embedding_vector]}
        mock_response.read.return_value = json.dumps(response_data).encode("utf-8")

        # Mock json.load to return the response data
        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            result = ollama_embeddings._ollama_embed(["test text"])

            assert len(result) == 1
            assert result[0] == embedding_vector
            assert len(result[0]) == 768

    @patch("urllib.request.urlopen")
    def test_ollama_embed_multiple_texts(self, mock_urlopen, ollama_embeddings):
        """Test embedding multiple texts via Ollama /api/embed endpoint."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        embedding_vectors = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        response_data = {"embeddings": embedding_vectors}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            texts = ["text 1", "text 2", "text 3"]
            result = ollama_embeddings._ollama_embed(texts)

            assert len(result) == 3
            assert result == embedding_vectors

    @patch("urllib.request.urlopen")
    def test_ollama_embed_uses_correct_endpoint(self, mock_urlopen, ollama_embeddings):
        """Test that the correct /api/embed endpoint is used (not deprecated /api/embeddings)."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        response_data = {"embeddings": [[0.1] * 768]}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            ollama_embeddings._ollama_embed(["test"])

            # Verify the correct endpoint was called
            call_args = mock_urlopen.call_args
            request_obj = call_args[0][0]
            assert "/api/embed" in request_obj.full_url
            assert "/api/embeddings" not in request_obj.full_url

    @patch("urllib.request.urlopen")
    def test_ollama_embed_request_payload(self, mock_urlopen, ollama_embeddings):
        """Test that the request payload is correctly formatted."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        response_data = {"embeddings": [[0.1] * 768]}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            texts = ["test text"]
            ollama_embeddings._ollama_embed(texts)

            # Verify the request payload
            call_args = mock_urlopen.call_args
            request_obj = call_args[0][0]
            payload = json.loads(request_obj.data.decode("utf-8"))

            assert payload["model"] == "nomic-embed-text"
            assert payload["input"] == texts

    @patch("urllib.request.urlopen")
    def test_ollama_embed_missing_embeddings_key(self, mock_urlopen, ollama_embeddings):
        """Test error handling when response is missing 'embeddings' key."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        # Old API response format (singular 'embedding')
        response_data = {"embedding": [0.1] * 768}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                ollama_embeddings._ollama_embed(["test"])

            assert "missing 'embeddings' key" in str(exc_info.value)
            assert "embedding" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_ollama_embed_embeddings_not_list(self, mock_urlopen, ollama_embeddings):
        """Test error handling when embeddings value is not a list."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        response_data = {"embeddings": "not a list"}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                ollama_embeddings._ollama_embed(["test"])

            assert "not a list" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_ollama_embed_count_mismatch(self, mock_urlopen, ollama_embeddings):
        """Test error handling when embedding count doesn't match text count."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        # Return only 1 embedding for 3 texts
        response_data = {"embeddings": [[0.1] * 768]}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                ollama_embeddings._ollama_embed(["text1", "text2", "text3"])

            assert "returned 1 embeddings for 3 input texts" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_ollama_embed_empty_embedding(self, mock_urlopen, ollama_embeddings):
        """Test error handling when embedding is empty."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        response_data = {"embeddings": [[]]}  # Empty embedding

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                ollama_embeddings._ollama_embed(["test"])

            assert "empty (dimension 0)" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_ollama_embed_wrong_dimension(self, mock_urlopen, ollama_embeddings):
        """Test error handling when embedding has wrong dimension."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        # Return embedding with wrong dimension (512 instead of 768)
        response_data = {"embeddings": [[0.1] * 512]}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                ollama_embeddings._ollama_embed(["test"])

            assert "dimension 512" in str(exc_info.value)
            assert "expected 768" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_ollama_embed_network_error(self, mock_urlopen, ollama_embeddings):
        """Test error handling for network failures."""
        mock_urlopen.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError) as exc_info:
            ollama_embeddings._ollama_embed(["test"])

        assert "Failed to get embeddings from Ollama" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_embed_texts_with_ollama(self, mock_urlopen, ollama_embeddings):
        """Test embed_texts method with Ollama backend."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        embedding_vectors = [[0.1] * 768, [0.2] * 768]
        response_data = {"embeddings": embedding_vectors}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            texts = ["text 1", "text 2"]
            result = ollama_embeddings.embed_texts(texts)

            assert len(result) == 2
            assert result == embedding_vectors

    @patch("urllib.request.urlopen")
    def test_embed_query_with_ollama(self, mock_urlopen, ollama_embeddings):
        """Test embed_query method with Ollama backend."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        embedding_vector = [0.1] * 768
        response_data = {"embeddings": [embedding_vector]}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            result = ollama_embeddings.embed_query("test query")

            assert result == embedding_vector
            assert len(result) == 768

    @patch("urllib.request.urlopen")
    def test_embed_query_empty_result(self, mock_urlopen, ollama_embeddings):
        """Test error handling when embed_query returns None embedding."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        # None value will be caught by dimension validation (not a list)
        response_data = {"embeddings": [None]}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                ollama_embeddings.embed_query("test query")

            # The validation catches None before embed_query check
            assert "not a list" in str(exc_info.value) or "empty result" in str(exc_info.value)

    def test_slice_texts_into_batches_single_batch(self, ollama_embeddings):
        """Test batch slicing with texts that fit in single batch."""
        texts = ["text1", "text2", "text3"]
        batches = ollama_embeddings._slice_texts_into_batches(texts, batch_size=10)

        assert len(batches) == 1
        assert batches[0] == texts

    def test_slice_texts_into_batches_multiple_batches(self, ollama_embeddings):
        """Test batch slicing with texts that span multiple batches."""
        texts = [f"text{i}" for i in range(100)]
        batches = ollama_embeddings._slice_texts_into_batches(texts, batch_size=32)

        assert len(batches) == 4  # 100 / 32 = 3.125, so 4 batches
        assert len(batches[0]) == 32
        assert len(batches[1]) == 32
        assert len(batches[2]) == 32
        assert len(batches[3]) == 4  # Remainder

    def test_slice_texts_into_batches_exact_boundary(self, ollama_embeddings):
        """Test batch slicing with texts that exactly match batch boundaries."""
        texts = [f"text{i}" for i in range(64)]
        batches = ollama_embeddings._slice_texts_into_batches(texts, batch_size=32)

        assert len(batches) == 2
        assert len(batches[0]) == 32
        assert len(batches[1]) == 32

    def test_slice_texts_into_batches_single_item(self, ollama_embeddings):
        """Test batch slicing with single text."""
        texts = ["single text"]
        batches = ollama_embeddings._slice_texts_into_batches(texts, batch_size=32)

        assert len(batches) == 1
        assert batches[0] == texts

    def test_slice_texts_into_batches_empty_list(self, ollama_embeddings):
        """Test batch slicing with empty text list."""
        texts = []
        batches = ollama_embeddings._slice_texts_into_batches(texts, batch_size=32)

        assert len(batches) == 0

    def test_slice_texts_into_batches_invalid_batch_size(self, ollama_embeddings):
        """Test batch slicing with invalid batch size."""
        texts = ["text1", "text2"]

        with pytest.raises(ValueError) as exc_info:
            ollama_embeddings._slice_texts_into_batches(texts, batch_size=0)

        assert "batch_size must be positive" in str(exc_info.value)

    @patch("data_engineering_copilot.infrastructure.embeddings.json.load")
    @patch("urllib.request.urlopen")
    def test_ollama_embed_with_batch_slicing_319_items(self, mock_urlopen, mock_json_load, ollama_embeddings):
        """Test embedding 319 items (the problematic case) with batch slicing."""
        # Create 319 texts
        texts = [f"text{i}" for i in range(319)]

        # Mock responses for each batch (319 / 32 = 10 batches)
        def mock_urlopen_side_effect(request):
            mock_response = MagicMock()
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            return mock_response

        def mock_json_load_side_effect(response):
            # Parse the request from the urlopen call
            request = mock_urlopen.call_args[0][0]
            payload = json.loads(request.data.decode("utf-8"))
            batch_texts = payload["input"]
            batch_size = len(batch_texts)

            # Create embeddings for this batch
            embedding_vectors = [[0.1] * 768 for _ in range(batch_size)]
            return {"embeddings": embedding_vectors}

        mock_urlopen.side_effect = mock_urlopen_side_effect
        mock_json_load.side_effect = mock_json_load_side_effect

        result = ollama_embeddings._ollama_embed(texts)

        # Verify we got embeddings for all 319 texts
        assert len(result) == 319
        assert all(len(emb) == 768 for emb in result)

        # Verify multiple requests were made (batch slicing occurred)
        assert mock_urlopen.call_count > 1

    @patch("data_engineering_copilot.infrastructure.embeddings.json.load")
    @patch("urllib.request.urlopen")
    def test_ollama_embed_batch_order_preserved(self, mock_urlopen, mock_json_load, ollama_embeddings):
        """Test that batch slicing preserves order of embeddings."""
        texts = [f"text{i}" for i in range(100)]

        def mock_urlopen_side_effect(request):
            mock_response = MagicMock()
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            return mock_response

        def mock_json_load_side_effect(response):
            # Parse the request from the urlopen call
            request = mock_urlopen.call_args[0][0]
            payload = json.loads(request.data.decode("utf-8"))
            batch_texts = payload["input"]

            # Create unique embeddings for each text (use index as first value)
            embedding_vectors = []
            for _i, text in enumerate(batch_texts):
                # Extract the number from "textN" to create unique embedding
                text_num = int(text.replace("text", ""))
                emb = [float(text_num)] + [0.1] * 767
                embedding_vectors.append(emb)

            return {"embeddings": embedding_vectors}

        mock_urlopen.side_effect = mock_urlopen_side_effect
        mock_json_load.side_effect = mock_json_load_side_effect

        result = ollama_embeddings._ollama_embed(texts)

        # Verify order is preserved by checking first value of each embedding
        for i, emb in enumerate(result):
            assert emb[0] == float(i), f"Embedding {i} has wrong order"

    @patch("urllib.request.urlopen")
    def test_ollama_embed_single_batch_no_slicing(self, mock_urlopen, ollama_embeddings):
        """Test that single batch doesn't trigger multiple requests."""
        texts = ["text1", "text2", "text3"]

        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        embedding_vectors = [[0.1] * 768 for _ in range(3)]
        response_data = {"embeddings": embedding_vectors}

        with patch("json.load", return_value=response_data):
            mock_urlopen.return_value = mock_response

            result = ollama_embeddings._ollama_embed(texts)

            # Should only make one request for single batch
            assert mock_urlopen.call_count == 1
            assert len(result) == 3
