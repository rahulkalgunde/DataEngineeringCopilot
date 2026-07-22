from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np  # noqa: F401 — import early prevents fork-related ImportError
import pytest


@pytest.fixture(autouse=True)
def _mock_sentence_transformers():
    """Prevent real sentence-transformers model loading across all unit tests.

    Without this, the first test that triggers ``from sentence_transformers
    import CrossEncoder`` pays a 3-5s penalty for importing torch and
    transformers wheel metadata.
    """
    mock_module = MagicMock()
    mock_module.CrossEncoder = MagicMock(return_value=MagicMock())
    with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
        yield


@pytest.fixture(autouse=True)
def _patch_reranker():
    mock_instance = MagicMock()
    mock_instance.is_available.return_value = False
    async_patcher = patch("data_engineering_copilot.services.async_rag.CrossEncoderReranker", return_value=mock_instance)
    async_patcher.start()
    yield
    async_patcher.stop()


@pytest.fixture
def mock_vector_store():
    m = MagicMock()
    m.upsert_chunks = AsyncMock()
    m.query = AsyncMock()
    m.count = AsyncMock()
    m.get_content_hash_for_url = AsyncMock()
    m.delete_by_url = AsyncMock()
    return m


@pytest.fixture
def mock_ollama():
    m = MagicMock()
    m.generate = AsyncMock()
    return m


@pytest.fixture
def mock_embedder():
    m = MagicMock()
    m.embed_texts = AsyncMock()
    m.embed_query = AsyncMock()
    return m
