from unittest.mock import MagicMock, patch

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
    rag_patcher = patch("data_engineering_copilot.services.rag.CrossEncoderReranker", return_value=mock_instance)
    async_patcher = patch("data_engineering_copilot.services.async_rag.CrossEncoderReranker", return_value=mock_instance)
    rag_patcher.start()
    async_patcher.start()
    yield
    async_patcher.stop()
    rag_patcher.stop()


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def mock_ollama():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    return MagicMock()
