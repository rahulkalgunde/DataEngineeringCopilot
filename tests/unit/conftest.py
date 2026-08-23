import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np  # noqa: F401 — import early prevents fork-related ImportError
import pytest

from data_engineering_copilot.evaluation.chunking_gold import (
    ChunkingGoldDoc,
    ChunkingGoldSpan,
    validate_gold_doc,
)

_ALLOWED_SOCKET_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


@pytest.fixture(autouse=True)
def _block_external_sockets(request, monkeypatch):
    """Fail fast when a unit test opens a non-loopback TCP connection.

    Mirrors Haystack's ``request_blocker`` / LangChain's ``--disable-socket``
    habit: unit tests are hermetic, so an outbound connection means a mock was
    forgotten and a real provider/API would be hit (hang or paid call in CI).
    Loopback and AF_UNIX stay allowed so local servers and unix transports
    keep working. Integration-marked tests collected here are exempt.
    """
    import socket

    if request.node.get_closest_marker("integration"):
        yield
        return

    real_socket = socket.socket
    real_create_connection = socket.create_connection

    def _assert_local(address: object) -> None:
        host = address[0] if isinstance(address, tuple) else None
        if isinstance(host, str) and host.lower() not in _ALLOWED_SOCKET_HOSTS:
            raise RuntimeError(
                f"unit test attempted external connection to {host!r} — "
                "mock the transport (respx/aresponses/fake double) instead"
            )

    class _GuardedSocket(real_socket):
        def connect(self, address):  # type: ignore[override]
            _assert_local(address)
            return super().connect(address)

        def connect_ex(self, address):  # type: ignore[override]
            _assert_local(address)
            return super().connect_ex(address)

    def _guarded_socket(*args, **kwargs):
        family = args[0] if args else kwargs.get("family", socket.AF_INET)
        if family in (socket.AF_INET, socket.AF_INET6):
            return _GuardedSocket(*args, **kwargs)
        return real_socket(*args, **kwargs)

    def _guarded_create_connection(address, *args, **kwargs):
        _assert_local(address)
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", _guarded_socket)
    monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)
    yield


@pytest.fixture(autouse=True)
def _mock_sentence_transformers():
    """Prevent real sentence-transformers model loading across all unit tests.

    Without this, the first test that triggers ``from sentence_transformers
    import CrossEncoder`` pays a 3-5s penalty for importing torch and
    transformers wheel metadata.

    NOTE: must NOT use ``patch.dict("sys.modules", ...)`` here — on exit it
    clears ``sys.modules`` and restores the enter-time snapshot, which erases
    any module first imported during the test (e.g. ``datasets``/``pyarrow``
    via ragas). Patch only the single key instead.
    """
    mock_module = MagicMock()
    mock_module.CrossEncoder = MagicMock(return_value=MagicMock())
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_module
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original


@pytest.fixture
def mock_vector_store():
    m = MagicMock()
    m.upsert_chunks = AsyncMock()
    m.query = AsyncMock()
    m.count = AsyncMock()
    m.count_urls = AsyncMock(return_value=0)
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


@pytest.fixture
def gold_chunking_dataset():
    """Load the committed chunking gold fixtures (synthetic + human slices)."""
    base = Path("tests/evaluation/golden/chunking")
    docs: list[ChunkingGoldDoc] = []
    for name in ["synthetic_gold.jsonl", "human_slice.jsonl"]:
        path = base / name
        if not path.exists():
            continue
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                doc = ChunkingGoldDoc(
                    doc_id=data["doc_id"],
                    text=data["text"],
                    gold_spans=[ChunkingGoldSpan(**s) for s in data["gold_spans"]],
                )
                validate_gold_doc(doc)
                docs.append(doc)
    return docs
