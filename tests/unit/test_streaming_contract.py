"""Contract tests for ``AsyncRagService.answer_stream``.

Unlike ``test_streaming_integration.py`` (which mocks ``answer_stream`` to
exercise the HTTP/SSE layer), these tests execute the *real* ``answer_stream``
body against deterministic doubles so the internal pipeline is covered.  A
final AST-level guard ensures no test in the suite swaps out ``answer_stream``
with a fake for the happy path (error-injection cases in the integration file
are explicitly allowlisted).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService
from tests.doubles.embedder import StubEmbedder
from tests.doubles.llm import StubLLM
from tests.doubles.vector_store import InMemoryVectorStore

_TOPICS = [
    (
        "Apache Spark",
        "Apache Spark is a unified analytics engine for large-scale data processing. "
        "It provides high-level APIs in Scala, Java, Python, and R.",
    ),
    (
        "Delta Lake",
        "Delta Lake is an open-source storage framework that brings ACID transactions "
        "to Apache Spark and big data workloads.",
    ),
    (
        "Apache Airflow",
        "Apache Airflow is a platform to programmatically author, schedule and monitor workflows defined as code.",
    ),
]


def _build_chunks() -> list[DocumentChunk]:
    chunks = []
    for i, (title, text) in enumerate(_TOPICS):
        chunks.append(
            DocumentChunk(
                chunk_id=f"stream:doc{i:03d}:chunk00",
                source_name="RAG Test Docs",
                title=title,
                url=f"https://example.com/docs/{title.lower().replace(' ', '-')}.html",
                text=text,
            )
        )
    return chunks


@pytest.fixture
async def _stream_rag():
    """Hermetic RAG service with a real QueryRewriter wired in."""
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)

    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    service = AsyncRagService(
        config=RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.05,
            max_context_chars=2000,
        ),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        query_rewriter=QueryRewriter(llm_client=StubLLM(), enabled=True, hyde_enabled=False),
    )
    yield service
    await embedder.close()
    await store.close()


def _collect_events(stream: list[str]) -> list[dict]:
    import json

    events = []
    for line in stream:
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.mark.asyncio
async def test_answer_stream_executes_real_body(_stream_rag):
    """The real ``answer_stream`` body runs end-to-end and emits a done event."""
    events = _collect_events([e async for e in _stream_rag.answer_stream("What is Apache Spark?")])
    types = [e["type"] for e in events]
    assert types[0] == "status", "Pipeline should emit status events first"
    assert "done" in types, f"Expected a done event, got {types}"
    done = next(e for e in events if e["type"] == "done")
    assert isinstance(done["text"], str) and len(done["text"]) > 0
    assert isinstance(done["confidence"], float)


@pytest.mark.asyncio
async def test_answer_stream_emits_token_events(_stream_rag):
    """LLM tokens are streamed as token events while retrieval proceeds."""
    events = _collect_events([e async for e in _stream_rag.answer_stream("What is Delta Lake?")])
    tokens = [e for e in events if e["type"] == "token"]
    assert tokens, "Expected token events from generate_stream"
    assert all("content" in t and isinstance(t["content"], str) for t in tokens)


@pytest.mark.asyncio
async def test_answer_stream_survives_rewriter_failure(_stream_rag):
    """A failing rewriter must not abort the stream; raw query is used instead."""

    class _BoomRewriter:
        async def async_rewrite(self, query: str):
            raise RuntimeError("rewriter exploded")

    _stream_rag.query_rewriter = _BoomRewriter()
    events = _collect_events([e async for e in _stream_rag.answer_stream("What is Apache Airflow?")])
    types = [e["type"] for e in events]
    assert "done" in types, f"Stream must complete despite rewriter failure, got {types}"
    done = next(e for e in events if e["type"] == "done")
    assert len(done["text"]) > 0


def _enclosing_function(node: ast.AST, module: ast.Module) -> str | None:
    """Name of the innermost enclosing FunctionDef containing *node*."""
    for parent in ast.walk(module):
        if isinstance(parent, ast.FunctionDef):
            for child in ast.walk(parent):
                if child is node:
                    return parent.name
    return None


# Error-injection cases in the integration file that deliberately replace
# answer_stream with a crashing coroutine.  Everything else that assigns to
# answer_stream is banned so the real body always runs somewhere.
_ALLOWED_MOCKING_TESTS = {"test_streaming_error_emits_error_event", "test_streaming_timeout_emits_timeout_error"}


@pytest.mark.unit
def test_no_test_mocks_answer_stream():
    """No test may replace ``answer_stream`` with a fake except the explicit
    error-injection cases in test_streaming_integration.py."""
    tests_dir = pathlib.Path(__file__).resolve().parents[1]
    violations = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        if path.name == "test_streaming_contract.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if not isinstance(target, ast.Attribute):
                        continue
                    if target.attr == "answer_stream":
                        enclosing = _enclosing_function(node, tree)
                        if path.name == "test_streaming_integration.py" and enclosing in _ALLOWED_MOCKING_TESTS:
                            continue
                        violations.append(f"{path}:{node.lineno}: replaces answer_stream")
    assert not violations, "answer_stream must not be mocked for happy-path tests:\n" + "\n".join(violations)
