"""Hermetic tests for Claude docs ingestion (fake embedder/store, no network, no Qdrant)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from data_engineering_copilot import cli
from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.services.claude_docs_ingestion import (
    _EMBED_RETRY_SLEEPS,
    LLMS_DOC_SITES,
    _chunk_embed_upsert,
    _normalize_chunks,
    build_parsed_documents,
    fetch_markdown_files,
    ingest_claude_docs,
    parse_llms_index,
    strip_frontmatter,
)
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker

_PLATFORM_PREFIX = LLMS_DOC_SITES["platform"]["url_prefix"]
_CODE_PREFIX = LLMS_DOC_SITES["code"]["url_prefix"]


def _run(coro):
    return asyncio.run(coro)


class FakeEmbedder:
    """Returns a fixed-dimension vector (deterministic) per text."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [
            [float((index + offset) % (self._dimension + 1)) for offset in range(self._dimension)]
            for index in range(len(texts))
        ]


class FakeStore:
    """Captures ``upsert_chunks(chunks, vectors)`` calls for contract pinning."""

    def __init__(self, indexed_urls: list[str] | None = None) -> None:
        self.upserts: list[tuple[list[DocumentChunk], list[list[float]]]] = []
        self._indexed_urls = set(indexed_urls or [])

    async def upsert_chunks(self, chunks, vectors) -> None:
        self.upserts.append((list(chunks), list(vectors)))

    async def scroll_urls(self, source_name: str) -> list[str]:
        return list(self._indexed_urls)


def _platform_mock_documents() -> list[ParsedDocument]:
    return [
        ParsedDocument(
            source_name=LLMS_DOC_SITES["platform"]["source_name"],
            title="Working with messages",
            url=f"{_PLATFORM_PREFIX}build-with-claude/working-with-messages.md",
            text=(
                "# Using the Messages API\n\n"
                "The Messages API is the primary way to interact with Claude.\n\n"
                "## Tool use\n\n"
                "Define an input schema for tools the model can call during generation.\n\n"
                "## Streaming\n\n"
                "Stream partial responses with SSE and handle content-block deltas as they arrive from the server.\n\n"
                "## Rate limits\n\n"
                "Each model tier exposes distinct per-minute token budgets that govern request throughput."
            ),
            doc_type="guide",
            file_path="build-with-claude/working-with-messages.md",
        )
    ]


# ---------------------------------------------------------------------------
# parse_llms_index
# ---------------------------------------------------------------------------


def test_parse_llms_index_extracts_only_md_links_under_prefix() -> None:
    text = "\n".join(
        [
            "## Docs",
            f"- [Working with messages]({_PLATFORM_PREFIX}build-with-claude/working-with-messages.md) - Example",
            f"- [Raw page]({_PLATFORM_PREFIX}build-with-claude/example.html)",
            "- [External](https://github.com/anthropics/some-repo) - not a doc page",
            "- [De only](https://platform.claude.com/docs/de/intro.md) - wrong language",
        ]
    )
    entries = parse_llms_index(text, _PLATFORM_PREFIX)
    assert entries == [("Working with messages", f"{_PLATFORM_PREFIX}build-with-claude/working-with-messages.md")]


def test_parse_llms_index_handles_code_prefix() -> None:
    text = f"- [Hooks]({_CODE_PREFIX}hooks.md)"
    entries = parse_llms_index(text, _CODE_PREFIX)
    assert entries == [("Hooks", f"{_CODE_PREFIX}hooks.md")]


# ---------------------------------------------------------------------------
# strip_frontmatter
# ---------------------------------------------------------------------------


def test_strip_frontmatter_removes_yaml_block() -> None:
    raw = "---\ntitle: Claude overview\nsidebar_label: Intro\n---\n# Intro\n\nBody text."
    assert strip_frontmatter(raw) == "# Intro\n\nBody text."


def test_strip_frontmatter_passthrough_without_block() -> None:
    raw = "# No frontmatter\n\nJust body."
    assert strip_frontmatter(raw) == raw


# ---------------------------------------------------------------------------
# build_parsed_documents
# ---------------------------------------------------------------------------


def test_build_parsed_documents_maps_fields_and_doc_type(tmp_path: Path) -> None:
    rel = "build-with-claude/working-with-messages.md"
    out = tmp_path / rel
    out.parent.mkdir(parents=True)
    out.write_text(
        "---\ntitle: X\n---\n# Using the Messages API\n\n"
        "The Messages API is the primary way to interact with Claude. "
        "You send a request and receive a streaming response with the model output.",
        encoding="utf-8",
    )
    entries = [("Working with messages", f"{_PLATFORM_PREFIX}{rel}")]

    docs = build_parsed_documents("platform", entries, tmp_path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_name == LLMS_DOC_SITES["platform"]["source_name"]
    assert doc.title == "Working with messages"
    assert doc.url == f"{_PLATFORM_PREFIX}{rel}"
    assert doc.file_path == rel
    assert doc.doc_type == "guide"
    assert doc.text.startswith("# Using the Messages API")


def test_build_parsed_documents_skips_short_stubs(tmp_path: Path) -> None:
    rel = "overview.md"
    out = tmp_path / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# stub", encoding="utf-8")
    entries = [("Overview", f"{_PLATFORM_PREFIX}{rel}")]

    assert build_parsed_documents("platform", entries, tmp_path) == []


def test_build_parsed_documents_api_reference_doc_type(tmp_path: Path) -> None:
    rel = "api/messages.md"
    out = tmp_path / rel
    out.parent.mkdir(parents=True)
    out.write_text(
        "# Messages\n\n" + ("Body words describing the messages endpoint in detail. " * 20), encoding="utf-8"
    )
    entries = [("Messages", f"{_PLATFORM_PREFIX}{rel}")]

    docs = build_parsed_documents("platform", entries, tmp_path)
    assert len(docs) == 1
    assert docs[0].doc_type == "api_reference"


# ---------------------------------------------------------------------------
# chunk -> embed -> upsert contract (real chunker + fakes)
# ---------------------------------------------------------------------------


def test_chunk_embed_upsert_pins_contract() -> None:
    chunker = HeaderAwareChunker()
    embedder = FakeEmbedder()
    store = FakeStore()
    documents = _platform_mock_documents()

    chunked_docs, total_chunks = _run(_chunk_embed_upsert(documents, chunker, embedder, store))

    assert chunked_docs == 1
    assert total_chunks >= 1
    assert len(store.upserts) == 1


def test_chunk_embed_upsert_flushes_per_document() -> None:
    """Each document is flushed immediately (not buffered to run end) so a
    mid-run provider outage never discards completed documents."""
    chunker = HeaderAwareChunker()
    embedder = FakeEmbedder()
    store = FakeStore()
    documents = _platform_mock_documents() * 3

    chunked_docs, total_chunks = _run(_chunk_embed_upsert(documents, chunker, embedder, store))

    assert chunked_docs == 3
    assert len(store.upserts) == 3
    assert total_chunks >= 3
    # Every embedded text was flushed; no buffered tail left behind.
    flushed_texts = sum(len(chunks) for chunks, _vectors in store.upserts)
    embedded_texts = sum(len(batch) for batch in embedder.calls)
    assert flushed_texts == embedded_texts == total_chunks
    # Each doc's upsert is internally consistent (chunks == vectors).
    for chunks, vectors in store.upserts:
        assert len(chunks) == len(vectors)
    # Payload-critical fields survive from document to chunk.
    for upsert_chunks, _vectors in store.upserts:
        for chunk in upsert_chunks:
            assert chunk.source_name == LLMS_DOC_SITES["platform"]["source_name"]
            assert chunk.url.startswith(_PLATFORM_PREFIX)
            assert chunk.title == "Working with messages"
            assert chunk.doc_type == "guide"
            assert chunk.file_path == "build-with-claude/working-with-messages.md"


def test_chunk_embed_upsert_retries_when_all_providers_down(monkeypatch) -> None:
    """A batch whose providers all 5xx'd (LLMClientError) is retried with
    backoff sleeps, and the doc still gets upserted once the provider recovers."""
    from data_engineering_copilot.infrastructure.llm_client import LLMClientError

    sleep_log: list[float] = []

    class _FlakyEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self._attempts = 0

        async def embed_texts(self, texts):
            self._attempts += 1
            self.calls.append(texts)
            if self._attempts < 3:
                raise LLMClientError("All providers in fallback chain failed")
            return [
                [float((index + offset) % (self._dimension + 1)) for offset in range(self._dimension)]
                for index in range(len(texts))
            ]

    def _fake_sleep(seconds: float) -> None:
        sleep_log.append(seconds)

    async def _fake_async_sleep(seconds: float) -> None:
        _fake_sleep(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_async_sleep)
    chunker = HeaderAwareChunker()
    embedder = _FlakyEmbedder()
    store = FakeStore()
    documents = _platform_mock_documents()

    chunked_docs, total_chunks = _run(_chunk_embed_upsert(documents, chunker, embedder, store))

    assert chunked_docs == 1
    assert total_chunks >= 1
    assert len(store.upserts) == 1  # doc flushed after retry recovery
    assert embedder._attempts == 3  # 2 failures + 1 recovery
    # Retry sleeps happened before the final successful attempt.
    assert len(sleep_log) == 2
    for seconds in sleep_log:
        assert seconds >= 60.0


def test_chunk_embed_upsert_gives_up_after_all_retries(monkeypatch) -> None:
    """If every retry fails, the batch failure propagates (job aborts loudly)."""
    from data_engineering_copilot.infrastructure.llm_client import LLMClientError

    class _AlwaysDownEmbedder(FakeEmbedder):
        async def embed_texts(self, texts):
            self.calls.append(texts)
            raise LLMClientError("All providers in fallback chain failed")

    async def _noop_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    chunker = HeaderAwareChunker()
    embedder = _AlwaysDownEmbedder()
    store = FakeStore()
    documents = _platform_mock_documents()

    with pytest.raises(LLMClientError):
        _run(_chunk_embed_upsert(documents, chunker, embedder, store))

    # 1 initial + 3 retry attempts, then the final attempt = 4 total.
    assert len(embedder.calls) == 1 + len(_EMBED_RETRY_SLEEPS)
    assert store.upserts == []


def test_chunk_embed_upsert_does_not_retry_permanent_errors(monkeypatch) -> None:
    """Non-retryable failures (EmbeddingError) fail fast with no retry sleeps."""
    from data_engineering_copilot.domain.exceptions import EmbeddingError

    class _BadInputEmbedder(FakeEmbedder):
        async def embed_texts(self, texts):
            self.calls.append(texts)
            raise EmbeddingError("Embedding input exceeds budget")

    monkeypatch.setattr(asyncio, "sleep", lambda seconds: None)
    chunker = HeaderAwareChunker()
    embedder = _BadInputEmbedder()
    store = FakeStore()
    documents = _platform_mock_documents()

    with pytest.raises(EmbeddingError, match="exceeds budget"):
        _run(_chunk_embed_upsert(documents, chunker, embedder, store))

    assert len(embedder.calls) == 1
    assert store.upserts == []


def test_normalize_chunks_splits_oversize_chunks_losslessly() -> None:
    from data_engineering_copilot.infrastructure.token_budget import DEFAULT_MAX_TOKENS, count_tokens

    # One chunk well over the token budget; one small chunk already in budget.
    big_text = "## Section\n\n" + "token " * 2000
    chunks = [
        DocumentChunk(
            chunk_id="claude:test:0000",
            source_name="Claude Platform Docs",
            title="Big",
            url=f"{_PLATFORM_PREFIX}big.md",
            text=big_text,
            doc_type="guide",
            file_path="big.md",
        ),
        DocumentChunk(
            chunk_id="claude:test:0001",
            source_name="Claude Platform Docs",
            title="Small",
            url=f"{_PLATFORM_PREFIX}small.md",
            text="## Small\n\nA short in-budget chunk body.",
            doc_type="guide",
            file_path="small.md",
        ),
    ]

    normalized = _normalize_chunks(chunks)

    # Big chunk must split into budget-safe segments; small chunk stays single.
    big = [c for c in normalized if c.url.endswith("big.md")]
    small = [c for c in normalized if c.url.endswith("small.md")]
    assert len(big) > 1
    assert all(count_tokens(c.text) <= DEFAULT_MAX_TOKENS for c in big)
    assert len(small) == 1
    assert small[0].segment_index == 0
    assert small[0].segment_total == 1
    # Lossless reconstruction + stable deterministic IDs.
    assert "".join(c.text for c in big) == big_text.strip()
    for c in big:
        assert c.chunk_id.startswith("claude:test:0000:seg:")
        assert c.parent_content_hash
        assert c.segment_index >= 0
        assert c.segment_total == len(big)
        assert c.token_count == count_tokens(c.text)


def test_normalize_chunks_threads_custom_encoder() -> None:
    """An injected encoder drives both the split budget and token_count metadata."""
    from data_engineering_copilot.infrastructure.token_budget import DEFAULT_MAX_TOKENS

    class _StubEncoder:
        def encode(self, text: str) -> list[int]:
            # Every 3 characters is "one token" — a deliberately different
            # scale than cl100k so we can prove the encoder is being used.
            return [0] * (len(text) // 3 + 1)

    chunks = [
        DocumentChunk(
            chunk_id="claude:enc:0000",
            source_name="Claude Platform Docs",
            title="Big",
            url=f"{_PLATFORM_PREFIX}big.md",
            text="## Section\n\n" + "token " * 1500,
            doc_type="guide",
            file_path="big.md",
        ),
    ]

    normalized = _normalize_chunks(chunks, encoder=_StubEncoder())

    assert len(normalized) > 1
    # token_count must reflect the stub encoder's scale, not cl100k's.
    for c in normalized:
        assert c.token_count == len(c.text) // 3 + 1
        # Every segment stays within the stub-encoder budget.
        assert c.token_count <= DEFAULT_MAX_TOKENS
    # Lossless reconstruction still holds.
    assert "".join(c.text for c in normalized) == chunks[0].text.strip()


# ---------------------------------------------------------------------------
# fetch_markdown_files (no network: transport injected via monkeypatch)
# ---------------------------------------------------------------------------


def test_fetch_markdown_files_skips_existing(tmp_path: Path, monkeypatch) -> None:
    import httpx

    rel = "build-with-claude/working-with-messages.md"
    dest = tmp_path
    existing = dest / rel
    existing.parent.mkdir(parents=True)
    existing.write_text("old content", encoding="utf-8")

    captured: list[str] = []

    async def _fake_get(self: httpx.AsyncClient, url: str, **kwargs):
        captured.append(url)
        return httpx.Response(200, text="", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

    entries = [("Working with messages", f"{_PLATFORM_PREFIX}{rel}")]
    downloaded, failed = _run(fetch_markdown_files("platform", entries, dest))

    # Existing file short-circuits before any HTTP call.
    assert downloaded == [existing]
    assert failed == []
    assert captured == []


# ---------------------------------------------------------------------------
# ingest_claude_docs orchestration (no network/none faked)
# ---------------------------------------------------------------------------


def test_ingest_claude_docs_returns_summary(monkeypatch) -> None:
    import data_engineering_copilot.services.claude_docs_ingestion as module

    documents = _platform_mock_documents()

    async def _fake_fetch_index(site: str, client=None):
        return [("Working with messages", f"{_PLATFORM_PREFIX}build-with-claude/working-with-messages.md")]

    async def _fake_fetch_files(site, entries, dest_dir, concurrency=8, max_docs=None):
        return [], []

    monkeypatch.setattr(module, "fetch_llms_index", _fake_fetch_index)
    monkeypatch.setattr(module, "fetch_markdown_files", _fake_fetch_files)
    monkeypatch.setattr(module, "build_parsed_documents", lambda site, entries, root_dir: documents)

    chunker = HeaderAwareChunker()
    embedder = FakeEmbedder()
    store = FakeStore()

    summary = _run(ingest_claude_docs(["platform"], None, chunker, embedder, store))

    assert summary["documents"] == 1
    assert summary["chunked_documents"] == 1
    assert summary["chunks"] >= 1  # type: ignore[operator]
    assert summary["fetch_failures"] == 0
    per_source = summary["per_source"]
    assert per_source[LLMS_DOC_SITES["platform"]["source_name"]] == 1  # type: ignore[index]
    assert store.upserts  # at least one upsert flushed


def test_ingest_claude_docs_skips_already_indexed_urls(monkeypatch) -> None:
    """Re-runs skip documents already present in Qdrant (resume across outages)."""
    import data_engineering_copilot.services.claude_docs_ingestion as module

    indexed_url = f"{_PLATFORM_PREFIX}build-with-claude/working-with-messages.md"
    documents = _platform_mock_documents()

    async def _fake_fetch_index(site: str, client=None):
        return [("Working with messages", indexed_url)]

    async def _fake_fetch_files(site, entries, dest_dir, concurrency=8, max_docs=None):
        return [], []

    monkeypatch.setattr(module, "fetch_llms_index", _fake_fetch_index)
    monkeypatch.setattr(module, "fetch_markdown_files", _fake_fetch_files)
    monkeypatch.setattr(module, "build_parsed_documents", lambda site, entries, root_dir: documents)

    chunker = HeaderAwareChunker()
    embedder = FakeEmbedder()
    # The URL is already indexed → the whole doc must be skipped, no embeds.
    store = FakeStore(indexed_urls=[indexed_url])

    summary = _run(ingest_claude_docs(["platform"], None, chunker, embedder, store))

    assert summary["documents"] == 0
    assert summary["chunked_documents"] == 0
    assert summary["chunks"] == 0
    assert embedder.calls == []
    assert store.upserts == []


def test_ingest_claude_docs_can_opt_out_of_skip(monkeypatch) -> None:
    """``skip_indexed=False`` re-embeds even already-indexed docs (force reindex)."""
    import data_engineering_copilot.services.claude_docs_ingestion as module

    indexed_url = f"{_PLATFORM_PREFIX}build-with-claude/working-with-messages.md"
    documents = _platform_mock_documents()

    async def _fake_fetch_index(site: str, client=None):
        return [("Working with messages", indexed_url)]

    async def _fake_fetch_files(site, entries, dest_dir, concurrency=8, max_docs=None):
        return [], []

    monkeypatch.setattr(module, "fetch_llms_index", _fake_fetch_index)
    monkeypatch.setattr(module, "fetch_markdown_files", _fake_fetch_files)
    monkeypatch.setattr(module, "build_parsed_documents", lambda site, entries, root_dir: documents)

    chunker = HeaderAwareChunker()
    embedder = FakeEmbedder()
    store = FakeStore(indexed_urls=[indexed_url])

    summary = _run(ingest_claude_docs(["platform"], None, chunker, embedder, store, skip_indexed=False))

    assert summary["documents"] == 1
    assert embedder.calls
    assert store.upserts


# ---------------------------------------------------------------------------
# cli._claude_source_filter routing
# ---------------------------------------------------------------------------


def test_claude_routing_defaults_to_claude_sources() -> None:
    filtered = cli._claude_source_filter("How do I format tool calls in the Claude Messages API?", None)
    assert filtered == [LLMS_DOC_SITES["platform"]["source_name"], LLMS_DOC_SITES["code"]["source_name"]]


def test_claude_routing_non_matching_returns_none() -> None:
    assert cli._claude_source_filter("What is a Spark DataFrame?", None) is None


def test_claude_routing_explicit_source_wins() -> None:
    filtered = cli._claude_source_filter(
        "claude messages api",
        ["Apache Spark Documentation"],
    )
    assert filtered == ["Apache Spark Documentation"]


def test_ingest_claude_docs_parser_flags() -> None:
    args = cli.build_parser().parse_args(["ingest-claude-docs", "--site", "code", "--max-docs", "5"])
    assert args.site == "code"
    assert args.max_docs == 5

    args_default = cli.build_parser().parse_args(["ingest-claude-docs"])
    assert args_default.site == "all"
    assert args_default.max_docs is None
