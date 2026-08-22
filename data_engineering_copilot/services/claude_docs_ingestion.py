"""Claude documentation ingestion via first-party ``llms.txt`` indexes.

High-value / low-effort acquisition path for the Claude Platform (API) and
Claude Code documentation:

1. Download the site's ``llms.txt`` index (each entry links a raw ``.md`` file).
2. Download every linked ``.md`` file at a bounded concurrency, caching under
   ``data/raw_sources/claude_docs/<site>/``.
3. Build ``ParsedDocument`` objects and hand them to the existing chunker /
   embedder / vector store via dependency injection (no provider calls here).

The chunker, embedder, and store are injected so tests can use hermetic fakes
and the CLI can wire the factory-built real components.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.infrastructure.token_budget import (
    _ENCODER,
    DEFAULT_MAX_TOKENS,
    TokenEncoder,
    coalesce_blank_segments,
    count_tokens,
    split_text_losslessly,
)

logger = logging.getLogger(__name__)
_structlog = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw_sources" / "claude_docs"

SOURCE_PLATFORM = "Claude Platform Docs"
SOURCE_CODE = "Claude Code Docs"

LLMS_DOC_SITES: dict[str, dict[str, str]] = {
    "platform": {
        "llms_url": "https://platform.claude.com/docs/llms.txt",
        "source_name": SOURCE_PLATFORM,
        "url_prefix": "https://platform.claude.com/docs/en/",
    },
    "code": {
        "llms_url": "https://code.claude.com/docs/llms.txt",
        "source_name": SOURCE_CODE,
        "url_prefix": "https://code.claude.com/docs/en/",
    },
}

_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_FRONTMATTER_RE = re.compile(r"^\ufeff?---\r?\n.*?\r?\n---\r?\n?", re.DOTALL)
_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MIN_DOC_CHARS = 100
# Texts per ``embed_texts`` call. Matches the embedding client's batch capacity
# (``embedding_batch_size``). 64 is the measured sweet spot across NVIDIA /
# OpenRouter / local-hf: high throughput without oversized payloads that make
# a single failed request (503) lose a lot of work, or spike CPU/RAM on the
# local model.
_EMBED_BATCH_SIZE = 64
# Job-level retry sleeps for a batch whose providers are all down (both
# providers 5xx'd and the fallback chain raised ``LLMClientError``). Each sleep
# must outlast the TEMPORARY_UNAVAILABLE cooldown (10s) the health registry set
# on the failed providers, otherwise the chain gate would skip them again
# instantly. These are the bulk-ingest analogue of the per-request tenacity
# backoff that lives inside each provider client. Because the NVIDIA→OpenRouter
# embedding fallback shares the same NVIDIA backend (OpenRouter relays
# nemotron-3-embed-1b), a simultaneous 503 is a backend blip that can last many
# minutes — the sleeps are generous (≈15 min total budget) so the job waits it
# out instead of dying; the resume + per-doc flush make long waits safe.
_EMBED_RETRY_SLEEPS = (60.0, 120.0, 240.0, 480.0)


def parse_llms_index(text: str, url_prefix: str) -> list[tuple[str, str]]:
    """Parse ``- [Title](url.md)`` lines from an ``llms.txt`` index.

    Only links under *url_prefix* that end with ``.md`` are kept (single-page
    guides are raw markdown; external/detail links are discarded). Returns
    ``(title, url)`` pairs in document order.
    """
    entries: list[tuple[str, str]] = []
    for match in _LINK_RE.finditer(text):
        title = match.group(1).strip()
        link = match.group(2).strip()
        if link.startswith(url_prefix) and link.endswith(".md"):
            entries.append((title, link))
    return entries


def strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block from *text* if present."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def _url_to_relpath(url: str, url_prefix: str) -> str:
    if url.startswith(url_prefix):
        return url[len(url_prefix) :]
    return urlparse(url).path.lstrip("/")


def build_parsed_documents(site: str, entries: Sequence[tuple[str, str]], root_dir: Path) -> list[ParsedDocument]:
    """Read fetched ``.md`` files and build ``ParsedDocument`` objects.

    Files shorter than ``_MIN_DOC_CHARS`` (after frontmatter stripping) are
    treated as redirect/placeholder stubs and skipped.
    """
    cfg = LLMS_DOC_SITES[site]
    documents: list[ParsedDocument] = []
    for title, url in entries:
        relpath = _url_to_relpath(url, cfg["url_prefix"])
        path = Path(root_dir) / relpath
        if not path.is_file():
            continue
        text = strip_frontmatter(path.read_text(encoding="utf-8"))
        if len(text.strip()) < _MIN_DOC_CHARS:
            continue
        documents.append(
            ParsedDocument(
                source_name=cfg["source_name"],
                title=title,
                url=url,
                text=text,
                doc_type="api_reference" if relpath.split("/")[0] == "api" else "guide",
                file_path=relpath,
            )
        )
    return documents


async def fetch_llms_index(site: str, client: httpx.AsyncClient | None = None) -> list[tuple[str, str]]:
    """Download and parse the ``llms.txt`` index for *site*."""
    cfg = LLMS_DOC_SITES[site]
    if client is None:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as own_client:
            return await _download_index(cfg, own_client)
    return await _download_index(cfg, client)


async def _download_index(cfg: dict[str, str], client: httpx.AsyncClient) -> list[tuple[str, str]]:
    resp = await client.get(cfg["llms_url"], headers=_default_headers())
    resp.raise_for_status()
    return parse_llms_index(resp.text, cfg["url_prefix"])


def _default_headers() -> dict[str, str]:
    return {"User-Agent": "DataEngineeringCopilot/1.0"}


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]


async def fetch_markdown_files(
    site: str,
    entries: Sequence[tuple[str, str]],
    dest_dir: Path,
    concurrency: int = 8,
    max_docs: int | None = None,
) -> tuple[list[Path], list[str]]:
    """Download raw markdown files for *entries* under *dest_dir*.

    Already-existing files are skipped (idempotent re-runs). Retryable failures
    (429/5xx/transport) are retried with backoff; permanent HTTP failures are
    logged and skipped. Returns ``(downloaded_paths, failed_urls)``.
    """
    cfg = LLMS_DOC_SITES[site]
    limited = entries[:max_docs] if max_docs is not None else list(entries)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    failed: list[str] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _fetch_one(title: str, url: str) -> None:
        relpath = _url_to_relpath(url, cfg["url_prefix"])
        out_path = dest_dir / relpath
        async with semaphore:
            if out_path.exists():
                downloaded.append(out_path)
                return
            for attempt in range(len(_BACKOFF_SECONDS) + 1):
                try:
                    resp = await client.get(url, headers=_default_headers())
                    if resp.status_code in _RETRYABLE_STATUSES and (attempt + 1) < len(_BACKOFF_SECONDS):
                        await asyncio.sleep(_backoff(attempt, resp.headers.get("Retry-After")))
                        continue
                    resp.raise_for_status()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(resp.content)
                    downloaded.append(out_path)
                    return
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in _RETRYABLE_STATUSES and (attempt + 1) < len(_BACKOFF_SECONDS):
                        await asyncio.sleep(_backoff(attempt))
                        continue
                    failed.append(url)
                    logger.warning("claude_docs.fetch_failed status=%s url=%s", exc.response.status_code, url)
                    return
                except (httpx.TransportError, httpx.TimeoutException):
                    if (attempt + 1) < len(_BACKOFF_SECONDS):
                        await asyncio.sleep(_backoff(attempt))
                        continue
                    failed.append(url)
                    logger.warning("claude_docs.fetch_transport_failed url=%s", url)
                    return

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        await asyncio.gather(*(_fetch_one(title, url) for title, url in limited))

    logger.info(
        "claude_docs.fetch_complete site=%s total=%d downloaded=%d skipped_existing=%d failed=%d",
        site,
        len(limited),
        len(downloaded),
        len(limited) - len(downloaded) - len(failed),
        len(failed),
    )
    return downloaded, failed


async def ingest_claude_docs(
    sites: Sequence[str],
    max_docs: int | None,
    chunker: object,
    embedder: object,
    store: object,
    raw_root: Path | None = None,
    encoder: TokenEncoder | None = None,
    skip_indexed: bool = True,
) -> dict[str, object]:
    """Fetch, parse, chunk, embed, and upsert Claude docs end to end.

    *chunker* must expose ``async chunk(ParsedDocument) -> list[DocumentChunk]``,
    *embedder* ``async embed_texts(list[str]) -> list[list[float]]``, and *store*
    ``async upsert_chunks(chunks, vectors)`` and ``async scroll_urls(source_name)``.
    No BM25 fitting is performed here (the shared collection may already hold
    sparse vectors from other sources). *encoder* (optional) is threaded to the
    lossless splitter so segments stay within the same token budget the
    embedder pre-flights against. With ``skip_indexed=True`` (default),
    documents whose URL is already stored in Qdrant are skipped so re-runs only
    embed the not-yet-indexed docs (resumable across provider outages).
    """
    root = raw_root or DEFAULT_RAW_ROOT
    all_documents: list[ParsedDocument] = []
    per_source: dict[str, int] = {}
    total_failed = 0
    for site in sites:
        entries = await fetch_llms_index(site)
        _downloaded, failed = await fetch_markdown_files(site, entries, root / site, max_docs=max_docs)
        total_failed += len(failed)
        limited = entries[:max_docs] if max_docs is not None else entries
        documents = build_parsed_documents(site, limited, root / site)
        source_name = LLMS_DOC_SITES[site]["source_name"]
        if skip_indexed:
            indexed_urls = set(await store.scroll_urls(source_name))  # type: ignore[attr-defined]  # injected store
            before = len(documents)
            documents = [d for d in documents if d.url not in indexed_urls]
            if documents:
                logger.info(
                    "claude_docs.skip_indexed site=%s source=%s total=%d already_indexed=%d to_embed=%d",
                    site,
                    source_name,
                    before,
                    before - len(documents),
                    len(documents),
                )
        all_documents.extend(documents)
        per_source[source_name] = len(documents)

    chunked_docs, total_chunks = await _chunk_embed_upsert(all_documents, chunker, embedder, store, encoder=encoder)
    return {
        "documents": len(all_documents),
        "chunked_documents": chunked_docs,
        "chunks": total_chunks,
        "fetch_failures": total_failed,
        "per_source": per_source,
    }


async def _embed_batch_with_retry(embedder: object, batch: list[str]) -> list[list[float]]:
    """Embed *batch*, retrying when every provider in the fallback chain failed.

    ``embedder`` is a ``FallbackEmbedder`` (or any object exposing
    ``async embed_texts``). When the underlying ``ProviderFallbackChain`` has
    exhausted NVIDIA → OpenRouter and raised ``LLMClientError``, sleep past the
    provider cooldowns and retry the whole batch — the bulk-ingest analogue of
    the tenacity backoff already used inside each provider client. Only
    ``LLMClientError`` is retried: permanent failures (4xx, budget) surface as
    ``EmbeddingError`` and must fail fast.
    """
    from data_engineering_copilot.infrastructure.llm_client import LLMClientError

    for attempt, sleep in enumerate(_EMBED_RETRY_SLEEPS, start=1):
        try:
            return await embedder.embed_texts(batch)  # type: ignore[attr-defined]  # injected embedder
        except LLMClientError as exc:
            _structlog.warning(
                "claude_docs.embed_all_providers_down",
                attempt=attempt,
                sleep=sleep,
                err=str(exc)[:160],
            )
            await asyncio.sleep(sleep)
    # Final attempt after the last sleep: surface the result (or raise).
    return await embedder.embed_texts(batch)  # type: ignore[attr-defined]  # injected embedder


async def _chunk_embed_upsert(
    documents: Sequence[ParsedDocument],
    chunker: object,
    embedder: object,
    store: object,
    encoder: TokenEncoder | None = None,
) -> tuple[int, int]:
    """Chunk, embed, and upsert a batch of documents.

    Returns ``(chunked_document_count, total_chunk_count)``. Each document's
    chunks are flushed to the store immediately after embedding (not buffered to
    the end) so a mid-run provider outage never discards completed documents.
    Oversized chunks are split losslessly into budget-safe segments (mirroring
    the Spark pipeline) so they never exceed the embedder's token budget.
    """
    chunked_docs = 0
    total_chunks = 0
    all_chunks: list[DocumentChunk] = []
    all_vectors: list[list[float]] = []
    all_texts: list[str] = []

    async def _flush() -> None:
        if all_chunks:
            await store.upsert_chunks(all_chunks, all_vectors)  # type: ignore[attr-defined]  # injected store

    for doc in documents:
        chunks = await chunker.chunk(doc)  # type: ignore[attr-defined]  # injected chunker
        if not chunks:
            continue
        normalized = _normalize_chunks(chunks, encoder=encoder)
        for i in range(0, len(normalized), _EMBED_BATCH_SIZE):
            batch = [c.text for c in normalized[i : i + _EMBED_BATCH_SIZE]]
            batch_vectors = await _embed_batch_with_retry(embedder, batch)
            all_chunks.extend(normalized[i : i + _EMBED_BATCH_SIZE])
            all_vectors.extend(batch_vectors)
            all_texts.extend(batch)
        chunked_docs += 1
        total_chunks += len(normalized)
        # Flush after every document so a mid-run outage (provider 5xx) never
        # discards completed work: the buffered tail of a crashed run is the
        # reason re-runs previously made zero durable progress. Combined with
        # ``skip_indexed``, only the still-missing docs get embedded on retry.
        await _flush()
        all_chunks.clear()
        all_vectors.clear()

    await _flush()
    # BM25 parity with the crawler path (which accumulates the tokenizer per
    # run): refit once over every chunk text after the final flush.
    fit = getattr(store, "fit_bm25", None)
    if callable(fit) and all_texts:
        fit(all_texts)
    return chunked_docs, total_chunks


def _normalize_chunks(
    chunks: Sequence[DocumentChunk],
    encoder: TokenEncoder | None = None,
) -> list[DocumentChunk]:
    """Split *chunks* into lossless, budget-safe segments (Spark-pipeline parity).

    Never truncates text; ``"".join(segment_texts)`` reconstructs the normalized
    parent chunk. A chunk already within budget yields a single segment with
    ``segment_index=0``. Deterministic ``chunk_id`` suffixes keep re-runs
    idempotent (each segment maps to a stable point ID). *encoder* (optional)
    is threaded to the token counter so the split budget matches the embedder's
    pre-flight counting.
    """
    normalized: list[DocumentChunk] = []
    effective_encoder = encoder if encoder is not None else _ENCODER
    for chunk in chunks:
        parent_hash = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
        segment_texts = coalesce_blank_segments(
            split_text_losslessly(chunk.text, max_tokens=DEFAULT_MAX_TOKENS, encoder=effective_encoder)
        )
        for index, text in enumerate(segment_texts):
            normalized.append(
                replace(
                    chunk,
                    chunk_id=f"{chunk.chunk_id}:seg:{index}",
                    text=text,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    word_count=len(text.split()),
                    parent_content_hash=parent_hash,
                    segment_index=index,
                    segment_total=len(segment_texts),
                    token_count=count_tokens(text, encoder=effective_encoder),
                    character_count=len(text),
                )
            )
    return normalized
