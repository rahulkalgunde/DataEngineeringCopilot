from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import pathlib
import sys
import urllib.error
import urllib.request
from typing import cast

from data_engineering_copilot.cli_catalog import main as catalog_probe_main
from data_engineering_copilot.cli_llm_probe import main as llm_probe_main
from data_engineering_copilot.cli_monitor import main as monitor_main
from data_engineering_copilot.config.logging import setup_logging
from data_engineering_copilot.config.naming import resolve_naming, validate_naming
from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.domain.protocols import EmbedderProtocol
from data_engineering_copilot.evaluation.langfuse_metrics import query_aliases
from data_engineering_copilot.infrastructure.token_budget import TokenEncoder
from data_engineering_copilot.profiler import cli as profiler_cli
from data_engineering_copilot.services.spark_index_builder import CoverageRecord

logger = logging.getLogger(__name__)


def _check_deps_before_dispatch(api_url: str = "http://localhost:8000") -> None:
    """Pre-flight check: verify the API's Docker image is not stale before dispatching."""
    try:
        req = urllib.request.Request(f"{api_url}/api/v1/version")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("deps_fingerprint_ok") is False:
                msg = data.get("deps_stale_message", "Docker image is stale.")
                print(f"\n{'=' * 60}")
                print(f"ERROR: {msg}")
                print(f"{'=' * 60}\n")
                sys.exit(1)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass  # API unreachable — let the dispatch fail naturally


def _poll_gave_up(task_id: str, cancel_url: str) -> None:
    """Print guidance after the status poll retries are exhausted.

    The ingestion task keeps running server-side; the CLI just cannot reach
    the API. Point the user at the monitor and cancel endpoints instead of
    failing with a traceback.
    """
    print(f"\n{'=' * 60}")
    print("Could not reach the ingestion API after several attempts.")
    print("The ingestion task is still running server-side.")
    print(f"  - Watch progress:  dec ingestion-monitor --task-id {task_id}")
    print(f"  - Cancel task:     dec cancel {task_id}")
    print(f"  - Direct cancel:   curl -X POST {cancel_url}")
    print(f"{'=' * 60}\n")


def ingest(max_pages: int | None, source_names: tuple[str, ...] | None) -> None:
    import time

    API_BASE_URL = "http://localhost:8000"

    logger.info("CLI async ingest started max_pages=%s sources=%s", max_pages, source_names or "all")

    # Pre-flight: refuse to dispatch if the Docker image is stale
    _check_deps_before_dispatch(API_BASE_URL)

    # Dispatch through the production API path (Celery task + Redis tracking)
    payload = json.dumps(
        {
            "source_names": list(source_names) if source_names else None,
            "max_pages": max_pages,
        }
    ).encode()
    req = urllib.request.Request(
        f"{API_BASE_URL}/api/v1/ingest",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            task_id = data.get("task_id")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ingestion dispatch failed (HTTP {exc.code}): {body}") from exc
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Cannot reach the API server at {API_BASE_URL}: {exc}\n"
            "Start it with: docker compose up -d backend-api celery_worker"
        ) from exc

    if not task_id:
        raise RuntimeError(f"API did not return a task_id: {data}")

    print(f"Dispatched ingestion task {task_id}")
    print(f"Polling status (Ctrl-C to stop; cancel via: dec cancel {task_id})")

    # Poll progress until completion. Transient failures (timeouts, network
    # errors, 5xx) are retried with backoff so a single slow status response
    # does not kill the CLI while the ingestion task keeps running.
    last_status = None
    cancel_url = f"{API_BASE_URL}/api/v1/ingest/{task_id}/cancel"
    consecutive_failures = 0
    try:
        while True:
            status_req = urllib.request.Request(f"{API_BASE_URL}/api/v1/ingest/status/{task_id}")
            try:
                with urllib.request.urlopen(status_req, timeout=15) as resp:
                    progress = json.loads(resp.read().decode())
                consecutive_failures = 0
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    progress = None
                elif exc.code >= 500:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        _poll_gave_up(task_id, cancel_url)
                        return
                    time.sleep(2 * consecutive_failures)
                    continue
                else:
                    raise
            except (TimeoutError, OSError):
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    _poll_gave_up(task_id, cancel_url)
                    return
                time.sleep(2 * consecutive_failures)
                continue

            if progress is not None:
                status = progress.get("status")
                if status != last_status:
                    print(
                        f"  Status: {status} | "
                        f"Pages: {progress.get('pages_fetched', 0)} | "
                        f"Chunks: {progress.get('chunks_indexed', 0)}"
                    )
                    last_status = status
                if status in ("COMPLETED", "FAILED", "CANCELLED"):
                    err = progress.get("error")
                    if err:
                        print(f"Ingestion finished with error: {err}")
                    else:
                        print(f"Ingestion completed: {progress.get('chunks_indexed', 0)} chunks indexed.")
                    return
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nCancelling ingestion task...")
        try:
            cancel_req = urllib.request.Request(cancel_url, method="POST")
            urllib.request.urlopen(cancel_req, timeout=5)
            print("Task cancelled.")
        except Exception:
            print(f"Could not cancel task. Cancel manually: curl -X POST {cancel_url}")
        sys.exit(130)


_CLAUDE_ROUTING_KEYWORDS = frozenset(
    {
        "claude",
        "anthropic",
        "messages api",
        "tool use",
        "tool_choice",
        "input_schema",
        "extended thinking",
        "thinking",
        "prompt caching",
        "agent sdk",
        "mcp",
        "system prompt",
    }
)


def _claude_source_filter(question: str, source_names: list[str] | None) -> list[str] | None:
    """Resolve the source filter for ``dec ask``.

    Explicit ``--source`` flags win; otherwise a Claude/Anthropic keyphrase in
    the question auto-routes to the Claude documentation sources.
    """
    if source_names:
        return list(source_names)
    from data_engineering_copilot.services.claude_docs_ingestion import SOURCE_CODE, SOURCE_PLATFORM

    lowered = question.lower()
    if any(keyword in lowered for keyword in _CLAUDE_ROUTING_KEYWORDS):
        return [SOURCE_PLATFORM, SOURCE_CODE]
    return None


def _resolve_embedding_encoder(settings: AppSettings) -> TokenEncoder:
    """Pick the token encoder matching the primary embedding provider.

    The lossless splitter and the embedder's pre-flight budget must count with
    the same encoder. Resolution follows ``EMBEDDING_FALLBACK_ORDER``: the first
    configured provider's model slug drives the encoder (unknown models resolve
    to the shared cl100k fallback without touching the network).
    """
    from data_engineering_copilot.infrastructure.tokenizer_registry import resolve_token_encoder

    order = settings.embedding_fallback_order or ["nvidia"]
    for provider in order:
        model = {
            "nvidia": settings.nvidia_embedding_model,
            "openrouter": settings.openrouter_embedding_model,
            "gemini": settings.gemini_embedding_model,
            "huggingface": settings.huggingface_embedding_model,
        }.get(provider)
        if model:
            return resolve_token_encoder(model)
    return resolve_token_encoder(settings.active_embedding_model_name())


def ask(
    question: str,
    user_id: str | None = None,
    session_id: str | None = None,
    source_names: list[str] | None = None,
) -> None:
    import asyncio

    from data_engineering_copilot.factory import build_rag_service

    source_filter = _claude_source_filter(question, source_names)
    logger.info(
        "CLI ask started question=%r source_filter=%s",
        question[:200],
        source_filter,
    )
    service = build_rag_service()
    answer = asyncio.run(service.answer(question, user_id=user_id, session_id=session_id, source_filter=source_filter))
    logger.info("CLI ask completed confidence=%.4f sources=%s", answer.confidence, len(answer.sources))
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            print(f"- {source.title}: {source.url}")
    print(f"\nConfidence: {answer.confidence:.2f}")


def ingest_claude_docs(site: str = "all", max_docs: int | None = None) -> None:
    """Fetch Claude Platform / Claude Code docs from their ``llms.txt`` indexes and ingest into Qdrant.

    Runs in-process (no Celery/API). Requires Qdrant and an embedder (default
    Ollama). ``--max-docs`` limits the number of markdown files fetched per
    site for a quick smoke run.
    """
    import asyncio

    from data_engineering_copilot.factory import (
        _build_provider_health_registry,
        _build_provider_rate_limiters,
        build_embedding_fallback_chain,
    )
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder
    from data_engineering_copilot.services.claude_docs_ingestion import (
        LLMS_DOC_SITES,
    )
    from data_engineering_copilot.services.claude_docs_ingestion import (
        ingest_claude_docs as run_ingest,
    )
    from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker

    sites = ["platform", "code"]
    if site != "all":
        if site not in LLMS_DOC_SITES:
            raise ValueError(f"Unknown site: {site!r}. Choose from: platform, code, all.")
        sites = [site]

    chunker = HeaderAwareChunker(
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
        min_chunk_words=int(settings.chunk_size_words * 0.1),
    )
    # Route embeddings through the unified fallback chain (per EMBEDDING_FALLBACK_ORDER)
    # with Ollama as degraded fallback, so a transient 5xx on the primary provider
    # fails over instead of aborting the whole ingest.
    provider_rate_limiters = _build_provider_rate_limiters(settings)
    health_registry = _build_provider_health_registry(settings)
    embedding_chain = build_embedding_fallback_chain(
        purpose="global",
        app_settings=settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
    )
    embedder: EmbedderProtocol = FallbackEmbedder(embedding_chain)
    store = AsyncQdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
        hybrid_search=settings.hybrid_search_enabled,
        hybrid_rrf_k=settings.hybrid_rrf_k,
        embedding_dimension=settings.get_embedding_dimension(),
        bm25_namespace=settings.namespace_bm25_enabled,
    )

    # Use the same token encoder as the embedding providers for the lossless
    # split budget, so segments that pass the splitter also pass the embedder's
    # pre-flight budget check.
    split_encoder = _resolve_embedding_encoder(settings)

    async def _run() -> dict[str, object]:
        await store.initialize()
        try:
            return await run_ingest(sites, max_docs, chunker, embedder, store, encoder=split_encoder)
        finally:
            await store.close()

    summary = asyncio.run(_run())
    print(
        f"Ingested {summary['documents']} documents → {summary['chunks']} chunks "
        f"({summary['chunked_documents']} chunked, {summary['fetch_failures']} fetch failures)"
    )
    for source_name, count in summary["per_source"].items():  # type: ignore[union-attr]
        print(f"  {source_name}: {count} documents")


def reenrich(source: str, urls_file: str | None, category: str = "enrichment") -> None:
    """Re-enrich pages whose contextual enrichment previously failed.

    For each URL this clears the vector-store chunks, the Redis
    ``crawl:url_registry:<source>`` entry, and resets the frontier row to
    ``DISCOVERED`` (fresh attempts budget), then re-runs ingestion for the
    source in-process so the pages are re-fetched, re-chunked and
    re-enriched.  URLs come from ``--urls <file>`` or, when omitted, from the
    Redis set ``ingest:enrichment_failed:<source>`` written by the ingestion
    pipeline's enrichment failure recorder.

    Use ``--category`` to filter by failure type:
    - enrichment (default): only enrichment failures from Redis
    - fetch: HTTP errors, timeouts, connection errors
    - embed: embedding failures
    - all: all failure types
    """
    import redis as sync_redis

    from data_engineering_copilot.factory import build_async_ingestion_service
    from data_engineering_copilot.infrastructure.crawl_db import PostgresCrawlFrontierDB
    from data_engineering_copilot.workers.progress import get_redis_client

    if not settings.crawl_db_url:
        raise RuntimeError("CRAWL_DB_URL must be set to re-enrich (the crawler frontier is PostgreSQL-backed).")

    if category == "enrichment":
        # Original behavior: use Redis set or --urls file
        if urls_file:
            with open(urls_file, encoding="utf-8") as fh:
                urls = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
        else:
            r = get_redis_client()
            member_bytes = r.smembers(f"ingest:enrichment_failed:{source}")
            urls = [m.decode() if isinstance(m, bytes) else m for m in member_bytes]
    else:
        # Use frontier queries for other categories
        frontier = PostgresCrawlFrontierDB(settings.crawl_db_url)

        async def _get_urls() -> list[str]:
            await frontier.initialize()
            if category == "all":
                failed = await frontier.get_failed_urls(source, "all")
                skipped = await frontier.get_skipped_urls(source)
                await frontier.close()
                return failed + skipped
            elif category == "fetch":
                await frontier.close()
                return await frontier.get_failed_urls(source, "fetch")
            elif category == "embed":
                await frontier.close()
                return await frontier.get_failed_urls(source, "embed")
            else:
                await frontier.close()
                return []

        urls = asyncio.run(_get_urls())

    if not urls:
        print(f"No URLs to reprocess for source '{source}' (category: {category}).")
        return

    url_set = sorted(set(urls))
    print(f"Reprocessing {len(url_set)} URLs for source '{source}' (category: {category})")

    service = build_async_ingestion_service()

    async def _run() -> int:
        if hasattr(service.vector_store, "initialize"):
            await service.vector_store.initialize()
        deleted = 0
        for url in url_set:
            try:
                await service.vector_store.delete_by_url(url)
                deleted += 1
            except Exception as exc:
                logger.warning("reenrich.delete_chunks_failed url=%s error=%s", url, exc)
        print(f"Deleted existing chunks for {deleted}/{len(url_set)} URLs")

        registry = sync_redis.Redis.from_url(settings.redis_url, socket_timeout=3)
        for url in url_set:
            registry.hdel(f"crawl:url_registry:{source}", url)
        print(f"Cleared URL-registry entries for {len(url_set)} URLs")

        frontier = PostgresCrawlFrontierDB(settings.crawl_db_url)
        await frontier.initialize()
        requeued = await frontier.requeue_urls(url_set)
        await frontier.close()
        print(f"Requeued {requeued}/{len(url_set)} frontier rows to DISCOVERED")

        return await service.ingest(source_names=[source], max_pages_per_source=settings.recovery_max_pages)

    total = asyncio.run(_run())
    print(f"Re-ingestion complete: {total} chunks indexed.")


def retry_failed(source: str, category: str | None) -> None:
    """Retry all failed pages for a source, optionally filtered by category.

    Categories: fetch (HTTP errors), embed (embedding failures),
    upsert (vector store failures), all (everything).
    """
    from data_engineering_copilot.factory import build_async_ingestion_service
    from data_engineering_copilot.infrastructure.crawl_db import PostgresCrawlFrontierDB

    if not settings.crawl_db_url:
        raise RuntimeError("CRAWL_DB_URL must be set to retry failed pages.")

    print(f"Retrieving failed pages for source '{source}'" + (f" (category: {category})" if category else ""))

    frontier = PostgresCrawlFrontierDB(settings.crawl_db_url)

    async def _run() -> int:
        await frontier.initialize()
        urls = await frontier.get_failed_urls(source, category)
        await frontier.close()

        if not urls:
            print(f"No failed pages found for source '{source}'" + (f" in category '{category}'" if category else ""))
            return 0

        url_set = sorted(set(urls))
        print(f"Found {len(url_set)} failed pages to retry")

        service = build_async_ingestion_service()

        # Clear Qdrant chunks
        if hasattr(service.vector_store, "initialize"):
            await service.vector_store.initialize()
        deleted = 0
        for url in url_set:
            try:
                await service.vector_store.delete_by_url(url)
                deleted += 1
            except Exception as exc:
                logger.warning("retry_failed.delete_chunks_failed url=%s error=%s", url, exc)
        print(f"Deleted existing chunks for {deleted}/{len(url_set)} URLs")

        # Clear URL registry
        import redis as sync_redis

        registry = sync_redis.Redis.from_url(settings.redis_url, socket_timeout=3)
        for url in url_set:
            registry.hdel(f"crawl:url_registry:{source}", url)
        print(f"Cleared URL-registry entries for {len(url_set)} URLs")

        # Requeue frontier rows
        frontier2 = PostgresCrawlFrontierDB(settings.crawl_db_url)
        await frontier2.initialize()
        requeued = await frontier2.requeue_urls(url_set)
        await frontier2.close()
        print(f"Requeued {requeued}/{len(url_set)} frontier rows to DISCOVERED")

        # Re-run ingestion
        return await service.ingest(source_names=[source], max_pages_per_source=settings.recovery_max_pages)

    total = asyncio.run(_run())
    print(f"Re-ingestion complete: {total} chunks indexed.")


def unskip(source: str) -> None:
    """Re-process SKIPPED pages for a source.

    SKIPPED pages are those where parsing returned no readable content.
    This command resets them to DISCOVERED so they can be re-fetched and re-processed.
    """
    from data_engineering_copilot.factory import build_async_ingestion_service
    from data_engineering_copilot.infrastructure.crawl_db import PostgresCrawlFrontierDB

    if not settings.crawl_db_url:
        raise RuntimeError("CRAWL_DB_URL must be set to unskip pages.")

    print(f"Retrieving skipped pages for source '{source}'")

    frontier = PostgresCrawlFrontierDB(settings.crawl_db_url)

    async def _run() -> int:
        await frontier.initialize()
        urls = await frontier.get_skipped_urls(source)
        await frontier.close()

        if not urls:
            print(f"No skipped pages found for source '{source}'")
            return 0

        url_set = sorted(set(urls))
        print(f"Found {len(url_set)} skipped pages to re-process")

        service = build_async_ingestion_service()

        # Clear Qdrant chunks (in case some were partially indexed)
        if hasattr(service.vector_store, "initialize"):
            await service.vector_store.initialize()
        deleted = 0
        for url in url_set:
            try:
                await service.vector_store.delete_by_url(url)
                deleted += 1
            except Exception as exc:
                logger.warning("unskip.delete_chunks_failed url=%s error=%s", url, exc)
        print(f"Deleted existing chunks for {deleted}/{len(url_set)} URLs")

        # Clear URL registry
        import redis as sync_redis

        registry = sync_redis.Redis.from_url(settings.redis_url, socket_timeout=3)
        for url in url_set:
            registry.hdel(f"crawl:url_registry:{source}", url)
        print(f"Cleared URL-registry entries for {len(url_set)} URLs")

        # Requeue frontier rows
        frontier2 = PostgresCrawlFrontierDB(settings.crawl_db_url)
        await frontier2.initialize()
        requeued = await frontier2.requeue_urls(url_set)
        await frontier2.close()
        print(f"Requeued {requeued}/{len(url_set)} frontier rows to DISCOVERED")

        # Re-run ingestion
        return await service.ingest(source_names=[source], max_pages_per_source=settings.recovery_max_pages)

    total = asyncio.run(_run())
    print(f"Re-ingestion complete: {total} chunks indexed.")


def _recreate_qdrant_collection() -> None:
    """Delete and recreate the Qdrant collection with the current dimension/hybrid config."""
    url = f"{settings.qdrant_url}/collections/{settings.collection_name}"
    logger.warning("Resetting Qdrant collection=%s url=%s", settings.collection_name, url)
    try:
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            print(f"Deleted collection '{settings.collection_name}': {body}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"Collection '{settings.collection_name}' does not exist (nothing to reset).")
        else:
            raise

    dim = settings.get_embedding_dimension()
    hybrid = settings.hybrid_search_enabled
    create_url = f"{settings.qdrant_url}/collections/{settings.collection_name}"
    if hybrid:
        payload = {
            "vectors": {"dense": {"size": dim, "distance": "Cosine"}},
            "sparse_vectors": {"sparse": {"index": None}},
        }
    else:
        payload = {"vectors": {"size": dim, "distance": "Cosine"}}
    req = urllib.request.Request(
        create_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
        print(f"Created collection '{settings.collection_name}' (dim={dim}, hybrid={hybrid}): {body}")

    logger.info("Qdrant collection reset completed collection=%s", settings.collection_name)


def _bm25_cache_path() -> pathlib.Path:
    """Return the default persisted BM25 tokenizer path for the current collection."""
    from data_engineering_copilot.config.settings import PROJECT_ROOT

    return PROJECT_ROOT / ".bm25_cache" / f"{settings.collection_name}.json"


def validate_spark_source_config() -> int:
    """Validate the pinned Spark source configuration without network access.

    Returns ``0`` on success and ``1`` on validation failure. Never downloads
    or mutates anything.
    """
    from data_engineering_copilot.config.settings import load_spark_source_config

    config_path = settings.spark_sources_path
    try:
        config = load_spark_source_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"❌ Invalid Spark source config: {exc}")
        return 1
    print(f"Spark source: {config.name}")
    print(f"  ref:      {config.ref}")
    print(f"  commit:   {config.commit}")
    print(f"  license:  {config.license}")
    print(f"  streams:  {len(config.streams)}")
    for stream in config.streams:
        print(f"    - {stream.name}: doc_type={stream.doc_type} language={stream.language} chunking={stream.chunking}")
    print("✅ Spark source config valid")
    return 0


def validate_spark_rendered_config() -> int:
    """Validate the pinned Spark rendered config and its commit alignment."""
    from data_engineering_copilot.config.settings import (
        load_spark_rendered_source_config,
        load_spark_source_config,
    )

    try:
        rendered_config = load_spark_rendered_source_config(settings.spark_rendered_sources_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"❌ Invalid Spark rendered config: {exc}")
        return 1
    try:
        native_config = load_spark_source_config(settings.spark_sources_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"❌ Invalid Spark source config: {exc}")
        return 1
    print(f"Spark rendered: {rendered_config.name}")
    print(f"  commit:   {rendered_config.commit}")
    print(f"  builds:   {len(rendered_config.builds)}")
    for build in rendered_config.builds:
        print(f"    - {build.name}: renderer={build.renderer} doc_type={build.doc_type} language={build.language}")
    if native_config.commit != rendered_config.commit:
        print(f"❌ Native commit {native_config.commit!r} != rendered commit {rendered_config.commit!r}")
        return 1
    print("✅ Spark rendered config valid")
    return 0


def _spark_generation_collection(generation: str) -> str:
    return resolve_naming(generation).collection_name


def _spark_commit_short(commit: str) -> str:
    return commit[:8]


def _resolve_spark_embedding_name() -> str:
    model = settings.active_embedding_model_name()
    if settings.embedding_provider == "nvidia":
        model = settings.nvidia_embedding_model
    elif settings.embedding_provider == "openrouter":
        model = settings.openrouter_embedding_model
    return model or "unknown-embedder"


def _default_spark_generation() -> str:
    import hashlib
    from dataclasses import asdict

    config = _load_spark_source_config_or_exit()
    embedding = _resolve_spark_embedding_name()
    identity = json.dumps(
        {"embedding": embedding, "config": asdict(config)},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"spark-{config.ref}-{_spark_commit_short(config.commit)}-{digest}"


def _load_spark_source_config_or_exit():
    from data_engineering_copilot.config.settings import load_spark_source_config

    try:
        return load_spark_source_config(settings.spark_sources_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"❌ Invalid Spark source config: {exc}")
        raise SystemExit(2) from exc


def _load_active_state() -> dict:
    active_path = settings.index_state_dir / "active.json"
    if active_path.exists():
        try:
            return json.loads(active_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def _write_active_state(state: dict) -> None:
    settings.index_state_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_state_dir / "active.json").write_text(json.dumps(state, indent=2))
    history_path = settings.index_state_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(state) + "\n")


def spark_manifest(output: str | None = None) -> int:
    """Materialize the pinned Spark source and write a manifest file."""
    from data_engineering_copilot.infrastructure.spark_source_resolver import SparkSourceResolver

    config = _load_spark_source_config_or_exit()
    resolver = SparkSourceResolver(config, settings.spark_cache_dir)
    try:
        manifest = resolver.resolve()
    except RuntimeError as exc:
        print(f"❌ Failed to materialize Spark source: {exc}")
        return 5

    generation = _default_spark_generation()
    output_path = pathlib.Path(output) if output else settings.spark_corpus_dir / generation / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "source_name": manifest.source_name,
        "ref": manifest.ref,
        "commit": manifest.commit,
        "manifest_hash": manifest.manifest_hash,
        "files": [
            {
                "stream": f.stream,
                "relative_path": f.relative_path,
                "doc_type": f.doc_type,
                "language": f.language,
                "source_url": f.source_url,
            }
            for f in manifest.files
        ],
    }
    output_path.write_text(json.dumps(manifest_data, indent=2))
    print(f"✅ Manifest written: {output_path}")
    print(f"  Files: {len(manifest.files)}  Manifest hash: {manifest.manifest_hash[:12]}")
    return 0


def _load_spark_rendered_source_config_or_exit():
    from data_engineering_copilot.config.settings import load_spark_rendered_source_config

    try:
        return load_spark_rendered_source_config(settings.spark_rendered_sources_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"❌ Invalid Spark rendered config: {exc}")
        raise SystemExit(2) from exc


def spark_render(generation: str | None = None) -> int:
    """Build pinned rendered Spark docs and write a rendered manifest."""
    from data_engineering_copilot.infrastructure.spark_rendered_builder import SparkRenderedBuilder
    from data_engineering_copilot.infrastructure.spark_source_resolver import SparkSourceResolver

    config = _load_spark_source_config_or_exit()
    rendered_config = _load_spark_rendered_source_config_or_exit()
    if config.commit != rendered_config.commit:
        print(
            f"❌ Native commit {config.commit!r} != rendered commit {rendered_config.commit!r}; configs are out of sync"
        )
        return 5

    gen = generation or _default_spark_generation()
    artifact_root = settings.spark_corpus_dir / gen
    artifact_root.mkdir(parents=True, exist_ok=True)

    resolver = SparkSourceResolver(config, settings.spark_cache_dir)
    try:
        source_root = resolver.materialize()
    except RuntimeError as exc:
        print(f"❌ Failed to materialize Spark source: {exc}")
        return 5

    builder = SparkRenderedBuilder(
        config=rendered_config,
        source_root=source_root,
        artifact_root=artifact_root,
        python_executable=_spark_pydocs_python(),
    )
    try:
        manifest = builder.render(log_name="render_build.log")
    except RuntimeError as exc:
        print(f"❌ Spark render failed: {exc}")
        return 5

    manifest_path = artifact_root / "rendered_manifest.json"
    manifest_data = {
        "source_name": manifest.source_name,
        "ref": manifest.ref,
        "commit": manifest.commit,
        "manifest_hash": manifest.manifest_hash,
        "files": [
            {
                "build": f.build,
                "relative_path": f.relative_path,
                "doc_type": f.doc_type,
                "language": f.language,
                "canonical_url": f.canonical_url,
            }
            for f in manifest.files
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
    print(f"✅ Rendered manifest written: {manifest_path}")
    print(f"  Rendered files: {len(manifest.files)}  Manifest hash: {manifest.manifest_hash[:12]}")
    print(f"  Build log: {artifact_root / 'render_build.log'}")
    return 0


def _spark_pydocs_python() -> pathlib.Path | None:
    """Return the PyDocs Sphinx interpreter, or None to let the builder decide."""
    candidate = settings.project_root / "dec_pydocs_venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _build_fallback_embedder() -> EmbedderProtocol:
    """Build the unified embedding fallback chain (NVIDIA -> OpenRouter, ...).

    Routes a 429/network failure on one provider over to the next instead of
    aborting the whole build. Used by both ``spark_build`` and ``gen_build``.
    """
    from data_engineering_copilot.factory import (
        _build_provider_health_registry,
        _build_provider_rate_limiters,
        build_embedding_fallback_chain,
    )
    from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder

    provider_rate_limiters = _build_provider_rate_limiters(settings)
    health_registry = _build_provider_health_registry(settings)
    embedding_chain = build_embedding_fallback_chain(
        purpose="global",
        app_settings=settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
    )
    return FallbackEmbedder(embedding_chain)


def spark_build(generation: str | None = None) -> int:
    """Build a Spark generation collection without activating it."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
    from data_engineering_copilot.infrastructure.spark_rendered_builder import load_rendered_manifest
    from data_engineering_copilot.infrastructure.spark_source_resolver import SparkSourceResolver
    from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
    from data_engineering_copilot.services.spark_chunker import SparkChunker
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder

    config = _load_spark_source_config_or_exit()
    gen = generation or _default_spark_generation()
    collection = _spark_generation_collection(gen)
    artifact_root = settings.spark_corpus_dir / gen

    rendered_config = _load_spark_rendered_source_config_or_exit()
    rendered_manifest = None
    manifest_path = artifact_root / "rendered_manifest.json"
    if manifest_path.is_file():
        try:
            rendered_manifest = load_rendered_manifest(manifest_path, artifact_root, rendered_config)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"❌ Invalid rendered manifest {manifest_path}: {exc}")
            return 5
        print(f"  Rendered manifest: {len(rendered_manifest.files)} files")
    else:
        print(f"  No rendered manifest at {manifest_path}; native-only build")

    store = AsyncQdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=collection,
        hybrid_search=settings.hybrid_search_enabled,
        hybrid_rrf_k=settings.hybrid_rrf_k,
        embedding_dimension=settings.get_embedding_dimension(),
        bm25_namespace=settings.namespace_bm25_enabled,
    )
    resolver = SparkSourceResolver(config, settings.spark_cache_dir)
    parser = NativeDocumentParser()
    header_chunker = HeaderAwareChunker(
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    chunker = SparkChunker(header_chunker=header_chunker)

    embedder = _build_fallback_embedder()
    from data_engineering_copilot.observability.telemetry import build_telemetry_tracer

    builder = SparkIndexBuilder(
        config=config,
        resolver=resolver,
        parser=parser,
        chunker=chunker,
        store=store,
        generation=gen,
        embedder=embedder,
        rendered_config=rendered_config,
        rendered_manifest=rendered_manifest,
        chunks_path=artifact_root / "chunks.jsonl",
        telemetry=build_telemetry_tracer(),
    )
    try:
        report = asyncio.run(builder.build())
    except Exception as exc:
        print(f"❌ Spark build failed: {exc}")
        return 5
    print(f"✅ Spark build complete: generation={report.generation}")
    print(f"  Chunks: {report.chunk_count}  Files: {report.source_file_count}")
    print(f"  BM25 vocab: {report.bm25_vocabulary_size}  Validation: {report.validation_passed}")
    print(f"  Collection: {report.qdrant_collection} (not activated)")
    return 0


def _validation_report_path(generation: str) -> pathlib.Path:
    return settings.index_state_dir / f"validation-{generation}.json"


def spark_validate(generation: str) -> int:
    """Validate a built Spark generation collection without mutation.

    Runs the strict artifact checks (coverage records, manifest path
    uniqueness, chunk IDs, per-chunk generation/commit metadata, point count
    vs ``chunks.jsonl``) plus the store-level checks (dense/sparse config,
    BM25 state, metadata presence). Writes a validation report that
    ``spark-activate`` requires.
    """
    import asyncio

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    if not generation or not re_fullmatch_identifier(generation):
        print("❌ Invalid generation identifier")
        return 2

    config = _load_spark_source_config_or_exit()
    collection = _spark_generation_collection(generation)
    artifact_root = settings.spark_corpus_dir / generation

    chunks, coverage, native_paths, rendered_paths = _load_generation_artifacts(generation, artifact_root)
    if chunks is None:
        return 3

    store = AsyncQdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=collection,
        hybrid_search=settings.hybrid_search_enabled,
        embedding_dimension=settings.get_embedding_dimension(),
        bm25_namespace=settings.namespace_bm25_enabled,
    )

    async def _validate_store() -> dict[str, object]:
        await store.initialize()
        report = await store.validate_index_generation(expected_points=len(chunks))
        report["metadata_complete"] = await _collection_has_metadata(store)
        report["payload_text_mismatches"] = await store.verify_payload_texts(
            {chunk.chunk_id: chunk.text for chunk in chunks}
        )
        return report

    try:
        store_report = asyncio.run(_validate_store())
    except Exception as exc:
        print(f"❌ Spark validation failed: {exc}")
        return 5
    if store_report.get("error"):
        print(f"❌ Validation failed: {store_report['error']}")
        return 3

    point_count_value = store_report.get("point_count")
    failures = validate_generation_artifacts(
        generation=generation,
        expected_commit=config.commit,
        chunks=chunks,
        coverage=coverage,
        native_manifest_paths=native_paths,
        rendered_manifest_paths=rendered_paths,
        qdrant_point_count=int(point_count_value) if isinstance(point_count_value, int) else None,
        bm25_ready=bool(store_report.get("bm25_ready")),
        sparse_configured=bool(store_report.get("sparse_configured")),
    )
    if not store_report.get("metadata_complete", False):
        failures.append("collection lacks doc_type metadata")
    payload_mismatches = store_report.get("payload_text_mismatches", [])
    if isinstance(payload_mismatches, list):
        failures.extend(str(item) for item in payload_mismatches)

    if failures:
        print("❌ Generation validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 3

    print(f"✅ Generation {generation} validated")
    print(f"  Collection: {store_report.get('collection')}  Points: {store_report.get('point_count')}")
    print(f"  Sparse: {store_report.get('sparse_configured')}  BM25 ready: {store_report.get('bm25_ready')}")
    print(f"  Chunks (chunks.jsonl): {len(chunks)}  Coverage records: {len(coverage)}")

    # Write validation report required by spark-activate.
    try:
        settings.index_state_dir.mkdir(parents=True, exist_ok=True)
        _validation_report_path(generation).write_text(
            json.dumps({"generation": generation, "collection": collection, "passed": True})
        )
    except OSError as exc:
        print(f"⚠️  Could not write validation report: {exc}")
    return 0


def _load_generation_artifacts(
    generation: str,
    artifact_root: pathlib.Path,
) -> tuple[list[DocumentChunk] | None, list[CoverageRecord], list[str], list[str] | None]:
    """Load chunks.jsonl + coverage.json + manifest paths for a generation.

    Returns ``(None, [], [], None)`` when ``chunks.jsonl`` is missing.
    """
    from data_engineering_copilot.infrastructure.spark_rendered_builder import load_rendered_manifest

    chunks_path = artifact_root / "chunks.jsonl"
    if not chunks_path.is_file():
        print(f"❌ Missing chunks.jsonl at {chunks_path}")
        return None, [], [], None

    chunks: list[DocumentChunk] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunks.append(DocumentChunk(**data))

    coverage_path = artifact_root / "coverage.json"
    coverage: list[CoverageRecord] = []
    if coverage_path.is_file():
        coverage = [CoverageRecord(**entry) for entry in json.loads(coverage_path.read_text(encoding="utf-8"))]

    native_paths: list[str] = []
    native_manifest_path = artifact_root / "native_manifest.json"
    if not native_manifest_path.is_file():
        native_manifest_path = artifact_root / "manifest.json"
    if native_manifest_path.is_file():
        native_paths = [
            entry["relative_path"] for entry in json.loads(native_manifest_path.read_text(encoding="utf-8"))["files"]
        ]

    rendered_paths: list[str] | None = None
    rendered_manifest_path = artifact_root / "rendered_manifest.json"
    if rendered_manifest_path.is_file():
        rendered_config = _load_spark_rendered_source_config_or_exit()
        rendered_manifest = load_rendered_manifest(rendered_manifest_path, artifact_root, rendered_config)
        rendered_paths = [record.relative_path for record in rendered_manifest.files]

    return chunks, coverage, native_paths, rendered_paths


async def _collection_has_metadata(store) -> bool:
    """Check that at least one point in the collection carries doc_type."""
    try:
        points, _ = await store._client.scroll(
            collection_name=store._collection_name,
            limit=1,
            with_payload=["doc_type"],
            with_vectors=False,
        )
        return bool(points and points[0].payload and points[0].payload.get("doc_type"))
    except Exception:
        return False


def re_fullmatch_identifier(value: str) -> bool:
    """Return True when ``value`` matches the safe generation identifier regex."""
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9_.:-]+", value))


def _confirm_required(action: str) -> bool:
    import os

    if os.environ.get("FORCE") == "1":
        return True
    try:
        answer = input(f"{action} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _qdrant_collection_aliases(name: str) -> list[str]:
    """Return the alias names that resolve to *name* (empty when not an alias)."""
    req = urllib.request.Request(f"{settings.qdrant_url}/collections/{name}/aliases", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
    result = body.get("result", {}) if isinstance(body, dict) else {}
    aliases = result.get("aliases", []) if isinstance(result, dict) else []
    found: list[str] = []
    for entry in aliases:
        if isinstance(entry, dict):
            value: object = entry.get("alias_name")
            if isinstance(value, str):
                found.append(value)
    return found


def _qdrant_change_alias(generation: str) -> None:
    """Atomically repoint the logical alias to a generation collection.

    Handles both prior states: the alias already existing (delete + recreate in
    one batch) or a plain collection shadowing the alias name (delete that
    collection first, then create the alias) — e.g. right after ``gen-reset``
    recreated the base collection. Qdrant refuses to create an alias whose name
    is already a collection.
    """
    collection = _spark_generation_collection(generation)
    alias = settings.active_collection_alias
    if alias in _qdrant_collection_aliases(alias):
        changes: list[dict[str, object]] = [{"delete_alias": {"alias_name": alias}}]
    else:
        if alias in _list_qdrant_collections():
            _qdrant_delete_collection(alias)
        changes = []
    changes.append({"create_alias": {"alias_name": alias, "collection_name": collection}})
    payload = {"actions": changes}
    req = urllib.request.Request(
        f"{settings.qdrant_url}/collections/aliases",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
        if body.get("status") != "ok":
            raise RuntimeError(f"Qdrant alias change failed: {body}")


def spark_activate(generation: str) -> int:
    """Activate a validated generation by repointing the logical alias.

    Refuses to activate unless ``spark-validate`` has written a passing
    validation report for the generation. Requires interactive confirmation
    (or ``FORCE=1`` in non-interactive shells).
    """
    report_path = _validation_report_path(generation)
    if not report_path.exists():
        print(f"❌ No validation report for generation {generation}; run `dec spark-validate` first")
        return 3
    try:
        report = json.loads(report_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        print(f"❌ Validation report for {generation} is corrupt; re-run `dec spark-validate`")
        return 3
    if not report.get("passed"):
        print(f"❌ Validation report for {generation} did not pass; re-run `dec spark-validate`")
        return 3

    if not _confirm_required(f"Activate generation {generation}? This changes the live index"):
        print("Aborted.")
        return 0
    try:
        _qdrant_change_alias(generation)
    except Exception as exc:
        print(f"❌ Activation failed: {exc}")
        return 5
    _write_active_state({"generation": generation, "collection": _spark_generation_collection(generation)})
    print(f"✅ Activated generation {generation} -> {settings.active_collection_alias}")
    return 0


def spark_rollback(generation: str) -> int:
    """Roll back the logical alias to a previously recorded generation."""
    state = _load_active_state()
    if state.get("generation") != generation:
        print(f"❌ Generation {generation} is not the active generation")
        return 4
    history = []
    history_path = settings.index_state_dir / "history.jsonl"
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                continue
    previous = None
    for entry in history:
        if entry.get("generation") == generation:
            break
        previous = entry
    if previous is None:
        print("❌ No previous generation recorded for rollback")
        return 4
    if not _confirm_required(f"Roll back to generation {previous['generation']}?"):
        print("Aborted.")
        return 0
    try:
        _qdrant_change_alias(previous["generation"])
    except Exception as exc:
        print(f"❌ Rollback failed: {exc}")
        return 5
    _write_active_state(previous)
    print(f"✅ Rolled back to generation {previous['generation']}")
    return 0


# ------------------------------------------------------------------
# Pinned generation (gen-*) pipeline
# ------------------------------------------------------------------


def _load_pinned_sources_or_exit() -> tuple:
    from data_engineering_copilot.config.settings import load_pinned_sources

    try:
        return load_pinned_sources(settings.pinned_sources_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"❌ Invalid pinned sources config: {exc}")
        raise SystemExit(2) from exc


def gen_config_check() -> int:
    """Validate the pinned sources configuration without network access."""
    from data_engineering_copilot.config.settings import load_pinned_sources

    try:
        sources = load_pinned_sources(settings.pinned_sources_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"❌ Invalid pinned sources config: {exc}")
        return 1
    print(f"Pinned sources: {len(sources)}")
    for source in sources:
        print(f"  {source.slug} ({source.type}): name={source.name} version={source.version}")
        if source.type == "github":
            print(f"    repository: {source.repository}")
            print(f"    commit:     {source.commit}")
            for stream in source.streams:
                print(
                    f"    - {stream.name}: doc_type={stream.doc_type} "
                    f"language={stream.language} chunking={stream.chunking}"
                )
        else:
            print(f"    url_prefix: {source.url_prefix}")
            print(f"    doc_type:   {source.doc_type}")
            if source.type == "local_mirror":
                print(f"    mirror_dir: {source.mirror_dir}")
                print(f"    commit:     {source.commit}")
                print(f"    license:    {source.license}")
            else:
                print(f"    index_url:  {source.index_url}")
    print("✅ Pinned sources config valid")
    return 0


def _default_generation() -> str:
    """Derive the combined pinned generation ID from config + embedder."""
    import hashlib
    from dataclasses import asdict

    sources = _load_pinned_sources_or_exit()
    embedding = _resolve_spark_embedding_name()
    identity = json.dumps(
        {"embedding": embedding, "sources": [asdict(source) for source in sources]},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"pinned-{digest}"


def _resolve_pinned_sources() -> list[dict[str, object]]:
    """Materialize every pinned source and return per-source manifest dicts."""
    from data_engineering_copilot.infrastructure.spark_source_resolver import SparkSourceResolver
    from data_engineering_copilot.services.url_index_resolver import LocalMirrorResolver, UrlIndexResolver

    results: list[dict[str, object]] = []
    for config in _load_pinned_sources_or_exit():
        if config.type == "github":
            manifest = SparkSourceResolver(config, settings.pinned_cache_dir).resolve()
            results.append(
                {
                    "slug": config.slug,
                    "type": config.type,
                    "name": config.name,
                    "commit": config.commit,
                    "files": [
                        {
                            "stream": record.stream,
                            "relative_path": record.relative_path,
                            "doc_type": record.doc_type,
                            "language": record.language,
                            "source_url": record.source_url,
                        }
                        for record in manifest.files
                    ],
                }
            )
        elif config.type == "local_mirror":
            manifest = LocalMirrorResolver(config, settings.claude_docs_mirror_dir).resolve()
            results.append(
                {
                    "slug": config.slug,
                    "type": config.type,
                    "name": config.name,
                    "commit": config.commit,
                    "files": [
                        {"relative_path": entry.relative_path, "title": entry.title, "url": entry.url}
                        for entry in manifest.entries
                    ],
                }
            )
        else:
            manifest = asyncio.run(UrlIndexResolver(config, settings.pinned_cache_dir).resolve())
            results.append(
                {
                    "slug": config.slug,
                    "type": config.type,
                    "name": config.name,
                    "commit": "",
                    "files": [
                        {"relative_path": entry.relative_path, "title": entry.title, "url": entry.url}
                        for entry in manifest.entries
                    ],
                }
            )
    return results


def gen_manifest(generation: str | None = None) -> int:
    """Materialize all pinned sources and write per-source + combined manifests."""
    gen = generation or _default_generation()
    naming = resolve_naming(gen)
    validate_naming(naming)
    artifact_root = settings.pinned_corpus_dir / naming.artifact_dir_name
    artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        results = _resolve_pinned_sources()
    except RuntimeError as exc:
        print(f"❌ Failed to materialize pinned sources: {exc}")
        return 5
    combined_files: list[dict[str, str]] = []
    total = 0
    for result in results:
        slug = str(result["slug"])
        files = cast(list[dict[str, str]], result["files"])
        (artifact_root / f"manifest-{slug}.json").write_text(json.dumps(result, indent=2))
        combined_files.extend({"relative_path": entry["relative_path"]} for entry in files)
        total += len(files)
        print(f"  {slug}: {len(files)} files")
    (artifact_root / "manifest.json").write_text(json.dumps({"files": combined_files}, indent=2))
    print(f"✅ Pinned manifest written: {artifact_root}")
    print(f"  Total files: {total}")
    return 0


def gen_build(generation: str | None = None) -> int:
    """Build a combined pinned generation collection without activating it."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.github_source_preparer import GithubSourcePreparer
    from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
    from data_engineering_copilot.services.pinned_index_builder import PinnedIndexBuilder
    from data_engineering_copilot.services.url_index_preparer import UrlIndexPreparer

    gen = generation or _default_generation()
    naming = resolve_naming(gen)
    validate_naming(naming)
    collection = naming.collection_name
    artifact_root = settings.pinned_corpus_dir / naming.artifact_dir_name
    artifact_root.mkdir(parents=True, exist_ok=True)

    store = AsyncQdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=collection,
        hybrid_search=settings.hybrid_search_enabled,
        hybrid_rrf_k=settings.hybrid_rrf_k,
        embedding_dimension=settings.get_embedding_dimension(),
        bm25_namespace=settings.namespace_bm25_enabled,
    )
    embedder = _build_fallback_embedder()
    header_chunker = HeaderAwareChunker(
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )

    packages = []
    for config in _load_pinned_sources_or_exit():
        if config.type == "github":
            package = asyncio.run(
                GithubSourcePreparer(config, settings.pinned_cache_dir, gen, header_chunker=header_chunker).prepare()
            )
        else:
            package = asyncio.run(
                UrlIndexPreparer(
                    config,
                    settings.pinned_cache_dir,
                    gen,
                    mirror_root=settings.claude_docs_mirror_dir,
                ).prepare()
            )
        packages.append(package)
        print(f"  {config.slug}: {len(package.chunks)} chunks, {len(package.coverage)} files")

    builder_kwargs: dict[str, object] = {"output_dir": artifact_root}
    if getattr(settings, "late_chunking_enabled", False):
        builder_kwargs.update(
            {
                "late_chunking_enabled": True,
                "late_chunking_max_tokens": settings.late_chunking_max_tokens,
                "late_chunking_model_name": settings.local_hf_embedding_model,
            }
        )
        print("  late chunking: ENABLED (dark-flag benchmark build)")
    builder = PinnedIndexBuilder(store, embedder, gen, **builder_kwargs)  # type: ignore[arg-type]
    try:
        report = asyncio.run(builder.build(packages))
    except Exception as exc:
        print(f"❌ Pinned build failed: {exc}")
        return 5
    print(f"✅ Pinned build complete: generation={report.generation}")
    print(f"  Chunks: {report.chunk_count}  Files: {report.source_file_count}")
    print(f"  BM25 vocab: {report.bm25_vocabulary_size}  Validation: {report.validation_passed}")
    print(f"  Collection: {report.qdrant_collection} (not activated)")
    return 0


def gen_validate(generation: str) -> int:
    """Validate a built pinned generation collection without mutation."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.pinned_index_builder import validate_pinned_generation_artifacts

    if not generation or not re_fullmatch_identifier(generation):
        print("❌ Invalid generation identifier")
        return 2

    naming = resolve_naming(generation)
    validate_naming(naming)
    collection = naming.collection_name
    artifact_root = settings.pinned_corpus_dir / naming.artifact_dir_name

    chunks, coverage, _native_paths, _rendered_paths = _load_generation_artifacts(generation, artifact_root)
    if chunks is None:
        return 3

    store = AsyncQdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=collection,
        hybrid_search=settings.hybrid_search_enabled,
        embedding_dimension=settings.get_embedding_dimension(),
        bm25_namespace=settings.namespace_bm25_enabled,
    )

    async def _validate_store() -> dict[str, object]:
        await store.initialize()
        report = await store.validate_index_generation(expected_points=len(chunks))
        report["metadata_complete"] = await _collection_has_metadata(store)
        report["payload_text_mismatches"] = await store.verify_payload_texts(
            {chunk.chunk_id: chunk.text for chunk in chunks}
        )
        return report

    try:
        store_report = asyncio.run(_validate_store())
    except Exception as exc:
        print(f"❌ Pinned validation failed: {exc}")
        return 5
    if store_report.get("error"):
        print(f"❌ Validation failed: {store_report['error']}")
        return 3

    expected_commits = {config.commit for config in _load_pinned_sources_or_exit() if config.commit}
    expected_commits.add("")
    point_count_value = store_report.get("point_count")
    failures = validate_pinned_generation_artifacts(
        generation=generation,
        expected_commits=expected_commits,
        chunks=chunks,
        coverage=coverage,
        qdrant_point_count=int(point_count_value) if isinstance(point_count_value, int) else None,
        bm25_ready=bool(store_report.get("bm25_ready")),
        sparse_configured=bool(store_report.get("sparse_configured")),
    )
    if not store_report.get("metadata_complete", False):
        failures.append("collection lacks doc_type metadata")
    payload_mismatches = store_report.get("payload_text_mismatches", [])
    if isinstance(payload_mismatches, list):
        failures.extend(str(item) for item in payload_mismatches)

    if failures:
        print("❌ Generation validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 3

    print(f"✅ Generation {generation} validated")
    print(f"  Collection: {store_report.get('collection')}  Points: {store_report.get('point_count')}")
    print(f"  Sparse: {store_report.get('sparse_configured')}  BM25 ready: {store_report.get('bm25_ready')}")
    print(f"  Chunks (chunks.jsonl): {len(chunks)}  Coverage records: {len(coverage)}")

    try:
        settings.index_state_dir.mkdir(parents=True, exist_ok=True)
        _validation_report_path(generation).write_text(
            json.dumps({"generation": generation, "collection": collection, "passed": True})
        )
    except OSError as exc:
        print(f"⚠️  Could not write validation report: {exc}")
    return 0


def gen_activate(generation: str) -> int:
    """Activate a validated pinned generation by repointing the logical alias.

    Shares the validation-report gate, alias change, and active-state write
    with ``spark_activate``.
    """
    return spark_activate(generation)


def gen_rollback(generation: str) -> int:
    """Roll the logical alias back to a previously recorded generation."""
    return spark_rollback(generation)


def _list_qdrant_collections() -> list[str]:
    """Return the collection names currently present in Qdrant."""
    req = urllib.request.Request(f"{settings.qdrant_url}/collections", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
    result = body.get("result", {}) if isinstance(body, dict) else {}
    collections = result.get("collections", []) if isinstance(result, dict) else []
    names = [entry.get("name") for entry in collections if isinstance(entry, dict) and entry.get("name")]
    return [name for name in names if isinstance(name, str)]


def _qdrant_delete_collection(name: str) -> None:
    req = urllib.request.Request(f"{settings.qdrant_url}/collections/{name}", method="DELETE")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
        if isinstance(body, dict) and body.get("status") != "ok":
            raise RuntimeError(f"Qdrant delete failed for {name}: {body}")


def _qdrant_drop_alias() -> None:
    """Best-effort removal of the logical alias (required before deleting its target)."""
    payload = {"actions": [{"delete_alias": {"alias_name": settings.active_collection_alias}}]}
    req = urllib.request.Request(
        f"{settings.qdrant_url}/collections/aliases",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
        if isinstance(body, dict) and body.get("status") != "ok":
            raise RuntimeError(f"Qdrant alias drop failed: {body}")


def _purge_generation_state() -> None:
    """Delete active.json, history.jsonl, and validation reports from index state."""
    state_dir = settings.index_state_dir
    for name in ("active.json", "history.jsonl"):
        path = state_dir / name
        if path.exists():
            path.unlink()
            print(f"  Deleted {path}")
    for report in sorted(state_dir.glob("validation-*.json")):
        report.unlink()
        print(f"  Deleted {report}")


def _purge_generation_bm25_caches() -> None:
    """Delete persisted BM25 tokenizers for generation collections."""
    from data_engineering_copilot.config.settings import PROJECT_ROOT

    cache_dir = PROJECT_ROOT / ".bm25_cache"
    if not cache_dir.is_dir():
        print("  No .bm25_cache dir")
        return
    for path in sorted(cache_dir.glob("data_engineering_docs*.json")):
        path.unlink()
        print(f"  Deleted {path}")


def gen_reset() -> int:
    """Wipe all generation state: alias, gen collections, index state, BM25 caches.

    Deletes every ``data_engineering_docs__*`` generation collection plus the
    active alias, clears ``.index_state`` (active.json / history.jsonl /
    validation reports) and the persisted BM25 tokenizers for generation
    collections, then runs the full ``reset_index()`` crawl-state purge. Disk
    source caches (``data/spark_src``, ``data/raw_sources``,
    ``data/pinned_src``) are preserved.
    """
    if not _confirm_required("Reset all generation collections? This deletes every gen index"):
        print("Aborted.")
        return 0
    try:
        _qdrant_drop_alias()
        print(f"  Dropped alias {settings.active_collection_alias}")
    except urllib.error.HTTPError as exc:
        print(f"  No alias to drop: {exc.code}" if exc.code == 404 else f"  Could not drop alias: {exc}")
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"  Could not drop alias: {exc}")
    try:
        collection_names = _list_qdrant_collections()
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"❌ Could not list Qdrant collections: {exc}")
        return 5
    for name in collection_names:
        if not name.startswith("data_engineering_docs__"):
            continue
        try:
            _qdrant_delete_collection(name)
            print(f"  Deleted collection {name}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                print(f"❌ Could not delete collection {name}: {exc}")
                return 5
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"❌ Could not delete collection {name}: {exc}")
            return 5
    _purge_generation_state()
    _purge_generation_bm25_caches()
    reset_index()
    print("✅ Generation reset complete (disk source caches preserved)")
    return 0


def gen_stale() -> int:
    """Report generation collections: active, stale, or orphan."""
    from data_engineering_copilot.services.pin_maintenance import (
        classify_generations,
        local_generation_collections,
    )

    try:
        names = _list_qdrant_collections()
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"❌ Could not list Qdrant collections: {exc}")
        return 5
    local = local_generation_collections([settings.spark_corpus_dir, settings.pinned_corpus_dir])
    active = _load_active_state().get("generation")
    statuses = classify_generations(names, active, local)
    if not statuses:
        print("No generation collections found")
        return 0
    markers = {"active": "✅ active", "stale": "🟡 stale", "orphan": "⚪ orphan"}
    stale_count = 0
    for status in statuses:
        print(f"  {markers[status.state]}  {status.name}")
        if status.state == "stale":
            stale_count += 1
    if stale_count:
        print(
            f"\n{stale_count} stale generation(s). Rebuild with `dec gen-build`, then `dec gen-validate` + `dec gen-activate`."
        )
    return 0


def _get_bm25_status() -> dict[str, object]:
    """Report BM25/hybrid state for the current collection.

    Reads the persisted BM25 cache file metadata (without loading the vector
    collection) and queries the Qdrant collection configuration for sparse
    vector support. Never mutates Qdrant/Redis/PostgreSQL. Network failures
    are reported as a JSON-safe status dictionary, not raised.
    """
    status_info: dict[str, object] = {
        "cache_exists": False,
        "cache_fitted": False,
        "sparse_configured": False,
        "hybrid_active": False,
    }
    try:
        bm25_path = _bm25_cache_path()
        status_info["cache_path"] = str(bm25_path)
        status_info["cache_exists"] = bm25_path.exists()
        if status_info["cache_exists"]:
            try:
                with bm25_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                status_info["cache_fitted"] = bool(payload.get("frozen", False) and payload.get("corpus_size", 0) > 0)
            except (OSError, ValueError, json.JSONDecodeError):
                status_info["cache_fitted"] = False

        req = urllib.request.Request(f"{settings.qdrant_url}/collections/{settings.collection_name}", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        result = data.get("result", {}) if isinstance(data, dict) else {}
        params = result.get("config", {}).get("params", {}) if isinstance(result, dict) else {}
        sparse_vectors = params.get("sparse_vectors", None) if isinstance(params, dict) else None
        status_info["sparse_configured"] = bool(sparse_vectors)
        status_info["hybrid_active"] = bool(status_info["cache_fitted"] and status_info["sparse_configured"])
    except urllib.error.HTTPError as exc:
        status_info["error"] = f"Qdrant HTTP {exc.code}"
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
        status_info["error"] = str(exc)
    return status_info


def _delete_bm25_cache() -> None:
    """Best-effort removal of the persisted BM25 tokenizer for the current collection."""
    path = _bm25_cache_path()
    if not path.exists():
        print(f"No BM25 cache to delete: {path}")
        return
    try:
        path.unlink()
        print(f"Deleted BM25 cache: {path}")
    except OSError as exc:
        print(f"Warning: could not delete BM25 cache {path}: {exc}")


def reset_qdrant() -> None:
    """Delete and recreate the Qdrant collection and its persisted BM25 cache."""
    _recreate_qdrant_collection()
    _delete_bm25_cache()


def reset_index() -> None:
    """Full clean rebuild: recreate Qdrant + BM25 cache, clear Redis, drop PG frontier.

    Wipes the crawl state (Redis ``crawl:*`` keys + PostgreSQL frontier tables)
    and the vector index (Qdrant collection + persisted BM25 tokenizer) so the
    next ingest rebuilds both sides consistently.  Qdrant is recreated first so
    a failure aborts before the frontier history is dropped.
    """
    _recreate_qdrant_collection()
    _delete_bm25_cache()

    # Clear crawl-related and query-cache keys from Redis
    from data_engineering_copilot.workers.progress import get_redis_client

    try:
        redis_client = get_redis_client()
        registry_keys = list(redis_client.scan_iter("crawl:url_registry:*"))
        if registry_keys:
            redis_client.delete(*registry_keys)
            logger.info("Cleared %d crawl registry keys", len(registry_keys))
        all_crawl_keys = list(redis_client.scan_iter("crawl:*"))
        non_registry = [
            k
            for k in all_crawl_keys
            if not (isinstance(k, str) and k.startswith("crawl:url_registry:"))
            and not (isinstance(k, bytes) and k.startswith(b"crawl:url_registry:"))
        ]
        if non_registry:
            redis_client.delete(*non_registry)
            logger.info("Cleared %d crawl cache keys", len(non_registry))
        rag_cache_keys = list(redis_client.scan_iter("rag:cache:*"))
        if rag_cache_keys:
            redis_client.delete(*rag_cache_keys)
            logger.info("Cleared %d rag query cache keys", len(rag_cache_keys))
    except Exception:
        logger.debug("Could not clear crawl Redis keys (Redis may be unavailable)")

    # Reset the crawl frontier database
    db_url = settings.crawl_db_url
    if db_url:
        from data_engineering_copilot.infrastructure.crawl_db import PostgresCrawlFrontierDB

        async def _reset_pg():
            f = PostgresCrawlFrontierDB(db_url)
            await f.initialize()
            await f.drop_all()
            await f.close()
            logger.info("Reset PostgreSQL crawl frontier database via %s", db_url)

        try:
            asyncio.run(_reset_pg())
        except RuntimeError:
            logger.warning("Skipping PostgreSQL reset — already running in an event loop")


def reset_crawler_db() -> None:
    """Clear crawler state without touching Qdrant.

    Resets Redis ``crawl:*`` keys (URL registry + HTTP cache) and PostgreSQL
    frontier tables (``crawl_frontier`` + ``sitemap_edges``).  Qdrant is
    preserved so the dedup mechanism (``content_hash`` in Qdrant payloads)
    still works — re-crawled pages with unchanged content are skipped.
    """
    print("Resetting crawler database (Qdrant preserved)...\n")

    # Step 1: Clear Redis crawl-related keys
    from data_engineering_copilot.workers.progress import get_redis_client

    redis_cleared = 0
    try:
        redis_client = get_redis_client()

        # Clear URL registry keys
        registry_keys = list(redis_client.scan_iter("crawl:url_registry:*"))
        if registry_keys:
            redis_client.delete(*registry_keys)
            redis_cleared += len(registry_keys)
            print(f"  Cleared {len(registry_keys)} URL registry keys")

        # Clear HTTP cache keys (crawl:<hash>)
        all_crawl_keys = list(redis_client.scan_iter("crawl:*"))
        cache_keys = [
            k
            for k in all_crawl_keys
            if not (isinstance(k, str) and k.startswith("crawl:url_registry:"))
            and not (isinstance(k, bytes) and k.startswith(b"crawl:url_registry:"))
        ]
        if cache_keys:
            redis_client.delete(*cache_keys)
            redis_cleared += len(cache_keys)
            print(f"  Cleared {len(cache_keys)} HTTP cache keys")

        # Clear enrichment failure sets (stale after reset)
        enrichment_keys = list(redis_client.scan_iter("ingest:enrichment_failed:*"))
        if enrichment_keys:
            redis_client.delete(*enrichment_keys)
            redis_cleared += len(enrichment_keys)
            print(f"  Cleared {len(enrichment_keys)} enrichment failure sets")

    except Exception as exc:
        print(f"  Warning: Could not clear Redis keys: {exc}")

    # Step 2: Drop and recreate PostgreSQL frontier tables
    db_url = settings.crawl_db_url
    pg_cleared = False
    if db_url:
        from data_engineering_copilot.infrastructure.crawl_db import PostgresCrawlFrontierDB

        async def _reset_pg():
            f = PostgresCrawlFrontierDB(db_url)
            await f.initialize()
            await f.drop_all()
            await f.close()

        try:
            asyncio.run(_reset_pg())
            pg_cleared = True
            print("  Dropped PostgreSQL tables (crawl_frontier, sitemap_edges)")
            print("  Tables will be recreated on next ingestion")
        except RuntimeError:
            print("  Warning: Could not reset PostgreSQL — already running in an event loop")
        except Exception as exc:
            print(f"  Warning: PostgreSQL reset failed: {exc}")
    else:
        print("  Skipped PostgreSQL reset (CRAWL_DB_URL not set)")

    # Summary
    print("\nCrawler DB reset complete.")
    print(f"  Redis: {redis_cleared} keys cleared")
    print(f"  PostgreSQL: {'reset' if pg_cleared else 'skipped'}")
    print("  Qdrant: preserved (dedup still works)")
    print("\nNext step: run 'dec ingest --source <name>' to re-crawl.")


def _clear_redis_keys(pattern: str) -> int:
    """Best-effort deletion of all Redis keys matching *pattern*. Returns count."""
    from data_engineering_copilot.workers.progress import get_redis_client

    try:
        redis_client = get_redis_client()
        keys = list(redis_client.scan_iter(pattern))
        if keys:
            redis_client.delete(*keys)
            return len(keys)
    except Exception as exc:
        print(f"  Warning: Could not clear Redis keys {pattern}: {exc}")
    return 0


def clear_query_cache() -> None:
    """Clear the RAG query cache (Redis ``rag:cache:*`` keys).

    Removes both tiers of the two-tier query cache — exact-match
    (``rag:cache:exact:*``) and semantic (``rag:cache:semantic:*``) — plus the
    semantic counter.  Qdrant, the BM25 cache, and crawler state are untouched,
    so the vector index keeps serving hits while stale answers are dropped.

    Note: each running API / Streamlit process keeps its own in-memory L1 copy
    of recently cached answers; restart those services for a full cold cache.
    """
    print("Clearing RAG query cache...\n")

    cleared = _clear_redis_keys("rag:cache:*")
    if cleared:
        print(f"  Cleared {cleared} query cache keys (exact + semantic tiers)")
    else:
        print("  No query cache keys found (cache already empty)")

    print("\nQuery cache clear complete.")
    print(f"  Redis: {cleared} keys cleared")
    print("  Qdrant / BM25 / crawler state: untouched")
    print(
        "  Note: running API or Streamlit processes still hold an in-memory copy; restart them for a fully cold cache."
    )


def clear_cache(
    *,
    query: bool = False,
    embedding: bool = False,
    crawl: bool = False,
    bm25: bool = False,
    all_types: bool = False,
) -> None:
    """Clear selected cache stores. With no ``--type`` flags (or ``--all``),
    every cache store is cleared.

    Query cache (``rag:cache:*``), embedding cache (``embed:cache:*``), and
    crawl cache (``crawl:*`` + ``ingest:enrichment_failed:*``) live in Redis;
    the BM25 cache is a persisted tokenizer under ``.bm25_cache/`` on disk.
    Qdrant and PostgreSQL are never touched. Running API / Streamlit processes
    hold in-memory L1 copies — restart them for a fully cold cache.
    """
    if not (query or embedding or crawl or bm25):
        all_types = True

    print("Clearing cache...\n")
    total = 0

    if all_types or query:
        cleared = _clear_redis_keys("rag:cache:*")
        print(f"  Query cache: {cleared} key(s) cleared")
        total += cleared
    if all_types or embedding:
        cleared = _clear_redis_keys("embed:cache:*")
        print(f"  Embedding cache: {cleared} key(s) cleared")
        total += cleared
    if all_types or crawl:
        cleared = _clear_redis_keys("crawl:*")
        cleared += _clear_redis_keys("ingest:enrichment_failed:*")
        print(f"  Crawl cache: {cleared} key(s) cleared")
        total += cleared
    if all_types or bm25:
        _purge_bm25_cache_dir()

    print("\nCache clear complete.")
    print(f"  Redis: {total} keys cleared")
    print(
        "  Note: running API or Streamlit processes still hold an in-memory copy; restart them for a fully cold cache."
    )


def _purge_bm25_cache_dir() -> None:
    """Best-effort removal of every persisted BM25 tokenizer under ``.bm25_cache``."""
    from data_engineering_copilot.config.settings import PROJECT_ROOT

    cache_dir = PROJECT_ROOT / ".bm25_cache"
    if not cache_dir.is_dir():
        print("  No .bm25_cache dir")
        return
    removed = 0
    for path in sorted(cache_dir.glob("*.json")):
        try:
            path.unlink()
            removed += 1
            print(f"  Deleted {path}")
        except OSError as exc:
            print(f"  Warning: could not delete BM25 cache {path}: {exc}")
    if removed == 0:
        print("  No BM25 cache files found")


def health() -> None:
    """Check health of all services."""

    print("Checking service health...\n")
    all_healthy = True

    # Check Docker image freshness
    print("Docker Image:")
    try:
        req = urllib.request.Request("http://localhost:8000/api/v1/version")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            deps_ok = data.get("deps_fingerprint_ok")
            if deps_ok is True:
                print("  ✅ Dependencies: fresh")
            elif deps_ok is False:
                print("  ❌ Dependencies: STALE — run `make docker-dev`")
                all_healthy = False
            else:
                print("  ℹ️  Dependencies: unknown (not running in Docker)")
    except Exception:
        print("  ❌ API not reachable at localhost:8000")

    # Check Qdrant
    print("\nQdrant:")
    try:
        req = urllib.request.Request(f"{settings.qdrant_url}/", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                print("  ✅ Healthy (200 OK)")
            else:
                print(f"  ❌ Unhealthy (status {resp.status})")
                all_healthy = False
    except Exception as e:
        print(f"  ❌ Unreachable: {e}")
        all_healthy = False

    # Check Redis
    print("\nRedis:")
    try:
        import redis

        redis_client = redis.Redis.from_url(settings.redis_url, socket_timeout=3)
        if redis_client.ping():
            print("  ✅ Healthy (PONG)")
        else:
            print("  ❌ Unhealthy (no PONG)")
            all_healthy = False
        redis_client.close()
    except Exception as e:
        print(f"  ❌ Unreachable: {e}")
        all_healthy = False

    # Check embedding provider
    print("\nEmbedding Provider:")
    provider = settings.embedding_provider
    if provider == "openrouter":
        print(f"  ℹ️  Configured: OpenRouter ({settings.openrouter_embedding_model})")
    elif provider == "nvidia":
        print(f"  ℹ️  Configured: NVIDIA ({settings.nvidia_embedding_model})")
    elif provider == "gemini":
        print(f"  ℹ️  Configured: Gemini ({settings.gemini_embedding_model})")
    else:
        print(f"  ⚠️  Unknown provider: {provider}")
        all_healthy = False

    # Check LLM provider
    print("\nLLM Provider:")
    llm_provider = settings.llm_provider
    if llm_provider == "openrouter":
        print(f"  ℹ️  Configured: OpenRouter ({settings.openrouter_model})")
        if settings.code_llm_provider:
            print(f"  ℹ️  Code Model: {settings.code_llm_provider} ({settings.code_llm_model})")
    elif llm_provider == "nvidia":
        print(f"  ℹ️  Configured: NVIDIA ({settings.nvidia_model})")
    elif llm_provider == "groq":
        print(f"  ℹ️  Configured: Groq ({settings.groq_model})")
    elif llm_provider == "cerebras":
        print(f"  ℹ️  Configured: Cerebras ({settings.cerebras_model})")
    elif llm_provider == "gemini":
        print(f"  ℹ️  Configured: Gemini ({settings.gemini_model})")
    elif llm_provider == "cloudflare":
        print(f"  ℹ️  Configured: Cloudflare ({settings.cloudflare_model})")
    elif llm_provider == "opencodezen":
        print(f"  ℹ️  Configured: OpenCode Zen ({settings.opencodezen_model})")
    elif llm_provider == "opencodego":
        print(f"  ℹ️  Configured: OpenCode Go ({settings.opencodego_model})")
    elif llm_provider == "sambanova":
        print(f"  ℹ️  Configured: SambaNova ({settings.sambanova_model})")
    elif llm_provider == "mistral":
        print(f"  ℹ️  Configured: Mistral ({settings.mistral_model})")
    elif llm_provider == "deepseek":
        print(f"  ℹ️  Configured: DeepSeek ({settings.deepseek_model})")
    elif llm_provider == "zai":
        print(f"  ℹ️  Configured: Z.AI ({settings.zai_model})")
    elif llm_provider == "siliconflow":
        print(f"  ℹ️  Configured: SiliconFlow ({settings.siliconflow_model})")
    elif llm_provider == "together":
        print(f"  ℹ️  Configured: Together AI ({settings.together_model})")
    elif llm_provider == "fireworks":
        print(f"  ℹ️  Configured: Fireworks AI ({settings.fireworks_model})")
    elif llm_provider == "llm7":
        print(f"  ℹ️  Configured: LLM7.io ({settings.llm7_model})")
    elif llm_provider == "agnes":
        print(f"  ℹ️  Configured: Agnes AI ({settings.agnes_model})")
    elif llm_provider == "ollama_cloud":
        print(f"  ℹ️  Configured: Ollama Cloud ({settings.ollama_cloud_model})")
    elif llm_provider == "helyx":
        print(f"  ℹ️  Configured: Helyx AI ({settings.helyx_model})")
    elif llm_provider == "anyapi":
        print(f"  ℹ️  Configured: AnyAPI.ai ({settings.anyapi_model})")
    elif llm_provider == "ollama":
        print(f"  ℹ️  Configured: Ollama ({settings.ollama_model})")
        # Check Ollama health
        try:
            req = urllib.request.Request(f"{settings.ollama_base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print("  ✅ Ollama service healthy")
                else:
                    print(f"  ❌ Ollama unhealthy (status {resp.status})")
                    all_healthy = False
        except Exception as e:
            print(f"  ❌ Ollama unreachable: {e}")
            all_healthy = False
    else:
        print(f"  ⚠️  Unknown provider: {llm_provider}")
        all_healthy = False

    print("\n" + "=" * 40)
    if all_healthy:
        print("✅ All services healthy")
        sys.exit(0)
    else:
        print("❌ Some services are unhealthy")
        sys.exit(1)


def status() -> None:
    """Show ingestion and system status."""

    print("System Status\n" + "=" * 40 + "\n")

    # Check Docker image freshness
    print("Docker Image:")
    try:
        req = urllib.request.Request("http://localhost:8000/api/v1/version")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            deps_ok = data.get("deps_fingerprint_ok")
            if deps_ok is True:
                print("  ✅ Dependencies: fresh")
            elif deps_ok is False:
                print("  ❌ Dependencies: STALE — run `make docker-dev`")
                print(f"     {data.get('deps_stale_message', '')}")
            else:
                print("  ℹ️  Dependencies: unknown (not running in Docker)")
            git_sha = data.get("git_sha", "unknown")
            print(f"  Git SHA: {git_sha[:8] if git_sha else 'unknown'}")
    except Exception:
        print("  ❌ API not reachable at localhost:8000")

    # Check Qdrant collection status
    print("\nQdrant Collection:")
    try:
        req = urllib.request.Request(f"{settings.qdrant_url}/collections/{settings.collection_name}", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if "result" in data:
                result = data["result"]
                print(f"  Collection: {settings.collection_name}")
                print(f"  Status: {result.get('status', 'unknown')}")
                if "vectors_count" in result:
                    print(f"  Vectors: {result.get('vectors_count', 0)}")
                if "segments_count" in result:
                    print(f"  Segments: {result.get('segments_count', 0)}")
            else:
                print("  ❌ Collection not found")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("  ❌ Collection does not exist (run `dec ingest` to create)")
        else:
            print(f"  ❌ Error: {e}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # BM25 tokenizer status
    bm25_status = _get_bm25_status()
    if bm25_status.get("error"):
        print(f"  BM25: unknown ({bm25_status['error']})")
    elif bm25_status.get("hybrid_active"):
        print("  Hybrid search: active")
        print("  BM25: fitted")
        print(f"  BM25 cache: {bm25_status.get('cache_path', 'unknown')}")
    elif bm25_status.get("cache_fitted") and not bm25_status.get("sparse_configured"):
        print("  BM25: fitted but collection has no sparse vectors")
    elif bm25_status.get("sparse_configured") and not bm25_status.get("cache_fitted"):
        print("  ⚠️  Sparse vectors present but BM25 cache missing — hybrid search inactive")
        print("  BM25: not fitted (run `dec ingest` to fit)")
    else:
        print("  BM25: not fitted (run `dec ingest` to fit)")

    # Check active Celery tasks
    print("\nCelery Workers:")
    try:
        import subprocess

        result = subprocess.run(
            ["celery", "-A", "data_engineering_copilot.workers.tasks", "inspect", "active"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "OK" in result.stdout:
            if "- empty -" in result.stdout:
                print("  ✅ No active tasks")
            else:
                print("  ⚠️  Active tasks detected:")
                print(result.stdout)
        else:
            print("  ❌ Workers not responding")
    except Exception as e:
        print(f"  ❌ Could not check workers: {e}")

    # Check crawl frontier DB
    print("\nCrawl Frontier:")
    db_url = settings.crawl_db_url
    if not db_url:
        print("  ⚠️  CRAWL_DB_URL not set")
    else:
        try:
            import asyncpg

            async def _check_pg():
                conn = await asyncpg.connect(db_url, timeout=5)
                try:
                    frontier_count = await conn.fetchval("SELECT COUNT(*) FROM crawl_frontier")
                    edge_count = await conn.fetchval("SELECT COUNT(*) FROM sitemap_edges")
                    states = await conn.fetch("SELECT state, COUNT(*)::int as cnt FROM crawl_frontier GROUP BY state")
                    print(f"  ✅ Connected ({frontier_count} pages, {edge_count} edges)")
                    for row in states:
                        print(f"     {row['state']}: {row['cnt']}")

                    # Show failure breakdown
                    failed_count = await conn.fetchval("SELECT COUNT(*) FROM crawl_frontier WHERE state = 'FAILED'")
                    if failed_count > 0:
                        failed_errors = await conn.fetch(
                            "SELECT last_error, COUNT(*)::int as cnt FROM crawl_frontier "
                            "WHERE state = 'FAILED' AND last_error IS NOT NULL AND last_error != '' "
                            "GROUP BY last_error ORDER BY cnt DESC LIMIT 5"
                        )
                        if failed_errors:
                            print("     Failed breakdown (top errors):")
                            for row in failed_errors:
                                error_preview = (row["last_error"] or "")[:60]
                                print(f"       - {error_preview}: {row['cnt']}")
                finally:
                    await conn.close()

            asyncio.run(_check_pg())
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Check Redis cache
    print("\nRedis Cache:")
    try:
        import redis

        redis_client = redis.Redis.from_url(settings.redis_url, socket_timeout=3)
        if redis_client.ping():
            info = redis_client.info()
            print(f"  ✅ Connected (keys: {info.get('db0', {}).get('keys', 0) if 'db0' in info else 'N/A'})")
        redis_client.close()
    except Exception as e:
        print(f"  ❌ Error: {e}")


def _percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile of a list (sorted internally)."""
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def _compute_spark_eval_result(
    item: dict,
    query: str,
    answer,
    context: str,
    prov_record: dict,
) -> dict:
    """Per-query Spark eval metrics, fusing answer-based recall with retrieval
    provenance (candidate vs final context, per-expected-source fused rank).

    Out-of-scope rows (``out_of_scope: true``) expect a scope refusal: the
    answer must declare the topic is outside the Spark corpus. Their recall
    metrics are not counted toward the Spark thresholds.
    """
    out_of_scope = bool(item.get("out_of_scope", False))
    expected_terms = set(item.get("expected_terms", []))
    expected_urls = set(item.get("expected_urls", []))
    forbidden_terms = [t.lower() for t in item.get("forbidden_terms", [])]

    context_lower = context.lower()
    term_recall = sum(1 for t in expected_terms if t.lower() in context_lower) / max(1, len(expected_terms))
    text_lower = (answer.text or "").lower()
    # A forbidden term (non-Spark tech like Delta Lake / Airflow) surfacing in
    # the final ANSWER is a scope violation: the model recommended out-of-scope
    # tooling instead of answering from the Spark corpus. Incidental mentions
    # in retrieved evidence are not failures.
    forbidden_hits = [t for t in forbidden_terms if t in text_lower]

    retrieved_urls = {c.url for c in answer.sources}
    source_recall = sum(1 for u in expected_urls if u in retrieved_urls) / max(1, len(expected_urls))

    fused_urls = [c["url"] for c in prov_record.get("fused", [])] if prov_record else []
    final_urls = [c["url"] for c in prov_record.get("final_context", [])] if prov_record else []
    candidate_source_recall = sum(1 for u in expected_urls if u in fused_urls) / max(1, len(expected_urls))
    expected_fused_ranks = {u: fused_urls.index(u) for u in expected_urls if u in fused_urls}
    dropped_expected = sorted(u for u in expected_urls if u in fused_urls and u not in final_urls)

    stage = (prov_record or {}).get("stage_times")
    stage = stage if isinstance(stage, dict) else dict(answer.stage_times)
    rerank = (prov_record or {}).get("rerank")
    rerank = rerank if isinstance(rerank, dict) else {}
    text_lower = (answer.text or "").lower()
    insufficient_context = (
        "cannot answer" in text_lower
        or "missing information:" in text_lower
        or '"insufficient_context"' in text_lower
        or "outside my knowledge" in text_lower
    )

    return {
        "id": item.get("id", ""),
        "question": query,
        "out_of_scope": out_of_scope,
        "term_recall": term_recall,
        "source_recall": source_recall,
        "candidate_source_recall": candidate_source_recall,
        "source_count": len(answer.sources),
        "context_chars": len(context),
        "forbidden_term_hits": forbidden_hits,
        "candidate_pool_size": (prov_record or {}).get("candidate_pool_size", 0),
        "rerank_enabled": bool(rerank.get("enabled", False)),
        "rerank_pool_size": rerank.get("pool_size"),
        "cache_hit": bool((prov_record or {}).get("cache_hit", False)),
        "insufficient_context": insufficient_context,
        "expected_fused_ranks": expected_fused_ranks,
        "dropped_expected_urls": dropped_expected,
        "retrieval_ms": stage.get("retrieval"),
        "rerank_ms": stage.get("rerank"),
        "total_ms": stage.get("total"),
        "stage_times": dict(stage),
    }


def _compute_spark_eval_metrics(results: list[dict]) -> dict:
    """Aggregate retrieval-stage metrics across evaluation rows.

    Out-of-scope rows are excluded from the Spark recall thresholds but their
    refusal rate is reported separately. Forbidden-term hits are summed so an
    evidence-quality regression is visible.
    """
    n = len(results)
    in_scope = [r for r in results if not r.get("out_of_scope")]
    m = len(in_scope)

    def avg(key: str, rows: list[dict]) -> float:
        return sum(float(r.get(key) or 0.0) for r in rows) / len(rows) if rows else 0.0

    retrieval_ms = sorted(float(r["retrieval_ms"]) for r in results if r.get("retrieval_ms") is not None)
    return {
        "query_count": n,
        "in_scope_query_count": m,
        "out_of_scope_query_count": n - m,
        "avg_term_recall": avg("term_recall", in_scope),
        "avg_source_recall": avg("source_recall", in_scope),
        "avg_candidate_source_recall": avg("candidate_source_recall", in_scope),
        "insufficient_context_rate": (sum(1 for r in in_scope if r.get("insufficient_context")) / m if m else 0.0),
        "out_of_scope_refusal_rate": (
            sum(1 for r in results if r.get("out_of_scope") and r.get("insufficient_context")) / (n - m)
            if n - m
            else 0.0
        ),
        "queries_dropping_expected_sources": sum(1 for r in in_scope if r.get("dropped_expected_urls")),
        "queries_with_forbidden_term_hits": sum(1 for r in in_scope if r.get("forbidden_term_hits")),
        "queries_with_cache_hit": sum(1 for r in results if r.get("cache_hit")),
        "median_retrieval_ms": _percentile(retrieval_ms, 0.5),
        "p95_retrieval_ms": _percentile(retrieval_ms, 0.95),
    }


def _eval_retrieval_row(query: str, intent: str, expected: list[str], retrieved: list[str], k: int) -> dict:
    """Compute Recall@K, Precision@K, MRR@K for a single query against expected URLs.

    Hit sets are DEDUPED: sources may return several chunks from the same
    page, and counting each duplicate inflated recall past 1.0 on real runs.
    """
    relevant = set(expected)
    topk_hits = {u for u in (retrieved or [])[:k] if u in relevant}
    recall = (len(topk_hits) / len(relevant)) if relevant else 1.0
    precision = (len(topk_hits) / k) if k else 0.0
    seen: set[str] = set()
    mrr = 0.0
    for rank, u in enumerate(retrieved or [], 1):
        if u in relevant and u not in seen:
            mrr = 1.0 / rank
            break
    return {"query": query, "intent": intent, "recall": recall, "precision": precision, "mrr": mrr}


def _disable_rewrites_for_eval(service) -> str:
    """Detach the query rewriter so retrieval-only eval makes zero LLM calls.

    Retrieval quality is a property of retriever+index; live rewrites add a
    confound AND cost ~1 LLM call per query. Eval runs must be deterministic
    and free.
    """
    with contextlib.suppress(AttributeError):
        service.query_rewriter = None
    return "disabled"


def eval_retrieval_main(
    dataset: str | None = None,
    k: int = 10,
    output_dir: str | None = None,
    compare_baseline: str | None = None,
    pool_file: str | None = None,
    batch_size: int | None = None,
) -> int:
    """Source-agnostic retrieval-only evaluation (Recall@K / MRR / Precision@K per intent).

    Runs ``service.answer(..., retrieval_only=True)`` over a recall-format golden
    dataset and reports metrics overall and grouped by ``intent``. With
    ``--output-dir`` writes ``retrieval_eval.json`` (consumed by ``eval-set-baseline``).
    With ``--compare-baseline <path>`` fails (exit 1) when overall Recall@K drops
    below the baseline minus a small tolerance (CI regression gate). A missing or
    unparseable baseline is treated as a warning, not a failure.
    """
    import asyncio
    import time as _time

    from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, percentile, recall_at_k
    from data_engineering_copilot.factory import build_rag_service

    dataset_path = pathlib.Path(dataset) if dataset else pathlib.Path("tests/evaluation/golden/recall_all.jsonl")
    if not dataset_path.exists():
        print(f"❌ Evaluation dataset not found at {dataset_path}")
        return 2

    # Pre-flight: corpus–golden scope check — warn before any LLM/embedding spend.
    try:
        from data_engineering_copilot.config.settings import resolve_active_generation
        from data_engineering_copilot.evaluation.eval_schema import parse_eval_rows
        from data_engineering_copilot.services.eval_coverage import CoverageValidator, resolve_generation_root

        project_root = pathlib.Path(__file__).resolve().parents[1]
        gen = resolve_active_generation()
        root = resolve_generation_root(gen, project_root / "data")
        if root is not None and dataset_path.suffix == ".jsonl":
            vrows = parse_eval_rows(dataset_path)
            vrep = CoverageValidator(root).report(vrows)
            if vrep["fail"]:
                pct = 100 * vrep["fail"] / max(1, vrep["rows"])
                print(
                    f"⚠️  pre-flight: {vrep['fail']}/{vrep['rows']} rows uncoverable "
                    f"against {gen!r} ({pct:.0f}% — expected_urls/terms not in index). "
                    "Results will understate retriever quality; fix golden or regenerate corpus."
                )
                if pct > 50:
                    print("   → For an inscope-only run use --dataset tests/evaluation/golden/recall_inscope.jsonl")
    except Exception:  # noqa: BLE001 — pre-flight is advisory, never blocks
        pass

    queries = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"❌ Invalid JSONL row in {dataset_path}: {exc}")
                    return 2
    if not queries:
        print("❌ No queries loaded from dataset")
        return 2

    print(f"Loaded {len(queries)} retrieval evaluation queries (dataset: {dataset_path.name})\n")
    service = build_rag_service()
    rewrite_mode = _disable_rewrites_for_eval(service)
    print(f"Eval rewrite mode: {rewrite_mode} (zero-LLM retrieval eval)")

    from data_engineering_copilot.evaluation.candidate_pool import load_pool, rank_from_pool, save_pool

    pools = load_pool(pool_file) if pool_file else {}
    replay = bool(pools)
    if replay:
        print(f"REPLAY mode: {len(pools)} frozen pools loaded — no vector-DB/LLM calls")

    per_intent: dict[str, list[float]] = {}
    per_intent_mrr: dict[str, list[float]] = {}
    per_intent_prec: dict[str, list[float]] = {}

    async def run_eval() -> list[dict]:
        if service.reranker is not None and not replay:
            await service.reranker.initialize()

        # Batched path: chunk queries, gather per batch with 2s backoff on RetrievalError.
        # Preserve order, reuse existing pool logic, no new deps.
        if batch_size is not None and batch_size > 0:

            async def _single(item: dict, global_idx: int) -> dict:
                query = item.get("question") or ""
                if not query:
                    return {"_skip": True}
                intent = item.get("intent", "unknown")
                expected = [u for u in (item.get("expected_urls") or []) if u]
                qid = str(item.get("id", f"q{global_idx}"))
                prov: list[dict] = []
                if replay:
                    cand = pools.get(qid) or pools.get(query, [])
                    t_start = t_end = _time.perf_counter()
                    retrieved = rank_from_pool(cand, k)
                else:
                    t_start = _time.perf_counter()
                    answer = await service.answer(
                        query,
                        provenance=prov,
                        bypass_cache=True,
                        retrieval_only=True,
                        expected_urls=expected,
                    )
                    t_end = _time.perf_counter()
                    retrieved = [c.url for c in answer.sources]
                    if pool_file:
                        pools[qid] = [
                            {
                                "url": c.url,
                                "dense_score": float(getattr(c, "dense_score", 0.0) or 0.0),
                                "sparse_score": float(getattr(c, "sparse_score", 0.0) or 0.0),
                                "fused_score": float(getattr(c, "confidence", 0.0) or 0.0),
                            }
                            for c in answer.sources
                        ]
                try:
                    row = _eval_retrieval_row(query, intent, expected, retrieved, k)
                except Exception as exc:  # noqa: BLE001
                    return {"id": qid, "question": query, "error": str(exc), "_intent": intent}
                row["id"] = qid
                row["question"] = query
                row["_intent"] = intent
                row["ndcg"] = ndcg_at_k(retrieved, expected, k)
                row["recall_at_5"] = recall_at_k(retrieved, expected, 5)
                row["recall_at_20"] = recall_at_k(retrieved, expected, 20)
                row["latency_ms"] = (t_end - t_start) * 1000.0
                return row

            rows: list[dict] = []
            # Chunk into batches, gather per batch to bound Qdrant concurrency.
            for batch_start in range(0, len(queries), batch_size):
                batch = queries[batch_start : batch_start + batch_size]
                indices = list(range(batch_start + 1, batch_start + len(batch) + 1))
                # 2s backoff retry on RetrievalError / Vector store query failed (transient Qdrant overload).
                attempt = 0
                batch_rows: list[dict] | None = None
                while True:
                    try:
                        tasks = [_single(item, idx) for item, idx in zip(batch, indices, strict=False)]
                        batch_rows = await asyncio.gather(*tasks)
                        break
                    except Exception as exc:  # noqa: BLE001
                        from data_engineering_copilot.domain.exceptions import RetrievalError

                        msg = str(exc)
                        is_retryable = (
                            isinstance(exc, RetrievalError)
                            or "Vector store query failed" in msg
                            or "RetrievalError" in type(exc).__name__
                        )
                        if is_retryable and attempt == 0:
                            print(f"⚠️  batch {batch_start // batch_size + 1} failed ({exc}); retrying in 2s...")
                            await asyncio.sleep(2)
                            attempt += 1
                            continue
                        if is_retryable and attempt == 1:
                            print(
                                f"⚠️  batch {batch_start // batch_size + 1} retry failed; falling back to sequential with 2s backoff..."
                            )
                            await asyncio.sleep(2)
                            # Sequential fallback preserves order and avoids thundering herd.
                            batch_rows = []
                            for item, idx in zip(batch, indices, strict=False):
                                try:
                                    r = await _single(item, idx)
                                except Exception as seq_exc:  # noqa: BLE001
                                    smsg = str(seq_exc)
                                    is_seq_retryable = (
                                        isinstance(seq_exc, RetrievalError)
                                        or "Vector store query failed" in smsg
                                        or "RetrievalError" in type(seq_exc).__name__
                                    )
                                    if is_seq_retryable:
                                        await asyncio.sleep(2)
                                        try:
                                            r = await _single(item, idx)
                                        except Exception as seq_exc2:  # noqa: BLE001
                                            qid = str(item.get("id", f"q{idx}"))
                                            q = item.get("question") or ""
                                            r = {
                                                "id": qid,
                                                "question": q,
                                                "error": str(seq_exc2),
                                                "_intent": item.get("intent", "unknown"),
                                            }
                                    else:
                                        qid = str(item.get("id", f"q{idx}"))
                                        q = item.get("question") or ""
                                        r = {
                                            "id": qid,
                                            "question": q,
                                            "error": str(seq_exc),
                                            "_intent": item.get("intent", "unknown"),
                                        }
                                batch_rows.append(r)
                            break
                        raise
                assert batch_rows is not None
                # Preserve order: batch_rows aligns with batch order.
                for item, row, g_idx in zip(batch, batch_rows, indices, strict=False):
                    if row.get("_skip"):
                        continue
                    if "error" in row:
                        print(f"[{g_idx}/{len(queries)}] {row.get('id', '')}: ERROR {row['error']}")
                        rows.append(row)
                        continue
                    intent = row.pop("_intent", "unknown")
                    per_intent.setdefault(intent, []).append(row["recall"])
                    per_intent_mrr.setdefault(intent, []).append(row["mrr"])
                    per_intent_prec.setdefault(intent, []).append(row["precision"])
                    rows.append(row)
                    print(
                        f"[{g_idx}/{len(queries)}] {item.get('id', '')} intent={intent}: "
                        f"R@{k}={row['recall']:.2f} MRR={row['mrr']:.2f} P@{k}={row['precision']:.2f}"
                    )
                # 2s backoff between batches to avoid Qdrant thundering herd (only if not last batch).
                if batch_start + batch_size < len(queries):
                    await asyncio.sleep(2)
            return rows

        # Legacy sequential path (batch_size=None) — behavior unchanged for regression safety.
        rows = []
        for i, item in enumerate(queries, 1):
            query = item.get("question") or ""
            if not query:
                continue
            intent = item.get("intent", "unknown")
            expected = [u for u in (item.get("expected_urls") or []) if u]
            qid = str(item.get("id", f"q{i}"))
            prov: list[dict] = []
            if replay:
                cand = pools.get(qid) or pools.get(query, [])
                t_start = t_end = _time.perf_counter()
                retrieved = rank_from_pool(cand, k)
                answer = None  # no service call in replay mode
            else:
                # retrieval_only=True measures raw retrieval (the GraphRAG / CRAG
                # LLM augmentations are skipped there) so the benchmark stays fast
                # and reflects base retrieval quality. Pool depth = service's
                # configured retrieval_top_k.
                t_start = _time.perf_counter()
                answer = await service.answer(
                    query,
                    provenance=prov,
                    bypass_cache=True,
                    retrieval_only=True,
                    expected_urls=expected,
                )
                t_end = _time.perf_counter()
                retrieved = [c.url for c in answer.sources]
                if pool_file:
                    pools[qid] = [
                        {
                            "url": c.url,
                            "dense_score": float(getattr(c, "dense_score", 0.0) or 0.0),
                            "sparse_score": float(getattr(c, "sparse_score", 0.0) or 0.0),
                            "fused_score": float(getattr(c, "confidence", 0.0) or 0.0),
                        }
                        for c in answer.sources
                    ]
            try:
                row = _eval_retrieval_row(query, intent, expected, retrieved, k)
            except Exception as exc:  # noqa: BLE001
                print(f"[{i}/{len(queries)}] {qid}: ERROR {exc}")
                rows.append({"id": qid, "question": query, "error": str(exc)})
                continue
            row["ndcg"] = ndcg_at_k(retrieved, expected, k)
            row["recall_at_5"] = recall_at_k(retrieved, expected, 5)
            row["recall_at_20"] = recall_at_k(retrieved, expected, 20)
            row["latency_ms"] = (t_end - t_start) * 1000.0
            rows.append(row)
            per_intent.setdefault(intent, []).append(row["recall"])
            per_intent_mrr.setdefault(intent, []).append(row["mrr"])
            per_intent_prec.setdefault(intent, []).append(row["precision"])
            print(
                f"[{i}/{len(queries)}] {item.get('id', '')} intent={intent}: "
                f"R@{k}={row['recall']:.2f} MRR={row['mrr']:.2f} P@{k}={row['precision']:.2f}"
            )
        return rows

    rows = asyncio.run(run_eval())
    if pool_file and not replay:
        save_pool(pool_file, pools)
        print(f"Pool written: {pool_file} ({len(pools)} queries)")
    if not rows:
        print("❌ No evaluation results produced")
        return 5

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    scored = [r for r in rows if "recall" in r]
    lats = [r["latency_ms"] for r in scored]
    overall = {
        "recall@k": _avg([r["recall"] for r in scored]),
        "mrr@k": _avg([r["mrr"] for r in scored]),
        "precision@k": _avg([r["precision"] for r in scored]),
        "ndcg@k": _avg([r.get("ndcg", 0.0) for r in scored]),
        "recall_at_5": _avg([r.get("recall_at_5", 0.0) for r in scored]),
        "recall_at_20": _avg([r.get("recall_at_20", 0.0) for r in scored]),
        "latency_ms_p50": percentile(lats, 0.5),
        "latency_ms_p95": percentile(lats, 0.95),
        "k": k,
        "n": len(scored),
    }
    per_intent_metrics = {
        intent: {
            "recall@k": _avg(vals),
            "mrr@k": _avg(per_intent_mrr[intent]),
            "precision@k": _avg(per_intent_prec[intent]),
            "n": len(vals),
        }
        for intent, vals in per_intent.items()
    }

    print("\n" + "=" * 40)
    print("Retrieval Evaluation Summary")
    print(f"Queries: {overall['n']} (k={k})")
    print(f"Overall Recall@{k}:   {overall['recall@k']:.3f}")
    print(f"Overall MRR@{k}:      {overall['mrr@k']:.3f}")
    print(f"Overall Precision@{k}: {overall['precision@k']:.3f}")
    print(f"Overall nDCG@{k}:     {overall['ndcg@k']:.3f}")
    print(f"Recall@5: {overall['recall_at_5']:.3f}  Recall@20: {overall['recall_at_20']:.3f}")
    print(f"Latency ms p50/p95: {overall['latency_ms_p50']:.0f}/{overall['latency_ms_p95']:.0f}")
    for intent, m in sorted(per_intent_metrics.items()):
        print(f"  [{intent}] R@{k}={m['recall@k']:.3f} MRR={m['mrr@k']:.3f} P@{k}={m['precision@k']:.3f} (n={m['n']})")

    if output_dir is not None:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "overall": overall,
            "per_intent": per_intent_metrics,
            "rewrite_mode": rewrite_mode,
            "per_query": [
                {"id": r.get("id", ""), "recall": r["recall"], "mrr": r["mrr"], "ndcg": r.get("ndcg", 0.0)}
                for r in scored
            ],
        }
        (out / "retrieval_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nMetrics written to {out / 'retrieval_eval.json'}")

    if compare_baseline:
        baseline_path = pathlib.Path(compare_baseline)
        if not baseline_path.exists():
            print(f"⚠️  Baseline not found at {baseline_path}; skipping regression gate.")
            return 0
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Could not parse baseline {baseline_path}: {exc}; skipping gate.")
            return 0
        base_recall = float((baseline.get("overall") or {}).get("recall@k", 1.0))
        base_pq = [p["recall"] for p in (baseline.get("per_query") or []) if "recall" in p]
        cur_pq = [r["recall"] for r in scored]
        # Honest inscope gate settings (config/settings.py) with fallbacks for tests.
        try:
            from data_engineering_copilot.config.settings import settings as _gate_settings

            global_tol = float(getattr(_gate_settings, "retrieval_gate_global_tolerance", 0.02))
            global_floor = float(getattr(_gate_settings, "retrieval_gate_global_floor", 0.24))
            per_intent_tol = float(getattr(_gate_settings, "retrieval_gate_per_intent_tolerance", 0.05))
            per_intent_min_n = int(getattr(_gate_settings, "retrieval_gate_per_intent_min_n", 5))
        except Exception:  # noqa: BLE001
            global_tol, global_floor, per_intent_tol, per_intent_min_n = 0.02, 0.24, 0.05, 5
        if base_pq and cur_pq:
            from data_engineering_copilot.evaluation.stats import regression_verdict

            ok, delta, (lo, hi) = regression_verdict(cur_pq, base_pq, tolerance=global_tol)
            print(f"Δ recall@k = {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] (tolerance −{global_tol:.2f})")
            if not ok:
                print(f"❌ Retrieval regression vs baseline (CI low {lo:+.4f} below −{global_tol:.2f} tolerance)")
                return 1
            print("✅ No retrieval regression vs baseline (CI-aware verdict)")
            # Absolute floor for 220-row inscope (baseline_inscope 0.259 -0.02 ≈0.24)
            if overall["recall@k"] < global_floor:
                print(f"❌ Retrieval below absolute floor {global_floor:.2f}: Recall@{k} {overall['recall@k']:.3f}")
                return 1
        elif overall["recall@k"] < base_recall - global_tol:
            print("⚠️  Baseline lacks per_query data; legacy point rule applied")
            print(
                f"❌ Retrieval regression: Recall@{k} {overall['recall@k']:.3f} < baseline {base_recall:.3f} −{global_tol:.2f}"
            )
            return 1
        else:
            print("⚠️  Baseline lacks per_query data; legacy point rule applied")
            print(f"✅ No retrieval regression vs baseline Recall@{k}={base_recall:.3f} (tol −{global_tol:.2f})")
        # Per-intent honest gates: R@10 >= max(0, baseline_intent -0.05) where n>=5.
        per_intent_baseline = baseline.get("per_intent") or {}
        if per_intent_baseline:
            per_intent_failures: list[str] = []
            print(
                f"\nPer-intent deltas (honest inscope: R@{k} >= max(0, baseline-{per_intent_tol:.2f}) where n>={per_intent_min_n}):"
            )
            # Optional bootstrap CIs per intent when per_query vectors can be grouped.
            try:
                from data_engineering_copilot.evaluation.stats import (
                    bootstrap_ci as _bootstrap_ci,
                )
                from data_engineering_copilot.evaluation.stats import (
                    per_intent_tolerance,
                )
            except Exception:  # noqa: BLE001
                _bootstrap_ci = None  # type: ignore[assignment]

                def per_intent_tolerance(baseline_recall: float, n: int, *, floor: float = 0.05) -> float:
                    return floor

            # Build cur per-intent recall lists for CI if available from rows grouped by intent.
            # Rows already aggregated in per_intent_metrics, but also have scored rows without intent;
            # reconstruct per-intent vectors from per_intent dict plus scored grouping.
            for intent, cur_m in sorted(per_intent_metrics.items()):
                base_m = per_intent_baseline.get(intent)
                if base_m is None:
                    print(f"  [{intent}] no baseline — skip (cur R@{k}={cur_m['recall@k']:.3f} n={cur_m['n']})")
                    continue
                base_r = float(base_m.get("recall@k", 0.0))
                cur_r = float(cur_m.get("recall@k", 0.0))
                delta = cur_r - base_r
                n_cur = int(cur_m.get("n", 0))
                n_base = int(base_m.get("n", 0))
                if min(n_cur, n_base) < per_intent_min_n:
                    print(
                        f"  [{intent}] n={n_cur}/{n_base} < {per_intent_min_n} — skip (Δ={delta:+.3f} cur={cur_r:.3f} base={base_r:.3f})"
                    )
                    continue
                tol_i = per_intent_tolerance(base_r, n_base, floor=per_intent_tol)
                required = max(0.0, base_r - tol_i)
                passed = cur_r >= required - 1e-9
                status = "✅" if passed else "❌"
                ci_note = ""
                # If bootstrap_ci available and we have enough samples, show delta CI hint
                if _bootstrap_ci is not None and n_cur >= 5 and n_base >= 5:
                    # Can't do paired CI without aligned per_query intent vectors; show per-intent CI widths instead
                    try:
                        cur_ci = _bootstrap_ci(per_intent.get(intent, []) or [cur_r])
                        ci_note = f" cur CI [{cur_ci[0]:.2f},{cur_ci[1]:.2f}]"
                    except Exception:  # noqa: BLE001
                        ci_note = ""
                print(
                    f"  {status} [{intent}] Δ={delta:+.3f} cur={cur_r:.3f} base={base_r:.3f} gate≥{required:.3f} (n={n_cur}){ci_note}"
                )
                if not passed:
                    per_intent_failures.append(intent)
            # Intents in baseline but missing in current (e.g., no queries for that intent)
            for intent in sorted(per_intent_baseline):
                if intent not in per_intent_metrics:
                    print(
                        f"  [..] [{intent}] missing in current run — skip (baseline n={per_intent_baseline[intent].get('n', 0)})"
                    )
            if per_intent_failures:
                print(
                    f"❌ Per-intent regression: {', '.join(per_intent_failures)} below noise-aware gate max(floor={per_intent_tol:.2f}, 2σ)"
                )
                return 1
            print(f"✅ Per-intent gates passed (noise-aware: tol ≥ {per_intent_tol:.2f} where n>={per_intent_min_n})")

    return 0


def eval_chunking_main(
    strategy: str = "all",
    gold: str = "all",
    output: str = "/tmp/chunking_eval.json",
) -> int:
    """Run isolated chunking quality evaluation (offline) on a gold dataset."""
    from data_engineering_copilot.evaluation.chunking_eval import run_chunking_eval

    try:
        report = run_chunking_eval(strategy, gold, output)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Chunking evaluation failed: {exc}")
        return 2

    print(f"{'Strategy':<15} {'IoU':>6} {'Prec':>6} {'B-Sim':>6} {'Fract':>6}")
    gates = report.get("gates") or {}
    for strat, m in report.items():
        if not isinstance(m, dict) or "iou" not in m:
            continue
        print(
            f"{strat:<15} {m['iou']:>6.3f} {m['precision']:>6.3f} "
            f"{m['boundary_similarity']:>6.3f} {m['fracture_rate']:>6.3f}"
        )
    if gates:
        verdict = "✅" if gates.get("fracture_ok") else "❌"
        print(
            f"\n{verdict} fracture gate: worst={gates.get('worst_fracture_rate', 0):.3f} "
            f"threshold<={gates.get('fracture_threshold', 0):.2f}"
        )
        if not gates.get("fracture_ok"):
            return 1
    print(f"\nReport written to {output}")
    return 0


def evaluate_spark_dataset(dataset_path: pathlib.Path, output_dir: pathlib.Path | None = None) -> int:
    """Run retrieval-recall evaluation against the Spark golden dataset.

    Measures expected-term recall, expected-source recall, candidate-source
    recall, and assembled-context recall. When ``output_dir`` is given, writes
    machine-readable retrieval provenance and aggregate metrics as JSON. Fails
    (exit 1) when the expected-source or expected-term recall thresholds are not
    met. Returns ``0`` on pass, ``2`` on bad input, ``5`` on operational failure.
    """
    import asyncio

    from data_engineering_copilot.domain.models import RetrievedChunk
    from data_engineering_copilot.factory import build_rag_service
    from data_engineering_copilot.services.context_assembler import ContextAssembler

    if not dataset_path.exists():
        print(f"❌ Evaluation dataset not found at {dataset_path}")
        return 2

    queries = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"❌ Invalid JSONL row in {dataset_path}: {exc}")
                    return 2
    if not queries:
        print("❌ No queries loaded from dataset")
        return 2

    print(f"Loaded {len(queries)} Spark evaluation queries (dataset: {dataset_path.name})\n")
    service = build_rag_service()

    provenance_records: list[dict] = []

    async def run_eval() -> list[dict]:
        if service.reranker is not None:
            await service.reranker.initialize()
        results = []
        for i, item in enumerate(queries, 1):
            query = item.get("question") or ""
            if not query:
                continue
            prov: list[dict] = []
            # Only rows that need the generated answer text pay for generation:
            # out-of-scope rows must produce a scope refusal, and rows carrying
            # forbidden terms are checked against the final answer. Every other
            # in-scope row is scored on retrieval alone (recall gates depend only
            # on the assembled context and final chunk sources).
            out_of_scope = bool(item.get("out_of_scope", False))
            forbidden_terms = item.get("forbidden_terms") or []
            needs_answer = out_of_scope or bool(forbidden_terms)
            try:
                # Evaluation must measure actual retrieval/answer quality, never
                # cached answers from earlier runs (which could be stale across
                # generations). Bypass the cache entirely.
                answer = await service.answer(
                    query,
                    provenance=prov,
                    bypass_cache=True,
                    expected_urls=item.get("expected_urls", []),
                    retrieval_only=not needs_answer,
                )
            except Exception as exc:
                print(f"[{i}/{len(queries)}] {item.get('id', '')}: ERROR {exc}")
                results.append({"id": item.get("id", f"q{i}"), "question": query, "error": str(exc)})
                continue
            sources = answer.sources
            assembled = ContextAssembler(max_context_chars=settings.max_context_chars)
            context, _, _ = assembled.assemble(
                [RetrievedChunk(chunk=c, distance=0.1, confidence=0.5) for c in sources],
                deduplicate=False,
            )

            prov_record = prov[-1] if prov else {}
            provenance_records.append(prov_record)
            result = _compute_spark_eval_result(item, query, answer, context, prov_record)
            results.append(result)
            print(
                f"[{i}/{len(queries)}] {item.get('id', '')}: "
                f"term_recall={result['term_recall']:.2f} "
                f"source_recall={result['source_recall']:.2f} "
                f"candidate_recall={result['candidate_source_recall']:.2f} "
                f"(sources={len(sources)}" + ("" if needs_answer else ", retrieval-only)") + ")"
            )
        return results

    results = asyncio.run(run_eval())

    if not results:
        print("❌ No evaluation results produced")
        return 5

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "retrieval_provenance.json").write_text(
            json.dumps(provenance_records, indent=2, default=str), encoding="utf-8"
        )
        metrics = _compute_spark_eval_metrics(results)
        from data_engineering_copilot.evaluation.provenance import config_fingerprint, eval_environment

        metrics["provenance"] = {**eval_environment(settings), "config_fingerprint": config_fingerprint(settings)}
        (output_dir / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        print(f"\nRetrieval diagnostics written to {output_dir}")

    # Out-of-scope rows are excluded from the Spark recall thresholds; they
    # must instead produce a scope refusal (insufficient context).
    in_scope = [r for r in results if not r.get("out_of_scope")]
    out_of_scope = [r for r in results if r.get("out_of_scope")]
    avg_term = (
        sum(r.get("term_recall", 0.0) for r in in_scope if "term_recall" in r) / len(in_scope) if in_scope else 0.0
    )
    avg_source = (
        sum(r.get("source_recall", 0.0) for r in in_scope if "source_recall" in r) / len(in_scope) if in_scope else 0.0
    )
    forbidden_hits = sum(len(r.get("forbidden_term_hits", [])) for r in in_scope)
    refused = sum(1 for r in out_of_scope if r.get("insufficient_context"))

    print("\n" + "=" * 40)
    print("Spark Evaluation Summary")
    print(f"Queries: {len(results)} (in-scope: {len(in_scope)}, out-of-scope: {len(out_of_scope)})")
    print(f"Expected-term recall (assembled context): {avg_term:.3f}")
    print(f"Expected-source recall: {avg_source:.3f}")
    print(f"Forbidden-term hits: {forbidden_hits}")
    print(f"Out-of-scope refusals: {refused}/{len(out_of_scope)}")

    term_threshold = 0.9
    source_threshold = 0.9
    ok = avg_term >= term_threshold and avg_source >= source_threshold
    if forbidden_hits:
        print(f"❌ Evaluation failed: {forbidden_hits} forbidden-term hit(s) in evidence")
        return 1
    if out_of_scope and refused != len(out_of_scope):
        print(f"❌ Evaluation failed: {len(out_of_scope) - refused} out-of-scope row(s) were not refused")
        return 1
    if not ok:
        print("❌ Evaluation failed: recall below threshold")
        print(f"  term_recall < {term_threshold} or source_recall < {source_threshold}")
        return 1
    print("✅ Evaluation passed")
    return 0


def _answer_correctness(answer: str, ground_truth: str) -> float:
    """Lexical correctness signal: token F1 between answer and ground truth.

    Returns 0.0 when no ground truth is available so it never inflates the
    aggregate; rows without ground truth are excluded from the average.
    """
    if not ground_truth:
        return 0.0
    from data_engineering_copilot.services.rag_evaluation import answer_token_f1

    return round(answer_token_f1(ground_truth, answer), 3)


def eval_generation_main(
    dataset: str | None = None,
    n_trials: int = 3,
    output: str | None = None,
    judge_provider_b: str | None = None,
    sample: int = 0,
    stratify_by: str = "intent",
    compare: str | None = None,
) -> int:
    """Evaluate the generation layer alone on a frozen gold-context dataset.

    Supplies ``(question, gold_context)`` directly to the answer LLM and scores
    faithfulness, answer relevance, and a 1-5 correctness rubric. Exits non-zero
    if any gate fails (CI gate). Latency is deliberately not measured.
    """
    import asyncio

    from data_engineering_copilot.evaluation.cost_estimate import enforce_cost_gate, estimate_calls
    from data_engineering_copilot.evaluation.generation_eval import (
        evaluate_generation,
        load_baseline_answers,
        load_generation_dataset,
    )

    default_dataset = pathlib.Path(__file__).parent.parent / "tests" / "evaluation" / "eval_dataset.jsonl"
    eval_path = pathlib.Path(dataset) if dataset else default_dataset
    if not eval_path.exists():
        print(f"❌ Evaluation dataset not found at {eval_path}")
        return 1

    n_rows = len(load_generation_dataset(str(eval_path)))
    estimate = estimate_calls("eval-generation", n_rows, n_trials=n_trials)
    enforce_cost_gate("eval-generation", estimate)
    print(f"💰 Estimated ~{estimate} paid LLM calls ({n_rows} rows)")

    async def _run():
        judge_b = None
        if judge_provider_b:
            from data_engineering_copilot.factory import build_llm_fallback_chain

            judge_b = build_llm_fallback_chain(
                purpose="evaluation",
                app_settings=settings,
                purpose_provider=judge_provider_b,
            )
        return await evaluate_generation(
            str(eval_path),
            settings,
            n_trials=n_trials,
            judge_b=judge_b,
            sample=sample,
            stratify_by=stratify_by,
            compare_answers=load_baseline_answers(compare) if compare else None,
        )

    report = asyncio.run(_run())
    print(report.to_markdown())
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nWrote report to {output}")
    return 0 if report.passed else 2


def eval_judge_calibrate_main(dataset: str | None = None, provider: str | None = None) -> int:
    """Score the evaluation-chain judge against human labels."""
    import json as _json
    import pathlib as _pathlib

    from data_engineering_copilot.evaluation.judge_calibration import (
        KAPPA_GATE,
        RAW_GATE,
        agreement,
        verdict_for,
    )

    path = _pathlib.Path(dataset or "tests/evaluation/golden/judge_calibration.jsonl")
    if not path.exists():
        print(f"❌ calibration dataset missing: {path}")
        return 2
    rows = [_json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unlabeled = [r for r in rows if r.get("needs_label") or r.get("human_faithfulness", -1) < 0]
    if unlabeled:
        print(
            f"❌ {len(unlabeled)} rows still need human labels "
            f"(fill human_faithfulness/human_relevance in {{0,1}}, set needs_label=false)"
        )
        return 2

    from data_engineering_copilot.evaluation.generation_eval import _judge_call_with_retry
    from data_engineering_copilot.factory import build_llm_fallback_chain

    settings_obj = __import__("data_engineering_copilot.config.settings", fromlist=["settings"]).settings
    judge = build_llm_fallback_chain("evaluation", app_settings=settings_obj)

    async def _score(r: dict) -> float:
        context = " ".join(r.get("contexts") or [])[:3000]
        prompt = (
            "You are a strict faithfulness grader. Context is the only truth.\n"
            f"CONTEXT:\n{context}\n"
            f"ANSWER:\n{(r.get('answer') or '')[:2000]}\n"
            'Output ONLY JSON: {"score": <float 0.0-1.0>}'
        )
        return await _judge_call_with_retry(judge, prompt, 0.0, 1.0)

    import asyncio as _asyncio

    async def _run_all() -> list[float]:
        return list(await _asyncio.gather(*[_score(r) for r in rows]))

    scores = _asyncio.run(_run_all())
    y_true_f = [int(r["human_faithfulness"] >= 0.5) for r in rows]
    y_pred_f = [int(s >= 0.5) for s in scores]
    raw_f, kappa_f = agreement(y_true_f, y_pred_f)
    passed = verdict_for(raw_f, kappa_f)
    print(
        f"faithfulness: raw={raw_f:.3f} kappa={kappa_f:.3f} "
        f"(gates raw>={RAW_GATE}, kappa>={KAPPA_GATE}) -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


def eval_proxy_validate_main(dataset: str | None = None, sample: int = 30, k: int = 5) -> int:
    """Judge a deterministic sample of queries; compare proxy labels vs LLM-judge."""
    import asyncio as _asyncio
    import json as _json
    import pathlib as _pathlib

    path = _pathlib.Path(dataset or "tests/evaluation/golden/recall_inscope.jsonl")
    if not path.exists():
        alt = _pathlib.Path("tests/evaluation/golden/recall_all.jsonl")
        if not alt.exists():
            print(f"❌ dataset missing: {path}")
            return 2
        path = alt
    rows = [_json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    idx = _deterministic_sample_indices(n_total=len(rows), n_sample=sample)
    picked = [rows[i] for i in idx]
    print(f"(PAID) judging {len(picked)} queries x top-{k} chunks with the evaluation chain to validate proxy recall…")

    from data_engineering_copilot.config.settings import settings as _settings
    from data_engineering_copilot.evaluation.judge_calibration import agreement
    from data_engineering_copilot.factory import build_llm_fallback_chain, build_rag_service

    service = build_rag_service(_settings)
    judge_chain = build_llm_fallback_chain("evaluation", app_settings=_settings)

    async def _one(row: dict) -> tuple[list[int], list[int]]:
        res = await service.answer(question=row["question"], retrieval_only=True, bypass_cache=True)
        chunks = list(res.sources)[:k]
        proxy_labels = [int(float(getattr(c, "confidence", 0.0) or 0.0) >= 0.45) for c in chunks]
        judged: list[int] = []
        for c in chunks:
            prompt = (
                'Is this chunk relevant to the question? Answer JSON {"score": 0-or-1}.\n'
                f"QUESTION:\n{row['question']}\nCHUNK:\n{getattr(c, 'text', '')[:1500]}"
            )
            from data_engineering_copilot.evaluation.generation_eval import _judge_call_with_retry

            score = await _judge_call_with_retry(judge_chain, prompt, 0.0, 1.0)
            judged.append(int(score >= 0.5))
        return proxy_labels, judged

    async def _all() -> list[tuple[list[int], list[int]]]:
        return await _asyncio.gather(*[_one(r) for r in picked])

    pairs = _asyncio.run(_all())
    proxy_flat = [p for pl, _ in pairs for p in pl]
    judge_flat = [j for _, jl in pairs for j in jl]
    raw, kappa = agreement(proxy_flat, judge_flat)
    print(f"proxy-vs-judge agreement over {len(proxy_flat)} chunk judgments: raw={raw:.3f} kappa={kappa:.3f}")
    print(
        "Guidance: raw >= 0.80 keeps proxy dashboards trustworthy; below => "
        "recalibrate threshold 0.45 or rely on eval-retrieval ground truth."
    )
    return 0 if raw >= 0.80 else 1


def evaluate(
    verbose: bool = False,
    dataset: str | None = None,
    source: str | None = None,
    experiment_name: str | None = None,
    dataset_name: str | None = None,
    output_dir: str | None = None,
    ragas: bool = False,
) -> None:
    """Run RAG evaluation on golden dataset.

    ``dataset`` selects a JSONL dataset file (default ``tests/evaluation/eval_dataset.jsonl``,
    or a per-source file like ``tests/evaluation/eval_dataset_airflow.jsonl``). ``source``
    filters the loaded rows by their ``source_name`` field.

    ``experiment_name`` uploads the evaluated rows to a Langfuse dataset and runs
    a RAG experiment over them (``dataset.run_experiment``). When ``dataset_name``
    is also given, the experiment runs directly against that existing Langfuse
    dataset instead of the freshly evaluated rows.

    ``output_dir`` writes a per-question results JSONL (id, question, confidence,
    correctness, contexts) for drift/bisection.
    """
    import asyncio

    from data_engineering_copilot.factory import build_rag_service

    print("Running RAG Evaluation...\n")

    # Load golden dataset
    eval_path = (
        pathlib.Path(dataset)
        if dataset
        else (pathlib.Path(__file__).parent.parent / "tests" / "evaluation" / "eval_dataset.jsonl")
    )
    if not eval_path.exists():
        print(f"❌ Evaluation dataset not found at {eval_path}")
        sys.exit(1)

    queries = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    if source:
        queries = [q for q in queries if q.get("source_name") == source]
        if not queries:
            print(f"❌ No queries with source_name={source!r} in {eval_path}")
            sys.exit(1)

    from data_engineering_copilot.evaluation.cost_estimate import enforce_cost_gate, estimate_calls

    estimate = estimate_calls("evaluate", len(queries), ragas=ragas)
    enforce_cost_gate("evaluate", estimate)
    print(f"💰 Estimated ~{estimate} paid LLM calls ({len(queries)} rows)")
    print(f"Loaded {len(queries)} evaluation queries (dataset: {eval_path.name})\n")

    # Run evaluation
    service = build_rag_service()

    async def run_eval():
        results = []
        latencies = []
        for i, item in enumerate(queries, 1):
            query = item.get("question") or item.get("query", "")

            print(f"[{i}/{len(queries)}] Query: {query[:60]}...")

            t0 = __import__("time").monotonic()
            answer = await service.answer(query)
            latency = __import__("time").monotonic() - t0
            latencies.append(latency)

            # RAGAS metrics need the retrieved contexts and ground truth.
            contexts = [c.text for c in answer.sources]
            if verbose:
                print(f"  Answer: {answer.text[:200]}...")
                print(f"  Confidence: {answer.confidence:.2f} ({len(contexts)} contexts retrieved, {latency:.1f}s)")
                if answer.sources:
                    print(f"  Sources: {', '.join(c.source_name for c in answer.sources[:3])}")
            else:
                status = "IC" if "INSUFFICIENT_CONTEXT" in (answer.text or "") else f"{answer.confidence:.2f}"
                print(f"  Confidence: {status} ({len(contexts)} contexts, {latency:.1f}s)")
            if latency > 10.0:
                print(f"  ⚠️  Slow query ({latency:.1f}s > 10s threshold)")
            print()

            results.append(
                {
                    "id": item.get("id", f"q{i}"),
                    "query": query,
                    "answer": answer.text,
                    "confidence": answer.confidence,
                    "contexts": contexts,
                    "ground_truth": item.get("ground_truth", ""),
                    "correctness": _answer_correctness(answer.text, item.get("ground_truth", "")),
                    "latency": latency,
                }
            )

        # Latency summary
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[n // 2]
            p95 = sorted_lat[int(n * 0.95)] if n >= 20 else sorted_lat[-1]
            p99 = sorted_lat[int(n * 0.99)] if n >= 100 else sorted_lat[-1]
            print(f"Latency: P50={p50:.1f}s P95={p95:.1f}s P99={p99:.1f}s")

        return results

    results = asyncio.run(run_eval())

    if output_dir:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "per_question_results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nPer-question results written to {out / 'per_question_results.jsonl'}")

    # Summary
    print("\n" + "=" * 40)
    print("Evaluation Complete")
    print(f"Total queries: {len(results)}")
    avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0
    print(f"Average confidence: {avg_confidence:.2f}")
    ic_rate = (
        sum(1 for r in results if "INSUFFICIENT_CONTEXT" in (r.get("answer") or "")) / len(results) if results else 0
    )
    print(f"INSUFFICIENT_CONTEXT rate: {ic_rate:.2f}")
    scored = [r["correctness"] for r in results if r.get("ground_truth")]
    if scored:
        print(
            f"Average answer correctness (token F1 vs ground truth): {sum(scored) / len(scored):.2f} "
            f"({len(scored)} rows with ground truth)"
        )

    # RAGAS metrics (context_recall, context_precision, faithfulness, answer_relevancy)
    # Opt-in only: RAGAS costs ~18-20 paid LLM calls per query.
    ragas_report = None
    if not ragas:
        print("RAGAS evaluation skipped (opt in with --ragas)")
    else:
        try:
            from data_engineering_copilot.services.ragas_evaluation import RagasEvaluator

            evaluator = RagasEvaluator()
            ragas_report = evaluator.evaluate(
                questions=[r["query"] for r in results],
                answers=[r["answer"] for r in results],
                contexts=[r["contexts"] for r in results],
                ground_truth=[r["ground_truth"] for r in results] if any(r["ground_truth"] for r in results) else None,
            )
        except Exception as e:
            print(f"\n⚠️  RAGAS evaluation failed: {e}")
            ragas_report = None

    if ragas_report is not None:
        print("\nRAGAS Metrics:")
        print(f"  context_recall:    {ragas_report.context_recall:.3f}")
        print(f"  context_precision: {ragas_report.context_precision:.3f}")
        print(f"  faithfulness:      {ragas_report.faithfulness:.3f}")
        print(f"  answer_relevancy:  {ragas_report.answer_relevancy:.3f}")
        print(f"  overall:           {ragas_report.overall:.3f}")
    else:
        print("\nRAGAS evaluation skipped: 'ragas' package not installed.")

    # Phase 6 (Task 6.1): upload evaluated rows to a Langfuse dataset
    from data_engineering_copilot.evaluation.langfuse_datasets import upload_evaluation_dataset_rows

    resolved_dataset_name = dataset_name or (
        f"dec-evaluate-{source or 'all'}-{__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%d-%H%M')}"
    )
    uploaded = False
    if results:
        items = [
            {
                "input": {"query": r["query"]},
                "expected_output": {"answer": r["ground_truth"]},
                "metadata": {
                    "confidence": r["confidence"],
                    "latency_ms": round(r["latency"] * 1000, 1),
                    "contexts": r["contexts"],
                },
            }
            for r in results
        ]
        uploaded = upload_evaluation_dataset_rows(dataset_name=resolved_dataset_name, items=items)
        if uploaded:
            print(f"\n📊 Uploaded {len(results)} evaluation results to Langfuse dataset: {resolved_dataset_name}")
        else:
            print("\n⚠️  Langfuse dataset upload skipped (Langfuse unavailable).")

    # Phase 6 (Task 6.2): RAG experiment over the dataset
    if experiment_name and uploaded:
        from data_engineering_copilot.evaluation.langfuse_datasets import run_rag_experiment

        print(f"\n🧪 Running experiment '{experiment_name}' on dataset '{resolved_dataset_name}'...\n")
        result = run_rag_experiment(
            dataset_name=resolved_dataset_name,
            experiment_name=experiment_name,
            source_filter=[source] if source else None,
        )
        if result is not None:
            print(result.format())
        else:
            print("⚠️  Experiment could not be run (Langfuse unavailable).")

    # Drift detection
    if settings.drift_detection_enabled and results:
        from data_engineering_copilot.evaluation.provenance import eval_environment
        from data_engineering_copilot.services.drift_detector import DriftDetector, EvalSnapshot, hash_eval_dataset

        detector = DriftDetector(
            storage_path=settings.drift_eval_history_path,
            window_days=settings.drift_window_days,
        )
        env = eval_environment(settings)
        snapshot = EvalSnapshot(
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            metrics={"confidence": avg_confidence},
            eval_dataset_hash=hash_eval_dataset(eval_path),
            **env,
        )
        detector.record(snapshot)
        report = detector.compare(snapshot)

        if report.drifted:
            print("\n⚠️  DRIFT DETECTED:")
            for c in report.comparisons:
                if c.drifted:
                    print(
                        f"  {c.metric}: {c.baseline:.2f} → {c.current:.2f} (delta: {c.delta:+.2f}, threshold: {c.threshold:.2f})"
                    )
        elif report.comparisons:
            print("\n✅ No drift detected (within thresholds)")
        else:
            print("\n📊 First eval recorded — baseline will be established on next run")


def _is_recall_file(p: pathlib.Path) -> bool:
    from data_engineering_copilot.evaluation.eval_schema import EvalKind, kind_of

    for line in p.read_text(encoding="utf-8").splitlines()[:5]:
        if not line.strip():
            continue
        return kind_of(json.loads(line)) is EvalKind.RECALL
    return False


def _default_coverage_paths(evals_dir: pathlib.Path) -> list[pathlib.Path]:
    """All recall-format dataset files: top-level legacy set + golden recalls."""
    candidates = sorted(evals_dir.glob("*.jsonl")) + sorted((evals_dir / "golden").glob("*.jsonl"))
    return [p for p in candidates if _is_recall_file(p)]


def _deterministic_sample_indices(*, n_total: int, n_sample: int, seed: int = 13) -> list[int]:
    import random as _random

    rng = _random.Random(seed)
    n = max(0, min(n_sample, n_total))
    return rng.sample(range(n_total), n)


def eval_coverage_main(
    dataset: str | None = None,
    generation: str | None = None,
    json_output: bool = False,
) -> int:
    """Validate evaluation datasets against a generation's indexed corpus.

    Every in-scope ``recall`` row must resolve its ``expected_urls`` to
    indexed chunks and its ``expected_terms`` to real corpus content. Fails
    (exit 1) when any row is orphaned/unanswerable; exit 2 on bad input.
    """
    from data_engineering_copilot.config.settings import resolve_active_generation
    from data_engineering_copilot.evaluation.eval_schema import (
        dataset_version_of,
        parse_eval_rows,
        validate_eval_row,
    )
    from data_engineering_copilot.services.eval_coverage import CoverageValidator, resolve_generation_root

    project_root = pathlib.Path(__file__).resolve().parents[1]
    evals_dir = project_root / "tests" / "evaluation"

    gen = generation or settings.active_index_generation or resolve_active_generation()
    root = resolve_generation_root(gen, project_root / "data")
    if root is None:
        print(f"❌ No corpus found for generation {gen!r} (checked data/pinned_corpus, data/spark_corpus)")
        return 2

    paths = [pathlib.Path(dataset)] if dataset else _default_coverage_paths(evals_dir)
    if not paths:
        print("❌ No recall-format evaluation datasets found")
        return 2

    validator = CoverageValidator(root)
    total_rows = total_fail = 0
    file_reports = []
    git_sha: str | None = None
    coverage_matrix: dict = {}
    for p in paths:
        if not p.exists():
            print(f"❌ Dataset not found: {p}")
            return 2
        rows = parse_eval_rows(p)
        schema_errors = [e for r in rows for e in validate_eval_row(r)]
        if schema_errors:
            print(f"❌ Schema errors in {p.name}:")
            for e in schema_errors[:10]:
                print(f"  - {e}")
            return 2
        report = validator.report(rows)
        total_rows += report["rows"]
        total_fail += report["fail"]
        git_sha = report["git_sha"]
        coverage_matrix = report["coverage_matrix"]
        file_reports.append(
            {
                "file": p.name,
                "rows": report["rows"],
                "pass": report["pass"],
                "fail": report["fail"],
                "version": dataset_version_of(p),
            }
        )
        for fail in report["failures"]:
            print(
                f"  ❌ {p.name} {fail['id']}: missing_urls={fail['missing_urls'][:2]} "
                f"missing_terms={fail['missing_terms'][:3]}"
            )

    if json_output:
        print(
            json.dumps(
                {
                    "generation": gen,
                    "corpus_root": str(root),
                    "git_sha": git_sha,
                    "files": file_reports,
                    "rows": total_rows,
                    "pass": total_rows - total_fail,
                    "fail": total_fail,
                },
                indent=2,
            )
        )
    else:
        print(f"\nGeneration: {gen}  (corpus: {root.name}, {validator.indexed_url_count} indexed URLs)")
        if git_sha:
            print(f"Dataset git sha: {git_sha}")
        print("Coverage matrix (intent × doc_type):")
        for cell, n in coverage_matrix["counts"].items():
            print(f"  {cell:<44} n={n}")
        if coverage_matrix["empty_cells"]:
            print(
                f"  ⚠️ empty cells (target ≥1 query per intent × doc_type): {', '.join(coverage_matrix['empty_cells'])}"
            )
        for fr in file_reports:
            ver = f" version={fr['version']}" if fr["version"] else ""
            print(f"  {fr['file']:<36} rows={fr['rows']:>3} pass={fr['pass']:>3} fail={fr['fail']:>3}{ver}")
        print(f"Total: {total_rows} rows, {total_rows - total_fail} pass, {total_fail} fail")

    return 1 if total_fail else 0


def eval_fast_main(
    generation: str | None = None,
    dataset: str | None = None,
    output_dir: str | None = None,
) -> int:
    """Run the free (zero-LLM) layered integrity evaluation.

    Executes the deterministic corpus / chunk / embedding / vector-DB /
    retrieval layers against the active generation's corpus and Qdrant index.
    No LLM calls, no cloud rerank — only the local embedder. Exits 0 on pass,
    1 when any integrity layer reports failures, 2 on infra/corpus problems.
    """
    import asyncio

    from data_engineering_copilot.config.settings import resolve_active_generation
    from data_engineering_copilot.evaluation.eval_schema import EvalKind, kind_of, parse_eval_rows, validate_eval_row
    from data_engineering_copilot.evaluation.fast_eval import run_fast_eval
    from data_engineering_copilot.factory import build_rag_service
    from data_engineering_copilot.services.eval_coverage import resolve_generation_root

    project_root = pathlib.Path(__file__).resolve().parents[1]

    gen = generation or settings.active_index_generation or resolve_active_generation()
    root = resolve_generation_root(gen, project_root / "data")
    if root is None:
        print(f"❌ No corpus found for generation {gen!r} (checked data/pinned_corpus, data/spark_corpus)")
        return 2

    if dataset:
        dataset_path = pathlib.Path(dataset)
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset}")
            return 2
        recall_rows = parse_eval_rows(dataset_path)
        schema_errors = [e for r in recall_rows for e in validate_eval_row(r)]
        if schema_errors:
            print(f"❌ Schema errors in {dataset_path.name}:")
            for e in schema_errors[:10]:
                print(f"  - {e}")
            return 2
        recall_rows = [r for r in recall_rows if kind_of(r) is EvalKind.RECALL]
    else:
        fast_dataset = project_root / "tests" / "evaluation" / "recall_fast.jsonl"
        recall_rows = parse_eval_rows(fast_dataset) if fast_dataset.exists() else []

    # Sanity pairs for the semantic-ordering check.
    sanity_pairs: list[dict] = []
    pairs_path = pathlib.Path(__file__).resolve().parent / "evaluation" / "fast_sanity_pairs.jsonl"
    if pairs_path.exists():
        sanity_pairs = parse_eval_rows(pairs_path)

    # Offline by construction: hardwire the in-process local-hf embedder so
    # eval-fast never depends on .env provider routing or paid API keys.
    from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import (
        LocalSentenceTransformerEmbeddings,
    )

    embedder = LocalSentenceTransformerEmbeddings(
        model_name=settings.local_hf_embedding_model,
        embedding_dimension=settings.get_embedding_dimension(),
    )
    service = build_rag_service()
    store = service.vector_store

    async def _run() -> dict:
        return await run_fast_eval(
            generation=gen,
            embedder=embedder,
            store=store,
            sanity_pairs=sanity_pairs,
            recall_rows=recall_rows,
        )

    try:
        report = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface infra failures cleanly
        print(f"❌ eval-fast failed: {exc}")
        return 2

    if report.get("status") == "error":
        print(f"❌ {report.get('error')}")
        return 2

    if output_dir:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fast_eval.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _print_fast_eval(report)

    failed = bool(
        (report["layers"].get("coverage") or {}).get("fail")
        or (report["layers"].get("chunk") or {}).get("over_token_budget")
        or (report["layers"].get("chunk") or {}).get("oversized")
    )
    return 1 if failed else 0


def _print_fast_eval(report: dict) -> None:
    layers = report["layers"]
    print("\n" + "=" * 40)
    print(f"FAST EVAL — generation {report.get('generation')}")
    print("=" * 40)

    corpus = layers.get("corpus", {})
    print(
        f"Corpus:       {corpus.get('chunk_count')} chunks, {corpus.get('source_count')} sources, "
        f"{corpus.get('content_hash_duplicates')} content-hash dupes, {corpus.get('empty_chunks')} empty"
    )

    chunk = layers.get("chunk", {})
    if chunk.get("count"):
        print(
            f"Chunks:       mean={chunk.get('mean_chars')} chars p95={chunk.get('p95_chars')} "
            f"p99={chunk.get('p99_chars')} oversized={chunk.get('oversized')} "
            f"over_token_budget={chunk.get('over_token_budget')} boundary_issues={chunk.get('boundary_issues')}"
        )

    cov = layers.get("coverage", {})
    print(f"Coverage:     {cov.get('pass')}/{cov.get('rows')} rows pass")

    emb = layers.get("embedding", {})
    consistency = emb.get("consistency", {})
    sanity = emb.get("semantic_sanity", {})
    print(
        f"Embedding:    consistency={consistency.get('similarity')} "
        f"semantic_sanity={sanity.get('passed')}/{sanity.get('pairs')}"
    )

    vdb = layers.get("vectordb", {})
    if "error" in vdb:
        print(f"Vector DB:    error — {vdb['error']}")
    else:
        print(
            f"Vector DB:    {vdb.get('point_count')} points vs {vdb.get('chunk_count')} chunks "
            f"(match={vdb.get('count_matches')}) self_retrieval={vdb.get('self_retrieval_hits')}/{len(vdb.get('self_retrieval', []))}"
        )

    ret = layers.get("retrieval", {})
    print(f"Retrieval:    source_recall={ret.get('source_recall')} MRR={ret.get('mrr')} rows={ret.get('rows')}")


def gen_synthetic_eval_main(
    source: str,
    generation: str | None = None,
    limit: int = 50,
    out: str | None = None,
    testset_size: int = 25,
) -> int:
    """Generate + gate a synthetic recall eval set for one source.

    Deterministic by default (offline). Pass ``--ragas`` to route through
    Ragas ``TestsetGenerator`` (requires ragas + an LLM + embeddings wired
    through the factory). Every row is filtered by ``CoverageValidator``.
    """
    from data_engineering_copilot.config.settings import resolve_active_generation
    from data_engineering_copilot.evaluation.synthetic_generator import generate
    from data_engineering_copilot.services.eval_coverage import resolve_generation_root

    project_root = pathlib.Path(__file__).resolve().parents[1]
    gen = generation or settings.active_index_generation or resolve_active_generation()
    root = resolve_generation_root(gen, project_root / "data")
    if root is None:
        print(f"❌ No corpus found for generation {gen!r}")
        return 2
    out_path = pathlib.Path(out) if out else project_root / "tests" / "evaluation" / f"recall_synthetic_{source}.jsonl"

    ragas_llm = ragas_embeddings = None
    # The Ragas path needs factory-wired LLM/embeddings; default is deterministic.
    written = generate(
        root,
        source,
        out_path,
        limit=limit,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
        testset_size=testset_size,
    )
    print(f"Generation: {gen}  source: {source}")
    print(f"Wrote {written} synthetic rows -> {out_path}")
    if written == 0:
        print("❌ No rows survived the coverage gate")
        return 1
    return 0


def config() -> None:
    """Validate and display configuration."""
    print("Configuration Validation\n" + "=" * 40 + "\n")

    errors = []
    warnings = []

    # Check required settings
    print("Required Settings:")
    required_vars = [
        ("LLM_PROVIDER", settings.llm_provider),
        ("EMBEDDING_PROVIDER", settings.embedding_provider),
        ("QDRANT_URL", settings.qdrant_url),
        ("REDIS_URL", settings.redis_url),
    ]

    for var_name, var_value in required_vars:
        if var_value:
            print(f"  ✅ {var_name}: {var_value}")
        else:
            print(f"  ❌ {var_name}: not set")
            errors.append(f"{var_name} is not set")

    # Validate URLs
    print("\nURL Validation:")

    # Qdrant URL
    if settings.qdrant_url:
        try:
            req = urllib.request.Request(f"{settings.qdrant_url}/", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                print("  ✅ Qdrant URL: reachable")
        except Exception as e:
            print(f"  ❌ Qdrant URL: unreachable ({e})")
            errors.append("Qdrant URL is not reachable")

    # Redis URL
    if settings.redis_url:
        try:
            import redis

            client = redis.Redis.from_url(settings.redis_url, socket_timeout=3)
            if client.ping():
                print("  ✅ Redis URL: reachable")
            client.close()
        except Exception as e:
            print(f"  ❌ Redis URL: unreachable ({e})")
            errors.append("Redis URL is not reachable")

    # Check embedding configuration
    print("\nEmbedding Configuration:")
    dim = settings.get_embedding_dimension()
    provider = settings.embedding_provider
    if provider == "openrouter":
        model = settings.openrouter_embedding_model
    elif provider == "nvidia":
        model = settings.nvidia_embedding_model
    elif provider == "gemini":
        model = settings.gemini_embedding_model
    else:
        model = settings.active_embedding_model_name()
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Dimension: {dim}")

    # Per-purpose LLM configuration
    print("\nPer-Purpose LLM Configuration:")
    purposes = [
        ("Answer", settings.answer_llm_provider, settings.answer_llm_model),
        ("Rewrite", settings.rewrite_llm_provider, settings.rewrite_llm_model),
        ("Groundedness", settings.groundedness_llm_provider, settings.groundedness_llm_model),
        ("Intent", settings.intent_llm_provider, settings.intent_llm_model),
        ("Enrichment", settings.enrichment_llm_provider, settings.enrichment_llm_model),
        ("Evaluation", settings.evaluation_llm_provider, settings.evaluation_llm_model),
        ("Code", settings.code_llm_provider, settings.code_llm_model),
    ]
    for name, provider, model in purposes:
        if provider:
            print(f"  {name}: {provider}/{model or '(global model)'}")
        else:
            print(f"  {name}: (global default — {settings.llm_provider}/{settings.llm_model})")

    # Check collection exists with correct dimension
    try:
        req = urllib.request.Request(f"{settings.qdrant_url}/collections/{settings.collection_name}", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if "result" in data:
                # Check if dimensions match
                # This is a simplified check
                print("  ✅ Collection exists")
    except Exception:
        print("  ℹ️  Collection does not exist yet (will be created on ingest)")

    # Summary
    print("\n" + "=" * 40)
    if errors:
        print(f"❌ Configuration invalid ({len(errors)} errors):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    elif warnings:
        print(f"⚠️  Configuration valid with warnings ({len(warnings)} warnings):")
        for warn in warnings:
            print(f"  - {warn}")
        sys.exit(0)
    else:
        print("✅ Configuration valid")
        sys.exit(0)


def inspect_db() -> None:
    """Inspect Qdrant collection: points, sources, chunk types, sample payload."""
    import collections

    qdrant_url = settings.qdrant_url
    collection_name = settings.collection_name

    print("Qdrant Database Inspection\n" + "=" * 40 + "\n")

    def _section(s: str) -> None:
        print(f"\n{s}\n" + "-" * len(s))

    # ── Collection overview ──────────────────────────────────────────────
    try:
        req = urllib.request.Request(f"{qdrant_url}/collections/{collection_name}", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            result = data.get("result", {})
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  ❌ Collection '{collection_name}' does not exist. Run `dec ingest` to create it.")
        else:
            print(f"  ❌ HTTP error: {e}")
        return
    except Exception as e:
        print(f"  ❌ Could not reach Qdrant at {qdrant_url}: {e}")
        return

    config = result.get("config", {})
    vectors_config = config.get("params", {}).get("vectors", {})
    sparse_config = config.get("params", {}).get("sparse_vectors", {})

    if isinstance(vectors_config, dict) and "dense" in vectors_config:
        dim = vectors_config["dense"].get("size", "?")
        distance = vectors_config["dense"].get("distance", "?")
        mode = "hybrid"
    elif isinstance(vectors_config, dict):
        dim = vectors_config.get("size", "?")
        distance = vectors_config.get("distance", "?")
        mode = "dense"
    else:
        dim = vectors_config.get("size", "?")
        distance = vectors_config.get("distance", "?")
        mode = "dense"

    status = result.get("status", "?")
    points_count = result.get("points_count", 0)
    indexed = result.get("indexed_vectors_count", 0)
    segments = result.get("segments_count", 0)

    vectors_per_point = 0
    if isinstance(vectors_config, dict):
        if "size" in vectors_config:
            vectors_per_point += 1
        else:
            vectors_per_point += len(vectors_config)
    if isinstance(sparse_config, dict):
        vectors_per_point += len(sparse_config)
    total_vectors = points_count * vectors_per_point

    print(f"  Collection:     {collection_name}")
    print(f"  Status:         {status}")
    print(f"  Points:         {points_count:,}")
    if vectors_per_point > 1:
        print(f"  Vectors:        {total_vectors:,}  ({vectors_per_point} per point)")
    print(f"  Indexed:        {indexed:,} of {total_vectors:,}" if total_vectors else f"  Indexed:        {indexed:,}")
    print(f"  Segments:       {segments}")
    print(f"  Mode:           {mode}")
    print(f"  Dense vector:   {dim}d ({distance})")
    print(f"  Sparse:         {'yes (BM25)' if sparse_config else 'no'}")

    # ── Embedding model info ─────────────────────────────────────────────
    _section("Embedding Model")
    provider = settings.embedding_provider
    if provider == "openrouter":
        model = settings.openrouter_embedding_model
    elif provider == "nvidia":
        model = settings.nvidia_embedding_model
    elif provider == "gemini":
        model = settings.gemini_embedding_model
    else:
        model = settings.active_embedding_model_name()
    expected_dim = settings.get_embedding_dimension()
    match_icon = "✅" if (isinstance(dim, int) and dim == expected_dim) or dim == "?" else "⚠️"
    print(f"  Provider:       {provider}")
    print(f"  Model:          {model}")
    print(f"  Expected dim:   {expected_dim}")
    print(f"  Collection dim: {dim}  {match_icon}")

    # ── Scroll points and aggregate payload stats ────────────────────────
    _section("Payload Distribution")
    if points_count == 0:
        print("  (no points in collection)")
        print()
        return

    source_counts: collections.Counter[str] = collections.Counter()
    type_counts: collections.Counter[str] = collections.Counter()
    url_counts: collections.Counter[str] = collections.Counter()
    sample_point: dict | None = None
    seen = 0
    next_offset: object = None

    while seen < points_count:
        body = json.dumps(
            {
                "limit": 1000,
                "with_payload": True,
                "with_vectors": False,
                "offset": next_offset,
            }
        ).encode()
        scroll_req = urllib.request.Request(
            f"{qdrant_url}/collections/{collection_name}/points/scroll",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(scroll_req, timeout=10) as resp:
            scroll_data = json.loads(resp.read().decode())

        points = scroll_data.get("result", {}).get("points", [])
        if not points:
            break

        for pt in points:
            payload = pt.get("payload", {})
            source = payload.get("source_name", "unknown")
            ctype = payload.get("chunk_type", "unknown")
            url = payload.get("url", "unknown")
            source_counts[source] += 1
            type_counts[ctype] += 1
            url_counts[url] += 1
            if sample_point is None:
                sample_point = payload

        seen += len(points)
        next_offset = scroll_data.get("result", {}).get("next_page_offset")
        if next_offset is None:
            break

    # ── Source distribution ──────────────────────────────────────────────
    print(f"\n  Sources ({len(source_counts)}):")
    for source, count in source_counts.most_common():
        print(f"    {source:<40} {count:>6,}")

    # ── Chunk type distribution ──────────────────────────────────────────
    print(f"\n  Chunk Types ({len(type_counts)}):")
    for ctype, count in type_counts.most_common():
        print(f"    {ctype:<40} {count:>6,}")

    # ── Top URLs ─────────────────────────────────────────────────────────
    print("\n  Top URLs (by chunk count):")
    for url, count in url_counts.most_common(10):
        truncated = url if len(url) <= 72 else url[:69] + "..."
        print(f"    {truncated:<72} {count:>6,}")

    # ── Sample payload ───────────────────────────────────────────────────
    _section("Sample Payload (first point)")
    if sample_point:
        for key in (
            "chunk_id",
            "source_name",
            "title",
            "url",
            "chunk_type",
            "word_count",
            "content_hash",
            "section_header",
        ):
            val = sample_point.get(key, "")
            print(f"  {key:<20} {val}")
        heading_path = sample_point.get("heading_path", [])
        if heading_path:
            print(f"  {'heading_path':<20} {list(heading_path)}")
        text = sample_point.get("text", "")
        print(f"\n  {'text (first 300 chars)':<20}")
        print(f"  {'─' * 60}")
        print(f"  {text[:300]}")
    else:
        print("  (no payload data)")

    print()


def cancel(task_id: str) -> None:
    """Cancel a running ingestion task via the API."""
    API_BASE_URL = "http://localhost:8000"
    cancel_url = f"{API_BASE_URL}/api/v1/ingest/{task_id}/cancel"
    req = urllib.request.Request(cancel_url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print(f"Task {task_id} cancelled: {data.get('status', 'unknown')}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Failed to cancel task: HTTP {exc.code}: {body}")
        sys.exit(1)
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        print(f"Cannot reach API server: {exc}\nStart it with: docker compose up -d backend-api celery_worker")
        sys.exit(1)


def _get_plan_phases() -> tuple:
    from data_engineering_copilot.plan_executor import PLAN_PHASES

    return PLAN_PHASES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RAG assistant for data engineering documentation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Crawl documentation and build the QdrantDB index.")
    ingest_parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to crawl per source.")
    ingest_parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Documentation source name to ingest. Repeat to ingest multiple sources. Defaults to all sources.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the local repository.")
    ask_parser.add_argument("question", help="Question to answer.")
    ask_parser.add_argument("--user-id", default=None, help="User identifier recorded on the Langfuse trace.")
    ask_parser.add_argument("--session-id", default=None, help="Session identifier recorded on the Langfuse trace.")
    ask_parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Restrict retrieval to a documentation source name. Repeatable. "
        "Default: auto-route Claude/Anthropic questions to the Claude docs.",
    )

    ingest_claude_parser = subparsers.add_parser(
        "ingest-claude-docs",
        help="Fetch Claude Platform / Claude Code docs from their llms.txt indexes and ingest into Qdrant.",
    )
    ingest_claude_parser.add_argument(
        "--site",
        choices=["platform", "code", "all"],
        default="all",
        help="Which documentation site to ingest (default: all).",
    )
    ingest_claude_parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of markdown files to fetch per site (for a quick smoke run).",
    )

    reenrich_parser = subparsers.add_parser(
        "reenrich",
        help="Re-enrich pages whose contextual enrichment failed: clear chunks/URL-registry, requeue frontier, re-ingest.",
    )
    reenrich_parser.add_argument(
        "--source",
        required=True,
        help="Documentation source name to re-enrich.",
    )
    reenrich_parser.add_argument(
        "--urls",
        type=str,
        default=None,
        help="File with one URL per line ('#' comments allowed). Defaults to the Redis set "
        "ingest:enrichment_failed:<source> populated by the pipeline.",
    )
    reenrich_parser.add_argument(
        "--category",
        choices=["enrichment", "fetch", "embed", "all"],
        default="enrichment",
        help="Failure category to reprocess (default: enrichment). Use 'all' for all failure types.",
    )

    retry_failed_parser = subparsers.add_parser(
        "retry-failed",
        help="Retry all failed pages for a source (fetch/embed/upsert failures).",
    )
    retry_failed_parser.add_argument(
        "--source",
        required=True,
        help="Documentation source name to retry.",
    )
    retry_failed_parser.add_argument(
        "--category",
        choices=["fetch", "embed", "upsert", "all"],
        default=None,
        help="Filter by failure category (default: all).",
    )

    unskip_parser = subparsers.add_parser(
        "unskip",
        help="Re-process SKIPPED pages for a source.",
    )
    unskip_parser.add_argument(
        "--source",
        required=True,
        help="Documentation source name to unskip.",
    )

    subparsers.add_parser(
        "reset-index",
        help="Full clean rebuild: recreate Qdrant + BM25 cache, clear Redis crawl keys, drop PostgreSQL frontier tables.",
    )
    subparsers.add_parser(
        "reset-qdrant",
        help="Delete and recreate the Qdrant collection and its persisted BM25 cache.",
    )
    subparsers.add_parser(
        "reset-crawler-db",
        help="Clear crawler state (Redis crawl:* + PostgreSQL frontier) without touching Qdrant.",
    )
    subparsers.add_parser(
        "clear-query-cache",
        help="Clear the RAG query cache (Redis rag:cache:* exact + semantic tiers) without touching the index.",
    )
    clear_cache_parser = subparsers.add_parser(
        "clear-cache",
        help="Clear cache stores: --query, --embedding, --crawl, --bm25 (default: all).",
    )
    clear_cache_parser.add_argument(
        "--query",
        action="store_true",
        help="Clear the RAG query cache (rag:cache:*).",
    )
    clear_cache_parser.add_argument(
        "--embedding",
        action="store_true",
        help="Clear the embedding cache (embed:cache:*).",
    )
    clear_cache_parser.add_argument(
        "--crawl",
        action="store_true",
        help="Clear the crawl cache (crawl:* + ingest:enrichment_failed:*).",
    )
    clear_cache_parser.add_argument(
        "--bm25",
        action="store_true",
        help="Clear the persisted BM25 tokenizers under .bm25_cache/.",
    )
    clear_cache_parser.add_argument(
        "--all",
        action="store_true",
        help="Clear every cache store (default when no --type flag is given).",
    )
    subparsers.add_parser(
        "spark-config-check",
        help="Validate the pinned Spark source configuration without network access.",
    )
    subparsers.add_parser(
        "gen-config-check",
        help="Validate the pinned sources configuration without network access.",
    )
    subparsers.add_parser(
        "gen-reset",
        help="Wipe all generation state: alias, gen collections, index state, BM25 caches.",
    )
    subparsers.add_parser(
        "gen-stale",
        help="Report generation collections as active, stale, or orphan.",
    )
    gen_manifest_parser = subparsers.add_parser(
        "gen-manifest",
        help="Materialize all pinned sources and write per-source + combined manifests.",
    )
    gen_manifest_parser.add_argument("--generation", type=str, default=None, help="Generation identifier.")
    gen_build_parser = subparsers.add_parser(
        "gen-build",
        help="Build a combined pinned generation collection without activating it.",
    )
    gen_build_parser.add_argument("--generation", type=str, default=None, help="Generation identifier.")
    for _cmd in ("gen-validate", "gen-activate", "gen-rollback"):
        _parser = subparsers.add_parser(_cmd, help=f"Manage pinned generation: {_cmd}.")
        _parser.add_argument("--generation", type=str, required=True, help="Generation identifier.")
    spark_manifest_parser = subparsers.add_parser(
        "spark-manifest",
        help="Materialize the pinned Spark source and write a file manifest.",
    )
    spark_manifest_parser.add_argument("--output", type=str, default=None, help="Output manifest path.")
    spark_render_parser = subparsers.add_parser(
        "spark-render",
        help="Build pinned rendered Spark docs (Jekyll + PySpark Sphinx) and write a rendered manifest.",
    )
    spark_render_parser.add_argument("--generation", type=str, default=None, help="Generation identifier.")
    spark_build_parser = subparsers.add_parser(
        "spark-build",
        help="Build a Spark generation collection without activating it.",
    )
    spark_build_parser.add_argument("--generation", type=str, default=None, help="Generation identifier.")
    for _cmd in ("spark-validate", "spark-activate", "spark-rollback"):
        _parser = subparsers.add_parser(_cmd, help=f"Manage Spark generation: {_cmd}.")
        _parser.add_argument("--generation", type=str, required=True, help="Generation identifier.")
    subparsers.add_parser("ui", help="Print the Streamlit command.")

    profile_parser = subparsers.add_parser("profile", help="Profile ingestion pipeline with concurrency sweep.")
    profile_parser.add_argument(
        "--sources", nargs="*", default=None, help="Documentation sources to profile (default: all)."
    )
    profile_parser.add_argument(
        "--load-sweep",
        type=str,
        default="10,20,50,100",
        help="Comma-separated max-pages values to test under production worker config (default: 10,20,50,100).",
    )
    profile_parser.add_argument(
        "--output-dir",
        type=str,
        default="./profiler_reports",
        help="Directory for reports (default: ./profiler_reports).",
    )

    # Health check
    subparsers.add_parser("health", help="Check health of all services (Redis, Qdrant, LLM, Embeddings).")

    # Status
    subparsers.add_parser("status", help="Show ingestion and system status.")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run RAG evaluation on a golden dataset.")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed per-query output")
    eval_parser.add_argument(
        "--dataset",
        help=(
            "Path to a JSONL evaluation dataset "
            "(default: tests/evaluation/eval_dataset.jsonl; "
            "per-source files: tests/evaluation/eval_dataset_{source}.jsonl)."
        ),
    )
    eval_parser.add_argument(
        "--source",
        help="Only evaluate queries whose `source_name` matches this value.",
    )
    eval_parser.add_argument(
        "--experiment-name",
        default=None,
        help="Upload evaluated rows to a Langfuse dataset and run a RAG experiment with this name.",
    )
    eval_parser.add_argument(
        "--dataset-name",
        default=None,
        help="Langfuse dataset name (default: dec-evaluate-{source}-{date}). With --experiment-name, "
        "run the experiment against this existing dataset directly instead of freshly evaluated rows.",
    )
    eval_parser.add_argument(
        "--spark",
        action="store_true",
        help="Run Spark retrieval-recall evaluation (expected terms/sources).",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Write machine-readable retrieval diagnostics (provenance + metrics JSON) to this directory.",
    )
    eval_parser.add_argument(
        "--ragas",
        action="store_true",
        help="Run optional deep-dive RAGAS metrics (~18-20 extra paid LLM calls per query). Default: off.",
    )

    # Generation-layer evaluation (retrieval frozen)
    eval_gen_parser = subparsers.add_parser(
        "eval-generation",
        help="Evaluate the generation layer alone on a frozen gold-context dataset (faithfulness, relevance, rubric).",
    )
    eval_gen_parser.add_argument(
        "--dataset",
        help="Path to a JSONL dataset with question/contexts/ground_truth (default: tests/evaluation/eval_dataset.jsonl).",
    )
    eval_gen_parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of judge trials averaged for the rubric score (default: 3).",
    )
    eval_gen_parser.add_argument(
        "--output",
        help="Optional path to write the JSON report.",
    )
    eval_gen_parser.add_argument(
        "--judge-provider-b",
        help=(
            "Optional second judge provider for an inter-judge agreement check "
            "(report.judge_agreement = fraction of rows where the two judges' "
            "rubric scores differ by <= 1)."
        ),
    )
    eval_gen_parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Evaluate a deterministic stratified subset of N rows (dev loop). 0 = all rows.",
    )
    eval_gen_parser.add_argument(
        "--stratify-by",
        choices=["intent", "source_name"],
        default="intent",
        help="Stratification key for --sample (default: intent).",
    )
    eval_gen_parser.add_argument(
        "--compare",
        default=None,
        help="Baseline answers JSONL ({id, answer}) for position-swapped A/B.",
    )

    # Config
    subparsers.add_parser("config", help="Validate and display configuration.")

    # Eval coverage validation
    coverage_parser = subparsers.add_parser(
        "eval-coverage",
        help="Validate eval datasets against a generation's indexed corpus.",
    )
    coverage_parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a recall-format JSONL eval dataset (default: all recall files in tests/evaluation).",
    )
    coverage_parser.add_argument(
        "--generation",
        default=None,
        help="Generation to validate against (default: active generation).",
    )
    coverage_parser.add_argument(
        "--json",
        action="store_true",
        help="Output the coverage report as JSON.",
    )

    # Fast (free, zero-LLM) layered integrity evaluation
    fast_parser = subparsers.add_parser(
        "eval-fast",
        help="Run the free deterministic integrity layers (corpus/chunk/embedding/vector-DB/retrieval).",
    )
    fast_parser.add_argument(
        "--generation",
        default=None,
        help="Generation to validate against (default: active generation).",
    )
    fast_parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a recall-format JSONL eval dataset (default: tests/evaluation/recall_fast.jsonl).",
    )
    fast_parser.add_argument(
        "--output-dir",
        default=None,
        help="Write the machine-readable fast_eval.json report to this directory.",
    )

    # Reranker evaluation (nDCG@K, MRR, Precision@K gains)
    eval_rerank_parser = subparsers.add_parser(
        "eval-rerank",
        help="Run isolated reranker evaluation on a golden dataset.",
    )
    eval_rerank_parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a JSONL rerank evaluation dataset (default: tests/evaluation/golden/rerank_eval_sample.jsonl).",
    )
    eval_rerank_parser.add_argument("--k", type=int, default=10, help="Cutoff position for metrics (default: 10).")
    eval_rerank_parser.add_argument("--pool-file", default=None, help="Path to save/load frozen candidate pool.")

    eval_assembly_parser = subparsers.add_parser(
        "eval-assembly",
        help="Run isolated context assembly evaluation on a golden dataset.",
    )
    eval_assembly_parser.add_argument("--dataset", default=None, help="Path to JSONL dataset.")
    eval_assembly_parser.add_argument("--k", type=int, default=20, help="Candidate pool size.")

    eval_prompt_aug_parser = subparsers.add_parser(
        "eval-prompt-aug",
        help="Run isolated prompt augmentation evaluation on a frozen dataset.",
    )
    eval_prompt_aug_parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    eval_prompt_aug_parser.add_argument(
        "--mode",
        choices=["template", "llm"],
        default="template",
        help="Evaluation mode: 'template' (hermetic, no LLM) or 'llm' (calls LLM for actual output quality).",
    )
    eval_prompt_aug_parser.add_argument(
        "--provider",
        default="ollama",
        help="LLM provider for 'llm' mode (default: ollama).",
    )

    # Source-agnostic retrieval-only evaluation (Recall@K/MRR/Precision@K per intent)
    eval_retrieval_parser = subparsers.add_parser(
        "eval-retrieval",
        help="Run source-agnostic retrieval-only evaluation (Recall@K/MRR/Precision@K per intent).",
    )
    eval_retrieval_parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a recall-format JSONL dataset (default: tests/evaluation/golden/recall_all.jsonl).",
    )
    eval_retrieval_parser.add_argument("--k", type=int, default=10, help="Cutoff position for metrics (default: 10).")
    eval_retrieval_parser.add_argument(
        "--output-dir", default=None, help="Write retrieval_eval.json metrics to this directory."
    )
    eval_retrieval_parser.add_argument(
        "--compare-baseline",
        default=None,
        help="Path to a baseline retrieval_eval.json; fail (exit 1) on Recall@K regression.",
    )
    eval_retrieval_parser.add_argument(
        "--pool-file",
        default=None,
        help="Frozen candidate pools JSON. Existing file => offline replay; else fetch+save.",
    )
    eval_retrieval_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for batched retrieval (e.g. 55 for 220-row inscope). None = legacy sequential (single batch).",
    )

    eval_chunking_parser = subparsers.add_parser(
        "eval-chunking",
        help="Run isolated chunking quality evaluation (offline) on a gold dataset.",
    )
    eval_chunking_parser.add_argument(
        "--strategy",
        choices=[
            "all",
            "recursive",
            "sentence",
            "header",
            "structured",
        ],
        default="all",
        help="Chunking strategy to evaluate (strategies supported by _build_chunker).",
    )
    eval_chunking_parser.add_argument(
        "--gold",
        choices=["synthetic", "human", "all"],
        default="all",
        help="Gold dataset source.",
    )
    eval_chunking_parser.add_argument("--output", default="/tmp/chunking_eval.json", help="Output JSON path.")
    eval_chunking_parser.set_defaults(func=eval_chunking_main)

    # Judge-vs-human calibration harness
    calib_parser = subparsers.add_parser(
        "eval-judge-calibrate",
        help="Score judge agreement vs human labels (kappa gate 0.6/raw 0.8).",
    )
    calib_parser.add_argument(
        "--dataset",
        default="tests/evaluation/golden/judge_calibration.jsonl",
    )
    calib_parser.add_argument(
        "--provider",
        default=None,
        help="Pin judge provider; default = evaluation chain order.",
    )
    calib_parser.set_defaults(func=eval_judge_calibrate_main)

    # Proxy-recall validation (paid, opt-in)
    proxyval_parser = subparsers.add_parser(
        "eval-proxy-validate",
        help="(Paid) LLM-judge a deterministic sample to validate confidence-proxy recall.",
    )
    proxyval_parser.add_argument(
        "--dataset",
        default="tests/evaluation/golden/recall_inscope.jsonl",
    )
    proxyval_parser.add_argument("--sample", type=int, default=30)
    proxyval_parser.add_argument("--k", type=int, default=5)
    proxyval_parser.set_defaults(func=eval_proxy_validate_main)

    # Synthetic recall-eval generation
    synth_parser = subparsers.add_parser(
        "gen-synthetic-eval",
        help="Generate a synthetic recall eval set from the active generation's corpus.",
    )
    synth_parser.add_argument(
        "--source", required=True, help="Source name to generate from (e.g. 'Claude Platform Docs')."
    )
    synth_parser.add_argument(
        "--generation", default=None, help="Generation to read the corpus from (default: active)."
    )
    synth_parser.add_argument("--limit", type=int, default=50, help="Max deterministic rows to generate (default: 50).")
    synth_parser.add_argument(
        "--out", default=None, help="Output JSONL path (default: tests/evaluation/recall_synthetic_<source>.jsonl)."
    )
    synth_parser.add_argument(
        "--testset-size", type=int, default=25, help="Ragas testset size (unused in deterministic mode)."
    )

    # Langfuse prompt seeding
    seed_parser = subparsers.add_parser(
        "langfuse-seed-prompts",
        help="Idempotently create/update Langfuse-managed prompts (rag-answer, query-*, groundedness-nli, ...).",
    )
    seed_parser.add_argument("--label", default="production", help="Prompt label (default: production).")
    seed_parser.add_argument(
        "--commit-message",
        default="seed prompts",
        help="Commit message recorded on each prompt version (default: 'seed prompts').",
    )

    # Langfuse production trace evaluation (LLM-as-a-judge)
    eval_run_parser = subparsers.add_parser(
        "langfuse-evaluate",
        help="Run LLM-as-a-judge (faithfulness/relevance/out-of-scope) over production rag-query-pipeline traces.",
    )
    eval_run_parser.add_argument(
        "--filter",
        default=None,
        help='Trace filter array JSON (default: [{"type": "string", "column": "name", "operator": "=", "value": "rag-query-pipeline"}]).',
    )
    eval_run_parser.add_argument("--max-items", type=int, default=None, help="Cap number of traces to judge.")
    eval_run_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Max concurrent evaluator runs (default: 5).",
    )
    eval_run_parser.add_argument("--verbose", action="store_true", help="Verbose output from the SDK runner.")

    # Langfuse score-config seeding
    score_parser = subparsers.add_parser(
        "langfuse-seed-score-configs",
        help="Idempotently create/update Langfuse score configs (confidence, groundedness, cache_hit, intent, ...).",
    )
    score_parser.add_argument(
        "--description-suffix",
        default=None,
        help="Optional suffix appended to score-config descriptions (e.g. an environment name).",
    )

    # Langfuse metrics API
    metrics_parser = subparsers.add_parser(
        "langfuse-metrics",
        help="Query the Langfuse Metrics API v2 (cost, latency, volume, scores).",
    )
    metrics_parser.add_argument(
        "query",
        nargs="?",
        choices=sorted(query_aliases()),
        help="Preset query name (cost-by-model, daily-volume-latency, score-summary).",
    )
    metrics_parser.add_argument("--days", type=int, default=7, help="Look-back window in days (default: 7).")
    metrics_parser.add_argument(
        "--score-name",
        default=None,
        help="For --query score-summary: restrict to one score name.",
    )
    metrics_parser.add_argument("--json", action="store_true", help="Pretty-print the raw JSON rows.")

    # OSS-compatible low-confidence review queue
    review_parser = subparsers.add_parser(
        "langfuse-review-queue",
        help="List low-confidence answers queued for manual review.",
    )
    review_parser.add_argument("--limit", type=int, default=100, help="Maximum items to display (default: 100).")
    review_parser.add_argument("--json", action="store_true", help="Pretty-print review items as JSON.")

    # Inspect DB
    subparsers.add_parser("inspect-db", help="Inspect Qdrant collection: points, sources, chunk types, sample payload.")

    # Cancel task
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running ingestion task.")
    cancel_parser.add_argument("task_id", help="Task ID to cancel.")

    # Ingestion Monitor
    monitor_parser = subparsers.add_parser("ingestion-monitor", help="Live ingestion dashboard (auto-refresh < 30s).")
    monitor_parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL.")
    monitor_parser.add_argument("--task-id", default=None, help="Specific task ID to monitor.")
    monitor_parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds.")

    # Probe LLM providers
    probe_parser = subparsers.add_parser(
        "probe-llm",
        help="Probe each configured LLM/embedding provider with one real request to verify status.",
    )
    probe_parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Probe only these providers (default: all configured). E.g. --providers openrouter groq.",
    )
    probe_parser.add_argument(
        "--purpose",
        type=str,
        default=None,
        help="Probe only the chain for one purpose (answer, rewrite, groundedness, intent, enrichment, evaluation, code).",
    )
    probe_parser.add_argument(
        "--prompt",
        type=str,
        default="Reply with exactly: pong",
        help="Prompt to send to each provider (default: 'Reply with exactly: pong').",
    )
    probe_parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-provider request timeout in seconds (default: 10).",
    )
    probe_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (machine-readable).",
    )
    probe_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show request headers, response preview, dimensions, and token usage.",
    )
    probe_parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip the embedding provider probe (LLM providers only).",
    )
    catalog_parser = subparsers.add_parser(
        "probe-catalog",
        help="Probe free_forever catalog models and build ranked fallback JSON (live by default).",
    )
    catalog_parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Probe only these providers (default: all free_forever models). E.g. --providers openrouter groq.",
    )
    catalog_parser.add_argument(
        "--purpose",
        type=str,
        default=None,
        help="Filter purpose for recommended order (global, answer, code, ...).",
    )
    catalog_parser.add_argument(
        "--prompt",
        type=str,
        default="Reply with exactly: pong",
        help="Prompt to send (default: 'Reply with exactly: pong').",
    )
    catalog_parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-provider timeout in seconds (default: 10).",
    )
    catalog_parser.add_argument(
        "--json",
        action="store_true",
        help="Output catalog as JSON to stdout.",
    )
    catalog_parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode — no network, writes SKIP skeleton.",
    )
    catalog_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for catalog JSON (default: settings.provider_catalog_path).",
    )

    # FLASH-executor driver for the general RAG improvement plan
    rag_plan_parser = subparsers.add_parser(
        "rag-plan",
        help="FLASH-executor driver for the general RAG improvement plan (phases 0-7).",
    )
    rag_plan_parser.add_argument(
        "--phase",
        type=int,
        choices=sorted(p.id for p in (_get_plan_phases())),
        default=None,
        help="Run only this phase (default: all remaining phases).",
    )
    rag_plan_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing anything.",
    )
    rag_plan_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run already-checkpointed phases and allow destructive rollout.",
    )
    rag_plan_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Reuse an existing run directory (resume from checkpoint).",
    )
    rag_plan_parser.add_argument(
        "--candidate-generation",
        type=str,
        default=None,
        help="Generation identifier for phases 3 and 7.",
    )
    rag_plan_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final summary as JSON (machine-readable).",
    )

    return parser


def main() -> None:
    if settings.logging_enabled:
        setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    logger.info("CLI command received command=%s", args.command)

    try:
        if args.command == "ingest":
            ingest(
                max_pages=args.max_pages,
                source_names=tuple(args.source) if args.source else None,
            )
        elif args.command == "ask":
            ask(
                question=args.question,
                user_id=args.user_id,
                session_id=args.session_id,
                source_names=list(args.source) if args.source else None,
            )
        elif args.command == "ingest-claude-docs":
            ingest_claude_docs(site=args.site, max_docs=args.max_docs)
        elif args.command == "reenrich":
            reenrich(source=args.source, urls_file=args.urls, category=args.category)
        elif args.command == "retry-failed":
            retry_failed(source=args.source, category=args.category)
        elif args.command == "unskip":
            unskip(source=args.source)
        elif args.command == "reset-index":
            reset_index()
        elif args.command == "reset-qdrant":
            reset_qdrant()
        elif args.command == "reset-crawler-db":
            reset_crawler_db()
        elif args.command == "clear-query-cache":
            clear_query_cache()
        elif args.command == "clear-cache":
            clear_cache(
                query=args.query,
                embedding=args.embedding,
                crawl=args.crawl,
                bm25=args.bm25,
                all_types=args.all,
            )
        elif args.command == "spark-config-check":
            sys.exit(validate_spark_source_config() or validate_spark_rendered_config())
        elif args.command == "spark-manifest":
            sys.exit(spark_manifest(output=args.output))
        elif args.command == "spark-render":
            sys.exit(spark_render(generation=args.generation))
        elif args.command == "spark-build":
            sys.exit(spark_build(generation=args.generation))
        elif args.command == "spark-validate":
            sys.exit(spark_validate(generation=args.generation))
        elif args.command == "spark-activate":
            sys.exit(spark_activate(generation=args.generation))
        elif args.command == "spark-rollback":
            sys.exit(spark_rollback(generation=args.generation))
        elif args.command == "gen-config-check":
            sys.exit(gen_config_check())
        elif args.command == "gen-reset":
            sys.exit(gen_reset())
        elif args.command == "gen-stale":
            sys.exit(gen_stale())
        elif args.command == "gen-manifest":
            sys.exit(gen_manifest(generation=args.generation))
        elif args.command == "gen-build":
            sys.exit(gen_build(generation=args.generation))
        elif args.command == "gen-validate":
            sys.exit(gen_validate(generation=args.generation))
        elif args.command == "gen-activate":
            sys.exit(gen_activate(generation=args.generation))
        elif args.command == "gen-rollback":
            sys.exit(gen_rollback(generation=args.generation))
        elif args.command == "ui":
            logger.info("CLI ui command displayed Streamlit launch command")
            print("Run: python -m streamlit run data_engineering_copilot/ui/streamlit_app.py")
        elif args.command == "profile":
            profiler_args = [
                "--sources",
                *(args.sources or []),
                "--load-sweep",
                args.load_sweep,
                "--output-dir",
                args.output_dir,
            ]
            profiler_cli.main(profiler_args)
        elif args.command == "health":
            health()
        elif args.command == "status":
            status()
        elif args.command == "evaluate":
            if getattr(args, "spark", False):
                dataset = getattr(args, "dataset", None) or str(
                    pathlib.Path(__file__).parent.parent / "tests" / "evaluation" / "eval_dataset_spark.jsonl"
                )
                output_dir = pathlib.Path(args.output_dir) if getattr(args, "output_dir", None) else None
                sys.exit(evaluate_spark_dataset(pathlib.Path(dataset), output_dir=output_dir))
            if getattr(args, "experiment_name", None) and getattr(args, "dataset_name", None):
                from data_engineering_copilot.evaluation.langfuse_datasets import run_rag_experiment

                print(f"🧪 Running experiment '{args.experiment_name}' on dataset '{args.dataset_name}'...\n")
                result = run_rag_experiment(
                    dataset_name=args.dataset_name,
                    experiment_name=args.experiment_name,
                    source_filter=[args.source] if getattr(args, "source", None) else None,
                )
                if result is None:
                    sys.exit(1)
                print(result.format())
            else:
                evaluate(
                    verbose=getattr(args, "verbose", False),
                    dataset=getattr(args, "dataset", None),
                    source=getattr(args, "source", None),
                    experiment_name=getattr(args, "experiment_name", None),
                    dataset_name=getattr(args, "dataset_name", None),
                    output_dir=getattr(args, "output_dir", None),
                    ragas=getattr(args, "ragas", False),
                )
        elif args.command == "eval-generation":
            sys.exit(
                eval_generation_main(
                    dataset=getattr(args, "dataset", None),
                    n_trials=getattr(args, "n_trials", 3),
                    output=getattr(args, "output", None),
                    judge_provider_b=getattr(args, "judge_provider_b", None),
                    sample=getattr(args, "sample", 0),
                    stratify_by=getattr(args, "stratify_by", "intent"),
                    compare=getattr(args, "compare", None),
                )
            )
        elif args.command == "eval-coverage":
            sys.exit(
                eval_coverage_main(
                    dataset=getattr(args, "dataset", None),
                    generation=getattr(args, "generation", None),
                    json_output=getattr(args, "json", False),
                )
            )
        elif args.command == "eval-fast":
            sys.exit(
                eval_fast_main(
                    generation=getattr(args, "generation", None),
                    dataset=getattr(args, "dataset", None),
                    output_dir=getattr(args, "output_dir", None),
                )
            )
        elif args.command == "eval-rerank":
            import asyncio

            from data_engineering_copilot.evaluation.rerank_eval import (
                RerankEvalServiceAdapter,
                load_rerank_eval_dataset,
                run_rerank_eval,
            )
            from data_engineering_copilot.factory import build_rag_service

            try:
                dataset_path = (
                    pathlib.Path(args.dataset)
                    if args.dataset
                    else pathlib.Path("tests/evaluation/golden/rerank_eval_sample.jsonl")
                )
                dataset = load_rerank_eval_dataset(dataset_path)
                rag_svc = build_rag_service()
                adapter = RerankEvalServiceAdapter(rag_svc)
                pool_path = pathlib.Path(args.pool_file) if args.pool_file else None
                report = asyncio.run(run_rerank_eval(dataset, adapter, k=args.k, candidate_pool_path=pool_path))
                print(report.summary())
            except Exception as exc:  # noqa: BLE001
                print(f"❌ eval-rerank failed: {exc}")
                sys.exit(2)
        elif args.command == "eval-assembly":
            from data_engineering_copilot.evaluation.assembly_eval import load_assembly_eval_dataset, run_assembly_eval
            from data_engineering_copilot.factory import build_rag_service

            try:
                dataset_path = (
                    pathlib.Path(args.dataset)
                    if args.dataset
                    else pathlib.Path("tests/evaluation/golden/assembly_eval_sample.jsonl")
                )
                dataset = load_assembly_eval_dataset(dataset_path)
                rag_svc = build_rag_service()
                reports = run_assembly_eval(dataset, rag_svc, k=args.k)
                for i, r in enumerate(reports):
                    print(f"Query {i + 1}: {r.summary()}")
            except Exception as exc:  # noqa: BLE001
                print(f"❌ eval-assembly failed: {exc}")
                sys.exit(2)
        elif args.command == "eval-prompt-aug":
            import asyncio

            from data_engineering_copilot.evaluation.prompt_aug_eval import run_prompt_aug_eval

            try:
                dataset_path = pathlib.Path(args.dataset)
                if args.mode == "llm":
                    from data_engineering_copilot.evaluation.prompt_aug_eval import run_prompt_aug_eval_llm

                    report = asyncio.run(
                        run_prompt_aug_eval_llm(
                            dataset_path,
                            provider=args.provider,
                            prompt_salted_xml_tags=settings.prompt_salted_xml_tags,
                            prompt_trailing_instructions=settings.prompt_trailing_instructions,
                            prompt_citation_enforcement=settings.prompt_citation_enforcement,
                        )
                    )
                else:
                    report = run_prompt_aug_eval(dataset_path)
                print(report.summary())
            except Exception as exc:  # noqa: BLE001
                print(f"❌ eval-prompt-aug failed: {exc}")
                sys.exit(2)
        elif args.command == "eval-retrieval":
            sys.exit(
                eval_retrieval_main(
                    dataset=getattr(args, "dataset", None),
                    k=getattr(args, "k", 10),
                    output_dir=getattr(args, "output_dir", None),
                    compare_baseline=getattr(args, "compare_baseline", None),
                    pool_file=getattr(args, "pool_file", None),
                    batch_size=getattr(args, "batch_size", None),
                )
            )
        elif args.command == "eval-chunking":
            sys.exit(
                eval_chunking_main(
                    strategy=getattr(args, "strategy", "all"),
                    gold=getattr(args, "gold", "all"),
                    output=getattr(args, "output", "/tmp/chunking_eval.json"),
                )
            )
        elif args.command == "eval-judge-calibrate":
            sys.exit(
                eval_judge_calibrate_main(
                    dataset=getattr(args, "dataset", None),
                    provider=getattr(args, "provider", None),
                )
            )
        elif args.command == "eval-proxy-validate":
            sys.exit(
                eval_proxy_validate_main(
                    dataset=getattr(args, "dataset", None),
                    sample=getattr(args, "sample", 30),
                    k=getattr(args, "k", 5),
                )
            )
        elif args.command == "gen-synthetic-eval":
            sys.exit(
                gen_synthetic_eval_main(
                    source=args.source,
                    generation=getattr(args, "generation", None),
                    limit=getattr(args, "limit", 50),
                    out=getattr(args, "out", None),
                    testset_size=getattr(args, "testset_size", 25),
                )
            )
        elif args.command == "config":
            config()
        elif args.command == "langfuse-seed-prompts":
            from data_engineering_copilot.observability.langfuse_prompts import seed_prompts

            created = seed_prompts(label=args.label, commit_message=args.commit_message)
            for name, prompt in created.items():
                version = getattr(prompt, "version", None)
                print(f"seeded {name} (version {version})" if version is not None else f"seeded {name}")
        elif args.command == "langfuse-evaluate":
            from data_engineering_copilot.evaluation.langfuse_evaluators import run_batched_trace_evaluation

            result = run_batched_trace_evaluation(
                filter=args.filter,
                max_items=args.max_items,
                max_concurrency=args.max_concurrency,
                verbose=args.verbose,
            )
            if result is None:
                print("No evaluation run executed (sampled out by LANGFUSE_SAMPLE_RATE or Langfuse unavailable).")
            else:
                print(f"Evaluated {getattr(result, 'total', '?')} traces")
                print(f"Result: {result}")
        elif args.command == "langfuse-seed-score-configs":
            from data_engineering_copilot.evaluation.langfuse_score_configs import seed_score_configs

            created = seed_score_configs(description_suffix=args.description_suffix)
            for name, is_created in created.items():
                print(f"seeded {name}" if is_created else f"already exists: {name}")
        elif args.command == "langfuse-metrics":
            from data_engineering_copilot.evaluation.langfuse_metrics import (
                cost_by_model,
                daily_volume_and_latency,
                score_summary,
            )

            preset = args.query
            if preset == "cost-by-model":
                rows = cost_by_model(days=args.days)
            elif preset == "daily-volume-latency":
                rows = daily_volume_and_latency(days=args.days)
            elif preset == "score-summary":
                rows = score_summary(name=args.score_name, days=args.days)
            else:
                print("Preset queries:")
                for name, desc in sorted(query_aliases().items()):
                    print(f"  {name:<22} {desc}")
                print("\nExample: dec langfuse-metrics cost-by-model --days 7")
                return
            if args.json:
                print(json.dumps(rows, indent=2, default=str))
                return
            if not rows:
                print("No metrics returned for the selected window.")
                return
            headers = list(rows[0].keys())
            print("\t".join(headers))
            for row in rows:
                print("\t".join(str(row.get(h, "")) for h in headers))
        elif args.command == "langfuse-review-queue":
            from data_engineering_copilot.evaluation.langfuse_datasets import list_review_items

            items = list_review_items(limit=args.limit)
            if args.json:
                print(json.dumps(items, indent=2, default=str))
            elif not items:
                print("Review queue is empty.")
            else:
                for item in items:
                    print(f"[{item['status']}] {item['item_id']}")
                    print(f"Question: {item['question']}")
                    print(f"Answer: {item['answer']}")
                    print(f"Source trace: {item['source_trace_id']}")
                    print(f"Created: {item['created_at']}\n")
        elif args.command == "inspect-db":
            inspect_db()
        elif args.command == "cancel":
            cancel(task_id=args.task_id)
        elif args.command == "ingestion-monitor":
            monitor_main(
                api_url=args.api_url,
                task_id=args.task_id,
                interval=args.interval,
            )
        elif args.command == "probe-llm":
            llm_probe_main(
                providers=args.providers,
                purpose=args.purpose,
                prompt=args.prompt,
                timeout=args.timeout,
                json_output=args.json,
                verbose=args.verbose,
                no_embeddings=args.no_embeddings,
            )
        elif args.command == "probe-catalog":
            catalog_probe_main(
                providers=args.providers,
                purpose=args.purpose,
                prompt=args.prompt,
                timeout=args.timeout,
                json_output=args.json,
                offline=args.offline,
                output=args.output,
            )
        elif args.command == "rag-plan":
            from data_engineering_copilot.plan_executor import PlanOptions, run_plan

            options = PlanOptions(
                run_id=args.run_id,
                phase=args.phase,
                dry_run=args.dry_run,
                force=args.force,
                candidate_generation=args.candidate_generation,
                json_output=args.json,
            )
            sys.exit(run_plan(options))
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("CLI command failed command=%s reason=%s", args.command, exc)
        exc_str = str(exc)
        if "redis" in exc_str and ("Name or service not known" in exc_str or "nodename nor servname" in exc_str):
            print(
                "\nERROR: Cannot resolve the 'redis' hostname. This happens when .env sets REDIS_URL to a Docker "
                "hostname.\n"
                "Fix: Run with REDIS_URL=redis://:local_secure_password_123@localhost:6379/0 "
                "or start Docker services first.\n",
                file=sys.stderr,
            )
        raise


if __name__ == "__main__":
    main()
