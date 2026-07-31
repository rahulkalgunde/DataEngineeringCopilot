from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import urllib.error
import urllib.request

from data_engineering_copilot.cli_monitor import main as monitor_main
from data_engineering_copilot.config.logging import setup_logging
from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.profiler import cli as profiler_cli

logger = logging.getLogger(__name__)


def ingest(max_pages: int | None, source_names: tuple[str, ...] | None) -> None:
    import time

    API_BASE_URL = "http://localhost:8000"

    logger.info("CLI async ingest started max_pages=%s sources=%s", max_pages, source_names or "all")

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

    # Poll progress until completion
    last_status = None
    cancel_url = f"{API_BASE_URL}/api/v1/ingest/{task_id}/cancel"
    try:
        while True:
            status_req = urllib.request.Request(f"{API_BASE_URL}/api/v1/ingest/status/{task_id}")
            try:
                with urllib.request.urlopen(status_req, timeout=5) as resp:
                    progress = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    progress = None
                else:
                    raise

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
                    break
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


def ask(question: str) -> None:
    import asyncio

    from data_engineering_copilot.factory import build_rag_service

    logger.info("CLI ask started question=%r", question[:200])
    service = build_rag_service()
    answer = asyncio.run(service.answer(question))
    logger.info("CLI ask completed confidence=%.4f sources=%s", answer.confidence, len(answer.sources))
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            print(f"- {source.title}: {source.url}")
    print(f"\nConfidence: {answer.confidence:.2f}")


def reset_qdrant() -> None:
    """Delete and recreate the Qdrant collection with the correct dimension/hybrid config."""
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


def reset_index() -> None:
    """Reset the crawl frontier: clear Redis crawl keys and drop PostgreSQL frontier tables.

    Does NOT touch the Qdrant collection — use ``dec reset-qdrant`` for that.
    """
    # Clear crawl-related keys from Redis (URL registry + HTTP conditional-GET cache)
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


def health() -> None:
    """Check health of all services."""

    print("Checking service health...\n")
    all_healthy = True

    # Check Qdrant
    print("Qdrant:")
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

    # Check Qdrant collection status
    print("Qdrant Collection:")
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


def evaluate() -> None:
    """Run RAG evaluation on golden dataset."""
    import asyncio

    from data_engineering_copilot.factory import build_rag_service

    print("Running RAG Evaluation...\n")

    # Load golden dataset
    eval_path = pathlib.Path(__file__).parent.parent / "tests" / "evaluation" / "eval_dataset.jsonl"
    if not eval_path.exists():
        print(f"❌ Evaluation dataset not found at {eval_path}")
        sys.exit(1)

    queries = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"Loaded {len(queries)} evaluation queries\n")

    # Run evaluation
    service = build_rag_service()

    async def run_eval():
        results = []
        for i, item in enumerate(queries, 1):
            query = item.get("query", "")

            print(f"[{i}/{len(queries)}] Query: {query[:60]}...")

            answer = await service.answer(query)

            # TODO: Get retrieved chunk IDs from answer for proper evaluation
            # For now, just show the answer
            print(f"  Answer: {answer.text[:100]}...")
            print(f"  Confidence: {answer.confidence:.2f}")
            print()

            results.append({"query": query, "answer": answer.text, "confidence": answer.confidence})

        return results

    results = asyncio.run(run_eval())

    # Summary
    print("\n" + "=" * 40)
    print("Evaluation Complete")
    print(f"Total queries: {len(results)}")
    avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0
    print(f"Average confidence: {avg_confidence:.2f}")

    # Drift detection
    if settings.drift_detection_enabled and results:
        from data_engineering_copilot.services.drift_detector import DriftDetector, EvalSnapshot, hash_eval_dataset

        detector = DriftDetector(
            storage_path=settings.drift_eval_history_path,
            window_days=settings.drift_window_days,
        )
        snapshot = EvalSnapshot(
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            metrics={"confidence": avg_confidence},
            eval_dataset_hash=hash_eval_dataset(eval_path),
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
        model = settings.embedding_model_name
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

    print(f"  Collection:     {collection_name}")
    print(f"  Status:         {status}")
    print(f"  Points:         {points_count:,}")
    print(f"  Indexed:        {indexed:,} of {points_count:,}")
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
        model = settings.embedding_model_name
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

    subparsers.add_parser(
        "reset-index", help="Reset the crawl frontier: clear Redis crawl keys and drop PostgreSQL frontier tables."
    )
    subparsers.add_parser(
        "reset-qdrant", help="Delete and recreate the Qdrant collection with the correct dimension/hybrid config."
    )
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
    subparsers.add_parser("evaluate", help="Run RAG evaluation on golden dataset.")

    # Config
    subparsers.add_parser("config", help="Validate and display configuration.")

    # Inspect DB
    subparsers.add_parser("inspect-db", help="Inspect Qdrant collection: points, sources, chunk types, sample payload.")

    # Cancel task
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running ingestion task.")
    cancel_parser.add_argument("task_id", help="Task ID to cancel.")

    # Monitor
    monitor_parser = subparsers.add_parser("monitor", help="Live ingestion dashboard (auto-refresh < 30s).")
    monitor_parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL.")
    monitor_parser.add_argument("--task-id", default=None, help="Specific task ID to monitor.")
    monitor_parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds.")

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
            ask(question=args.question)
        elif args.command == "reset-index":
            reset_index()
        elif args.command == "reset-qdrant":
            reset_qdrant()
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
            evaluate()
        elif args.command == "config":
            config()
        elif args.command == "inspect-db":
            inspect_db()
        elif args.command == "cancel":
            cancel(task_id=args.task_id)
        elif args.command == "monitor":
            monitor_main(
                api_url=args.api_url,
                task_id=args.task_id,
                interval=args.interval,
            )
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
