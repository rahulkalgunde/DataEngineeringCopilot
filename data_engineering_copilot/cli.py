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
    print(f"Polling status (Ctrl-C to stop; cancel via: curl -X POST {API_BASE_URL}/api/v1/ingest/{task_id}/cancel)...")

    # Poll progress until completion
    last_status = None
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


def reset_index() -> None:
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

    # Recreate collection with correct dimension for the active embedding provider
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

        asyncio.run(_reset_pg())
    db_path = settings.crawl_db_path
    if db_path.exists():
        db_path.unlink()
        logger.info("Deleted crawl frontier database: %s", db_path)


def migrate_crawl_db() -> None:
    """Migrate crawl frontier from SQLite to PostgreSQL."""
    sqlite_path = settings.crawl_db_path
    pg_url = settings.crawl_db_url

    if not pg_url:
        print("ERROR: CRAWL_DB_URL is not set. Set it to a PostgreSQL connection string.")
        raise SystemExit(1)
    if not sqlite_path.exists():
        print(f"ERROR: SQLite database not found at {sqlite_path}")
        raise SystemExit(1)

    async def _migrate():
        import aiosqlite
        import asyncpg

        # Read from SQLite
        sq_db = await aiosqlite.connect(str(sqlite_path))
        sq_db.row_factory = aiosqlite.Row
        logger.info("Connected to SQLite at %s", sqlite_path)

        frontier_cursor = await sq_db.execute("SELECT * FROM crawl_frontier ORDER BY url_hash")
        frontier_rows = await frontier_cursor.fetchall()
        logger.info("Read %d rows from crawl_frontier", len(frontier_rows))

        edges_cursor = await sq_db.execute("SELECT * FROM sitemap_edges ORDER BY parent_hash, child_hash")
        edges_rows = await edges_cursor.fetchall()
        logger.info("Read %d rows from sitemap_edges", len(edges_rows))

        # Connect to PostgreSQL
        pool = await asyncpg.create_pool(pg_url, min_size=2, max_size=5)
        logger.info("Connected to PostgreSQL at %s", pg_url)

        # Create schema
        from data_engineering_copilot.infrastructure.crawl_db import PG_SCHEMA_SQL

        async with pool.acquire() as conn:
            await conn.execute(PG_SCHEMA_SQL)
        logger.info("Created PostgreSQL schema")

        # Insert crawl_frontier in batches
        BATCH_SIZE = 500
        frontier_cols = [
            "url_hash",
            "url",
            "source_name",
            "state",
            "parent_hash",
            "depth",
            "etag",
            "last_modified",
            "attempts",
            "last_error",
            "created_at",
            "updated_at",
        ]
        placeholders = ", ".join(f"${i + 1}" for i in range(len(frontier_cols)))
        cols_str = ", ".join(frontier_cols)
        insert_sql = f"INSERT INTO crawl_frontier ({cols_str}) VALUES ({placeholders}) ON CONFLICT(url_hash) DO NOTHING"

        inserted_frontier = 0
        for i in range(0, len(frontier_rows), BATCH_SIZE):
            batch = frontier_rows[i : i + BATCH_SIZE]
            async with pool.acquire() as conn:
                for row in batch:
                    await conn.execute(insert_sql, *(row[col] for col in frontier_cols))
                    inserted_frontier += 1
            if (i + BATCH_SIZE) % 2000 == 0 or (i + BATCH_SIZE) >= len(frontier_rows):
                logger.info("  crawl_frontier: %d / %d inserted", inserted_frontier, len(frontier_rows))

        # Insert sitemap_edges in batches
        edge_cols = ["parent_hash", "child_hash"]
        edge_placeholders = ", ".join(f"${i + 1}" for i in range(len(edge_cols)))
        edge_cols_str = ", ".join(edge_cols)
        edge_insert_sql = (
            f"INSERT INTO sitemap_edges ({edge_cols_str}) VALUES ({edge_placeholders}) ON CONFLICT DO NOTHING"
        )

        inserted_edges = 0
        for i in range(0, len(edges_rows), BATCH_SIZE):
            batch = edges_rows[i : i + BATCH_SIZE]
            async with pool.acquire() as conn:
                for row in batch:
                    await conn.execute(edge_insert_sql, row["parent_hash"], row["child_hash"])
                    inserted_edges += 1
            if (i + BATCH_SIZE) % 2000 == 0 or (i + BATCH_SIZE) >= len(edges_rows):
                logger.info("  sitemap_edges: %d / %d inserted", inserted_edges, len(edges_rows))

        # Verify
        async with pool.acquire() as conn:
            pg_frontier = await conn.fetchval("SELECT COUNT(*) FROM crawl_frontier")
            pg_edges = await conn.fetchval("SELECT COUNT(*) FROM sitemap_edges")

        await pool.close()
        await sq_db.close()

        print("\nMigration complete!")
        print(f"  crawl_frontier: {len(frontier_rows)} SQLite → {pg_frontier} PostgreSQL")
        print(f"  sitemap_edges:  {len(edges_rows)} SQLite → {pg_edges} PostgreSQL")
        print(f"\nSQLite database preserved at: {sqlite_path}")
        print("Set CRAWL_DB_URL to empty to revert to SQLite.")

    asyncio.run(_migrate())


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
    if provider == "openai":
        print(f"  ℹ️  Configured: OpenAI ({settings.openai_embedding_model})")
    elif provider == "openrouter":
        print(f"  ℹ️  Configured: OpenRouter ({settings.openrouter_embedding_model})")
    elif provider == "nvidia":
        print(f"  ℹ️  Configured: NVIDIA NIM ({settings.nvidia_embedding_model})")
    else:
        print(f"  ⚠️  Unknown provider: {provider}")
        all_healthy = False

    # Check LLM provider
    print("\nLLM Provider:")
    llm_provider = settings.llm_provider
    if llm_provider == "openrouter":
        print(f"  ℹ️  Configured: OpenRouter ({settings.openrouter_model})")
        # Check code model if configured
        if settings.code_llm_provider:
            print(f"  ℹ️  Code Model: {settings.code_llm_provider} ({settings.code_llm_model})")
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
    db_path = settings.crawl_db_path
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Database exists ({size_mb:.2f} MB)")
    else:
        print("  ℹ️  Database not created yet")

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

    # Check embedding dimension consistency
    print("\nEmbedding Configuration:")
    dim = settings.get_embedding_dimension()
    print(f"  Dimension: {dim}")
    print(f"  Provider: {settings.embedding_provider}")

    if settings.embedding_provider == "openrouter":
        print(f"  Model: {settings.openrouter_embedding_model}")
    elif settings.embedding_provider == "nvidia":
        print(f"  Model: {settings.nvidia_embedding_model}")
    elif settings.embedding_provider == "openai":
        print(f"  Model: {settings.openai_embedding_model}")

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

    subparsers.add_parser("reset-index", help="Delete the Qdrant collection so ingestion can rebuild it.")
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

    # Migrate
    subparsers.add_parser("migrate-crawl-db", help="Migrate crawl frontier from SQLite to PostgreSQL.")

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
        elif args.command == "migrate-crawl-db":
            migrate_crawl_db()
        elif args.command == "monitor":
            monitor_main(
                api_url=args.api_url,
                task_id=args.task_id,
                interval=args.interval,
            )
    except SystemExit:
        raise
    except Exception:
        logger.exception("CLI command failed command=%s", args.command)
        raise


if __name__ == "__main__":
    main()
