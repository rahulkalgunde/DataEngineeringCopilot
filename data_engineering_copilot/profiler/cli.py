"""CLI harness for running profiling sweeps across load levels via the API."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.profiler.concurrency_tuner import ConcurrencyTuner
from data_engineering_copilot.profiler.report_generator import ReportGenerator
from data_engineering_copilot.profiler.telemetry import Profiler

API_BASE_URL = "http://localhost:8000"


def _parse_sweep(value: str) -> list[int]:
    """Parse a comma-separated string of integers."""
    try:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid load sweep: {value!r}") from exc


def _dispatch_ingest(sources: list[str] | None, max_pages: int) -> str:
    """Dispatch an ingestion task via the FastAPI endpoint.

    Returns the Celery task_id. Raises RuntimeError on failure.
    """
    payload = json.dumps({"source_names": sources, "max_pages": max_pages}).encode()
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
            if not task_id:
                raise RuntimeError(f"API did not return a task_id: {data}")
            return task_id
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ingestion dispatch failed (HTTP {exc.code}): {body}") from exc
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Cannot reach the API server at {API_BASE_URL}: {exc}\n"
            "Start it with: docker compose up -d backend-api celery_worker"
        ) from exc


def _poll_status(task_id: str) -> dict:
    """Poll ingestion progress from the API. Returns the Redis progress doc."""
    req = urllib.request.Request(f"{API_BASE_URL}/api/v1/ingest/status/{task_id}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def _build_concurrency_map(settings: AppSettings, sweep_value: int) -> dict[str, int]:
    """Build a map of stage name to concurrency from the active settings."""
    return {
        "crawler": settings.crawl_async_concurrency,
        "parser": settings.parse_concurrency,
        "chunker": settings.chunk_concurrency,
        "embedder": settings.processing_concurrency,
        "vector_store": 2,
    }


def _build_rate_limit_hits(stages: dict) -> dict[str, int]:
    """Extract rate_limit_hits from stage metrics summary."""
    return {name: s.get("rate_limit_hits", 0) for name, s in stages.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingestion pipeline profiler")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Documentation sources to profile (default: all)",
    )
    parser.add_argument(
        "--load-sweep",
        type=_parse_sweep,
        default="10,20,50,100",
        help="Comma-separated max-pages values to test under production worker config (default: 10,20,50,100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./profiler_reports",
        help="Directory for reports (default: ./profiler_reports)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Resource monitor sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between status polls while ingestion runs (default: 2.0)",
    )
    return parser


async def _run_single(
    profiler: Profiler,
    sources: list[str] | None,
    max_pages: int,
    poll_interval: float,
) -> dict | None:
    """Run a single ingestion pass through the production API path.

    Dispatches via ``POST /api/v1/ingest`` (Celery task), polls progress from
    Redis while collecting host resource metrics, and returns the final
    progress document so the report can include production-style metrics
    (pages fetched, chunks indexed, errors, per-source breakdown).
    """
    task_id = _dispatch_ingest(sources, max_pages)
    print(f"  Dispatched task_id={task_id}")

    await profiler.start()
    # Trace the full ingestion pass as a stage so the report always has data
    # to summarize (duration + throughput), even when the worker reuses a
    # cached crawl and indexes 0 new chunks.
    async with profiler.trace("ingest", items=1):
        try:
            last_status = None
            while True:
                try:
                    progress = _poll_status(task_id)
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        # Task status expired or not yet written; keep waiting.
                        progress = None
                    else:
                        raise

                if progress is not None:
                    status = progress.get("status")
                    if status != last_status:
                        print(f"  Status: {status}")
                        last_status = status
                    if status in ("COMPLETED", "FAILED", "CANCELLED"):
                        return progress
                await asyncio.sleep(poll_interval)
        finally:
            await profiler.stop()


async def run_sweep(args: argparse.Namespace) -> None:
    """Run profiling sweep across load levels using the production path."""
    settings = AppSettings()
    sweep_values = args.load_sweep
    all_results: list[dict] = []
    best_throughput = 0.0
    best_config = None

    for sweep_val in sweep_values:
        print(f"\n--- Load sweep (max_pages={sweep_val}) ---")

        profiler = Profiler(sample_interval_sec=args.sample_interval)
        progress = await _run_single(
            profiler, args.sources, sweep_val, args.poll_interval
        )

        summary = profiler.get_summary()
        stages_summary = summary["stages"]
        peak_cpu = summary["peak_cpu_pct"]
        rl_hits = _build_rate_limit_hits(stages_summary)

        tuner = ConcurrencyTuner()
        concurrency_map = _build_concurrency_map(settings, sweep_val)
        recommendations = tuner.analyze_all(
            stage_metrics=profiler.stages,
            peak_cpu_pct=peak_cpu,
            rate_limit_hits=rl_hits,
            concurrency_map=concurrency_map,
        )

        # Enrich the result with production progress metrics when available.
        prod_metrics = None
        if progress is not None:
            prod_metrics = {
                "pages_fetched": progress.get("pages_fetched", 0),
                "chunks_indexed": progress.get("chunks_indexed", 0),
                "pages_skipped": progress.get("pages_skipped", 0),
                "errors": progress.get("error"),
                "status": progress.get("status"),
            }

        result = {
            "sweep_value": sweep_val,
            "summary": summary,
            "recommendations": recommendations,
            "production_metrics": prod_metrics,
        }
        all_results.append(result)

        # Throughput is based on profiler host metrics; fallback to chunks indexed.
        total_throughput = sum(
            s.get("throughput_per_sec", 0) for s in stages_summary.values()
        )
        if total_throughput <= 0 and prod_metrics is not None:
            total_throughput = prod_metrics["chunks_indexed"] / max(summary["total_duration_sec"], 0.1)
        if total_throughput > best_throughput:
            best_throughput = total_throughput
            best_config = result

        # Print per-sweep summary
        print(f"  Duration: {summary['total_duration_sec']:.1f}s")
        print(f"  Peak CPU: {peak_cpu:.1f}%")
        if prod_metrics is not None:
            print(
                f"  Pages fetched: {prod_metrics['pages_fetched']} | "
                f"Chunks indexed: {prod_metrics['chunks_indexed']} | "
                f"Status: {prod_metrics['status']}"
            )
        for r in recommendations:
            print(f"  {r.stage_name}: {r.current_concurrency} → {r.recommended_concurrency} ({r.action})")

    # Generate final report
    reporter = ReportGenerator()
    if best_config:
        out_dir = reporter.save_report(
            summary=best_config["summary"],
            recommendations=best_config["recommendations"],
            production_metrics=best_config.get("production_metrics"),
            output_dir=args.output_dir,
            name="telemetry_report",
        )
        print(f"\nReport saved to {out_dir}/")

    # Print best config
    if best_config:
        print("\n--- Best Configuration ---")
        print(f"  Load level (max_pages): {best_config['sweep_value']}")
        print(f"  Throughput: {best_throughput:.2f} items/s")
        for r in best_config["recommendations"]:
            print(f"  {r.stage_name}: {r.action}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    asyncio.run(run_sweep(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
