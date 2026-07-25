"""CLI harness for running profiling sweeps across concurrency levels."""

from __future__ import annotations

import argparse
import asyncio
import sys

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.profiler.concurrency_tuner import ConcurrencyTuner
from data_engineering_copilot.profiler.report_generator import ReportGenerator
from data_engineering_copilot.profiler.telemetry import Profiler


def _parse_sweep(value: str) -> list[int]:
    """Parse a comma-separated string of integers."""
    try:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid concurrency sweep: {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingestion pipeline profiler")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Documentation sources to profile (default: all)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Pages per source for benchmark (default: 10)",
    )
    parser.add_argument(
        "--concurrency-sweep",
        type=_parse_sweep,
        default="1,2,4,8",
        help="Comma-separated concurrency values to test (default: 1,2,4,8)",
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
    return parser


async def _run_single(
    settings: AppSettings,
    profiler: Profiler,
    sources: list[str] | None,
    max_pages: int,
) -> None:
    """Run a single ingestion pass with the profiler attached."""
    from data_engineering_copilot.factory import build_async_ingestion_service

    service = build_async_ingestion_service(app_settings=settings)
    await profiler.start()
    try:
        await service.ingest(
            source_names=sources,
            max_pages_per_source=max_pages,
        )
    finally:
        await profiler.stop()


def _build_concurrency_map(settings: AppSettings, sweep_value: int) -> dict[str, int]:
    """Build a map of stage name to concurrency."""
    return {
        "crawler": settings.crawl_async_concurrency,
        "parser": settings.parse_concurrency,
        "chunker": settings.chunk_concurrency,
        "embedder": sweep_value,
        "vector_store": 2,
    }


def _build_rate_limit_hits(stages: dict) -> dict[str, int]:
    """Extract rate_limit_hits from stage metrics summary."""
    return {name: s.get("rate_limit_hits", 0) for name, s in stages.items()}


async def run_sweep(args: argparse.Namespace) -> None:
    """Run profiling sweep across concurrency levels."""
    sweep_values = args.concurrency_sweep
    all_results: list[dict] = []
    best_throughput = 0.0
    best_config = None

    for sweep_val in sweep_values:
        print(f"\n--- Concurrency sweep: {sweep_val} ---")

        settings = AppSettings(
            processing_concurrency=sweep_val,
            crawl_async_concurrency=sweep_val * 2,
            parse_concurrency=min(sweep_val, 4),
            chunk_concurrency=min(sweep_val, 4),
        )

        profiler = Profiler(sample_interval_sec=args.sample_interval)
        await _run_single(settings, profiler, args.sources, args.max_pages)

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

        result = {
            "sweep_value": sweep_val,
            "summary": summary,
            "recommendations": recommendations,
        }
        all_results.append(result)

        total_throughput = sum(
            s.get("throughput_per_sec", 0) for s in stages_summary.values()
        )
        if total_throughput > best_throughput:
            best_throughput = total_throughput
            best_config = result

        # Print per-sweep summary
        print(f"  Duration: {summary['total_duration_sec']:.1f}s")
        print(f"  Peak CPU: {peak_cpu:.1f}%")
        for r in recommendations:
            print(f"  {r.stage_name}: {r.current_concurrency} → {r.recommended_concurrency} ({r.action})")

    # Generate final report
    reporter = ReportGenerator()
    if best_config:
        out_dir = reporter.save_report(
            summary=best_config["summary"],
            recommendations=best_config["recommendations"],
            output_dir=args.output_dir,
            name="telemetry_report",
        )
        print(f"\nReport saved to {out_dir}/")

    # Print best config
    if best_config:
        print("\n--- Best Configuration ---")
        print(f"  Concurrency: {best_config['sweep_value']}")
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
