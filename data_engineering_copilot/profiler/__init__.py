"""Profiling, telemetry, and concurrency auto-tuning for the ingestion pipeline."""

from __future__ import annotations

from data_engineering_copilot.profiler.cli import main as cli_main
from data_engineering_copilot.profiler.concurrency_tuner import ConcurrencyTuner, StageRecommendation
from data_engineering_copilot.profiler.rate_limit_tracker import RateLimitEvent, RateLimitTracker
from data_engineering_copilot.profiler.report_generator import ReportGenerator
from data_engineering_copilot.profiler.telemetry import Profiler, ResourceMonitor, StageMetrics

__all__ = [
    "Profiler",
    "ResourceMonitor",
    "StageMetrics",
    "RateLimitEvent",
    "RateLimitTracker",
    "ConcurrencyTuner",
    "StageRecommendation",
    "ReportGenerator",
    "cli_main",
]
