"""Real-time system & per-service resource tracer for ingestion pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass
class StageMetrics:
    """Per-stage performance metrics collected during ingestion."""

    stage_name: str
    items_processed: int = 0
    total_latency_sec: float = 0.0
    errors: int = 0
    rate_limit_hits: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        return self.total_latency_sec / self.items_processed if self.items_processed > 0 else 0.0

    @property
    def throughput_per_sec(self) -> float:
        return self.items_processed / self.total_latency_sec if self.total_latency_sec > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lats = sorted(self.latencies)
        k = (len(sorted_lats) - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_lats) else f
        if f == c:
            return sorted_lats[f]
        return sorted_lats[f] * (c - k) + sorted_lats[c] * (k - f)

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p90(self) -> float:
        return self.percentile(90)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    def record(self, latency: float, items: int = 1, error: bool = False) -> None:
        self.items_processed += items
        self.total_latency_sec += latency
        self.latencies.append(latency)
        if error:
            self.errors += 1


class ResourceMonitor:
    """Non-blocking background system & process resource collector.

    Samples CPU, RSS memory, network I/O, and active asyncio tasks
    at a configurable interval. Runs as a background asyncio task.
    """

    def __init__(self, sample_interval_sec: float = 1.0) -> None:
        self.sample_interval = sample_interval_sec
        self.process = psutil.Process()
        self._running = False
        self._task: asyncio.Task | None = None
        self.snapshots: list[dict[str, Any]] = []

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    @property
    def peak_cpu_pct(self) -> float:
        return max((s["cpu_percent"] for s in self.snapshots), default=0.0)

    @property
    def peak_memory_mb(self) -> float:
        return max((s["rss_mb"] for s in self.snapshots), default=0.0)

    @property
    def avg_cpu_pct(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(s["cpu_percent"] for s in self.snapshots) / len(self.snapshots)

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                cpu_pct = self.process.cpu_percent(interval=0)
                mem_info = self.process.memory_info()
                net_io = psutil.net_io_counters()

                self.snapshots.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_pct,
                    "rss_mb": mem_info.rss / (1024 * 1024),
                    "bytes_sent_mb": net_io.bytes_sent / (1024 * 1024),
                    "bytes_recv_mb": net_io.bytes_recv / (1024 * 1024),
                    "active_asyncio_tasks": len(asyncio.all_tasks()),
                })
            except Exception:
                pass
            await asyncio.sleep(self.sample_interval)


class Profiler:
    """Top-level orchestrator for pipeline profiling.

    Holds a ResourceMonitor and per-stage StageMetrics.
    Provides an async context manager for tracing pipeline stages.
    """

    def __init__(self, sample_interval_sec: float = 1.0) -> None:
        self.resource_monitor = ResourceMonitor(sample_interval_sec)
        self._stages: dict[str, StageMetrics] = {}
        self._start_time: float | None = None

    @contextlib.asynccontextmanager
    async def trace(self, stage_name: str, items: int = 1) -> AsyncGenerator[StageMetrics, None]:
        """Async context manager that measures stage latency and records metrics."""
        stage = self._stages.setdefault(stage_name, StageMetrics(stage_name=stage_name))
        start = time.monotonic()
        error = False
        try:
            yield stage
        except Exception:
            error = True
            raise
        finally:
            latency = time.monotonic() - start
            stage.record(latency, items=items, error=error)

    async def start(self) -> None:
        self._start_time = time.time()
        await self.resource_monitor.start()

    async def stop(self) -> None:
        await self.resource_monitor.stop()

    @property
    def stages(self) -> dict[str, StageMetrics]:
        return dict(self._stages)

    @property
    def elapsed_sec(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_summary(self) -> dict[str, Any]:
        """Return a structured summary of all profiling data."""
        monitor = self.resource_monitor
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_duration_sec": round(self.elapsed_sec, 2),
            "peak_cpu_pct": round(monitor.peak_cpu_pct, 1),
            "avg_cpu_pct": round(monitor.avg_cpu_pct, 1),
            "peak_memory_mb": round(monitor.peak_memory_mb, 2),
            "stages": {
                name: {
                    "items_processed": s.items_processed,
                    "avg_latency": round(s.avg_latency, 4),
                    "p50": round(s.p50, 4),
                    "p90": round(s.p90, 4),
                    "p99": round(s.p99, 4),
                    "throughput_per_sec": round(s.throughput_per_sec, 2),
                    "errors": s.errors,
                    "rate_limit_hits": s.rate_limit_hits,
                }
                for name, s in self._stages.items()
            },
        }
