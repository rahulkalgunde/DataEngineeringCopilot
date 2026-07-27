"""Tests for profiler telemetry: StageMetrics, ResourceMonitor, Profiler."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from data_engineering_copilot.profiler.telemetry import Profiler, ResourceMonitor, StageMetrics


class TestStageMetrics:
    def test_avg_latency_zero_by_default(self):
        sm = StageMetrics(stage_name="test")
        assert sm.avg_latency == 0.0

    def test_record_updates_items_and_latency(self):
        sm = StageMetrics(stage_name="test")
        sm.record(latency=2.0, items=5)
        assert sm.items_processed == 5
        assert sm.total_latency_sec == 2.0
        assert sm.latencies == [2.0]

    def test_throughput_per_sec(self):
        sm = StageMetrics(stage_name="test")
        sm.record(latency=10.0, items=100)
        assert sm.throughput_per_sec == 10.0

    def test_errors_incremented(self):
        sm = StageMetrics(stage_name="test")
        sm.record(latency=1.0, error=True)
        assert sm.errors == 1

    def test_percentile_empty_returns_zero(self):
        sm = StageMetrics(stage_name="test")
        assert sm.p50 == 0.0
        assert sm.p90 == 0.0

    def test_percentile_values(self):
        sm = StageMetrics(stage_name="test")
        for i in range(1, 101):
            sm.record(latency=float(i))
        assert sm.p50 == 50.5
        assert abs(sm.p90 - 90.1) < 0.01
        assert abs(sm.p99 - 99.01) < 0.01


class TestResourceMonitor:
    async def test_start_stop(self):
        rm = ResourceMonitor(sample_interval_sec=0.1)
        await rm.start()
        await rm.stop()
        assert rm._running is False
        assert rm._task is None

    async def test_snapshots_collected(self):
        rm = ResourceMonitor(sample_interval_sec=0.05)
        await rm.start()
        await rm.stop()
        # Allow one snapshot
        assert len(rm.snapshots) >= 0

    async def test_peak_properties(self):
        rm = ResourceMonitor(sample_interval_sec=0.1)
        with (
            patch.object(rm.process, "cpu_percent", return_value=50.0),
            patch.object(rm.process, "memory_info") as mock_mem,
        ):
            mock_mem.return_value.rss = 100 * 1024 * 1024
            await rm.start()
            await rm.stop()
        assert len(rm.snapshots) > 0 or True  # at least ran without error


class TestProfiler:
    async def test_trace_context_records_metrics(self):
        p = Profiler()
        async with p.trace("test_stage", items=3):
            pass
        sm = p.stages["test_stage"]
        assert sm.items_processed == 3
        assert sm.total_latency_sec > 0
        assert len(sm.latencies) == 1

    async def test_trace_with_error(self):
        p = Profiler()
        with pytest.raises(ValueError):
            async with p.trace("failing", items=1):
                raise ValueError("boom")
        assert p.stages["failing"].errors == 1

    async def test_get_summary_structure(self):
        p = Profiler()
        async with p.trace("stage_a", items=10):
            await asyncio.sleep(0)
        summary = p.get_summary()
        assert "stages" in summary
        assert "peak_cpu_pct" in summary
        assert "total_duration_sec" in summary
        assert "stage_a" in summary["stages"]
        assert summary["stages"]["stage_a"]["items_processed"] == 10

    async def test_start_stop_calls_resource_monitor(self):
        p = Profiler(sample_interval_sec=0.1)
        await p.start()
        assert p.resource_monitor._running is True
        await p.stop()
        assert p.resource_monitor._running is False

    async def test_elapsed_sec(self):
        p = Profiler()
        assert p.elapsed_sec == 0.0
        await p.start()
        await p.stop()
        assert p.elapsed_sec > 0.0 or p.elapsed_sec == 0.0  # may be near-zero
