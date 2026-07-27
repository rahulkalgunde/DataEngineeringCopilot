"""Tests for ConcurrencyTuner and StageRecommendation."""

from __future__ import annotations

from data_engineering_copilot.profiler.concurrency_tuner import ConcurrencyTuner, StageRecommendation
from data_engineering_copilot.profiler.telemetry import StageMetrics


class TestStageRecommendation:
    def test_dataclass_fields(self):
        r = StageRecommendation(
            stage_name="crawler",
            current_concurrency=10,
            recommended_concurrency=15,
            action="SCALE_UP",
            bottleneck_reason="Room to grow",
            max_throughput_achieved=100.0,
            p99_latency=1.5,
        )
        assert r.stage_name == "crawler"
        assert r.action == "SCALE_UP"


class TestConcurrencyTuner:
    def test_rate_limited_scale_down(self):
        tuner = ConcurrencyTuner(max_rate_limit_ratio=0.02)
        metrics = StageMetrics(stage_name="embedder")
        metrics.record(latency=10.0, items=10)

        rec = tuner.analyze_stage(
            "embedder",
            current_concurrency=10,
            metrics=metrics,
            peak_cpu_pct=30.0,
            rate_limit_hits=3,
        )
        assert rec.action == "RATE_LIMITED"
        assert rec.recommended_concurrency < rec.current_concurrency

    def test_cpu_oversaturation_scale_down(self):
        tuner = ConcurrencyTuner(target_cpu_max_pct=80.0)
        metrics = StageMetrics(stage_name="crawler")
        metrics.record(latency=5.0, items=50)

        rec = tuner.analyze_stage(
            "crawler",
            current_concurrency=20,
            metrics=metrics,
            peak_cpu_pct=95.0,
            rate_limit_hits=0,
        )
        assert rec.action == "SCALE_DOWN"
        expected = max(1, int(20 * (80.0 / 95.0)))
        assert rec.recommended_concurrency == expected

    def test_underutilized_scale_up(self):
        tuner = ConcurrencyTuner()
        metrics = StageMetrics(stage_name="chunker")
        metrics.record(latency=1.0, items=10)

        rec = tuner.analyze_stage(
            "chunker",
            current_concurrency=4,
            metrics=metrics,
            peak_cpu_pct=40.0,
            rate_limit_hits=0,
        )
        assert rec.action == "SCALE_UP"
        assert rec.recommended_concurrency == int(4 * 1.5)

    def test_optimal_when_no_bottleneck(self):
        tuner = ConcurrencyTuner()
        metrics = StageMetrics(stage_name="parser")
        metrics.record(latency=2.0, items=5)

        rec = tuner.analyze_stage(
            "parser",
            current_concurrency=2,
            metrics=metrics,
            peak_cpu_pct=30.0,
            rate_limit_hits=0,
        )
        assert rec.action in ("SCALE_UP", "OPTIMAL")

    def test_scale_up_factor_at_1_returns_optimal(self):
        tuner = ConcurrencyTuner(scale_up_factor=1.0)
        metrics = StageMetrics(stage_name="parser")
        metrics.record(latency=1.0, items=5)

        rec = tuner.analyze_stage(
            "parser",
            current_concurrency=2,
            metrics=metrics,
            peak_cpu_pct=30.0,
            rate_limit_hits=0,
        )
        assert rec.action == "OPTIMAL"

    def test_compute_optimal_concurrency(self):
        result = ConcurrencyTuner.compute_optimal_concurrency(
            peak_throughput=100.0,
            avg_latency=0.5,
            headroom_factor=1.2,
        )
        assert result == 60  # 100 * 0.5 * 1.2 = 60

    def test_compute_optimal_concurrency_zero_throughput(self):
        result = ConcurrencyTuner.compute_optimal_concurrency(
            peak_throughput=0.0,
            avg_latency=0.5,
        )
        assert result == 1

    def test_analyze_all_returns_list(self):
        tuner = ConcurrencyTuner()
        metrics = StageMetrics(stage_name="crawler")
        metrics.record(latency=1.0, items=10)

        recs = tuner.analyze_all(
            stage_metrics={"crawler": metrics},
            peak_cpu_pct=50.0,
            rate_limit_hits={"crawler": 0},
            concurrency_map={"crawler": 5},
        )
        assert len(recs) == 1
        assert recs[0].stage_name == "crawler"

    def test_analyze_all_empty_metrics(self):
        tuner = ConcurrencyTuner()
        recs = tuner.analyze_all(
            stage_metrics={},
            peak_cpu_pct=50.0,
            rate_limit_hits={},
            concurrency_map={},
        )
        assert recs == []
