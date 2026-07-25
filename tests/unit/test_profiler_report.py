"""Tests for ReportGenerator."""

from __future__ import annotations

import json
import tempfile

from data_engineering_copilot.profiler.concurrency_tuner import StageRecommendation
from data_engineering_copilot.profiler.report_generator import ReportGenerator


class TestReportGenerator:
    def test_generate_markdown(self):
        reporter = ReportGenerator()
        summary = {
            "timestamp": "2024-01-01T00:00:00",
            "total_duration_sec": 120.5,
            "peak_cpu_pct": 75.3,
            "avg_cpu_pct": 50.0,
            "peak_memory_mb": 512.0,
            "stages": {
                "crawler": {
                    "items_processed": 100,
                    "avg_latency": 0.5,
                    "p50": 0.4,
                    "p90": 0.8,
                    "p99": 1.2,
                    "throughput_per_sec": 200.0,
                    "errors": 0,
                    "rate_limit_hits": 0,
                }
            },
        }
        recommendations = [
            StageRecommendation(
                stage_name="crawler",
                current_concurrency=10,
                recommended_concurrency=15,
                action="SCALE_UP",
                bottleneck_reason="Headroom exists",
                max_throughput_achieved=200.0,
                p99_latency=1.2,
            )
        ]
        md = reporter.generate_markdown(summary, recommendations)
        assert "SCALE UP" in md
        assert "crawler" in md
        assert "200.0" in md

    def test_generate_json(self):
        reporter = ReportGenerator()
        summary = {"stages": {}}
        recommendations = [
            StageRecommendation(
                stage_name="test", current_concurrency=2, recommended_concurrency=4,
                action="SCALE_UP", bottleneck_reason="Slow", max_throughput_achieved=10.0,
            )
        ]
        data = reporter.generate_json(summary, recommendations)
        assert "summary" in data
        assert "recommendations" in data
        assert data["recommendations"][0]["stage_name"] == "test"
        assert data["recommendations"][0]["action"] == "SCALE_UP"

    def test_save_report_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator()
            summary = {"stages": {}, "total_duration_sec": 10.0, "peak_cpu_pct": 50.0, "avg_cpu_pct": 30.0, "peak_memory_mb": 256.0, "timestamp": "now"}
            recommendations = []
            out_path = reporter.save_report(
                summary, recommendations, output_dir=tmpdir, name="test_report"
            )
            md_path = out_path / "test_report.md"
            json_path = out_path / "test_report.json"
            assert md_path.exists()
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert data["summary"]["total_duration_sec"] == 10.0

    def test_empty_stages(self):
        reporter = ReportGenerator()
        reporter.generate_markdown({"stages": {}, "total_duration_sec": 0.0, "peak_cpu_pct": 0.0, "avg_cpu_pct": 0.0, "peak_memory_mb": 0.0, "timestamp": "now"}, [])
