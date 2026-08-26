"""Unit tests for drift detection service."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from data_engineering_copilot.services.drift_detector import (
    DriftDetector,
    DriftReport,
    DriftResult,
    EvalSnapshot,
    hash_eval_dataset,
)


class TestEvalSnapshot:
    def test_creation(self) -> None:
        snap = EvalSnapshot(timestamp="2026-07-29T00:00:00Z", metrics={"faithfulness": 0.85})
        assert snap.timestamp == "2026-07-29T00:00:00Z"
        assert snap.metrics["faithfulness"] == 0.85

    def test_provenance_defaults(self) -> None:
        snap = EvalSnapshot(timestamp="t", metrics={})
        assert snap.git_commit == ""
        assert snap.generation == ""
        assert snap.config_fingerprint == ""

    def test_provenance_fields_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            detector = DriftDetector(storage_path=path)
            snap = EvalSnapshot(
                timestamp="2026-07-29T00:00:00Z",
                metrics={"confidence": 0.9},
                eval_dataset_hash="abc123",
                git_commit="deadbeefcafe",
                generation="pinned-test-123",
                embedding_model="test-embedder",
                reranker="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
                chunk_size=375,
                chunk_overlap=90,
                retrieval_top_k=50,
                config_fingerprint="0123456789abcdef",
            )
            detector.record(snap)
            loaded = detector.load_history()
            assert len(loaded) == 1
            restored = loaded[0]
            assert restored.git_commit == "deadbeefcafe"
            assert restored.generation == "pinned-test-123"
            assert restored.embedding_model == "test-embedder"
            assert restored.reranker == "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
            assert restored.chunk_size == 375
            assert restored.chunk_overlap == 90
            assert restored.retrieval_top_k == 50
            assert restored.config_fingerprint == "0123456789abcdef"


class TestDriftDetectorRecord:
    def test_record_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            detector = DriftDetector(storage_path=path)
            snap = EvalSnapshot(timestamp="2026-07-29T00:00:00Z", metrics={"confidence": 0.9})
            detector.record(snap)
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["metrics"]["confidence"] == 0.9

    def test_record_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            detector = DriftDetector(storage_path=path)
            for i in range(3):
                detector.record(EvalSnapshot(timestamp=f"2026-07-2{i}T00:00:00Z", metrics={"f": 0.8}))
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 3


class TestDriftDetectorLoadHistory:
    def test_load_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = DriftDetector(storage_path=Path(tmpdir) / "missing.jsonl")
            assert detector.load_history() == []

    def test_load_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            path.write_text(json.dumps({"timestamp": "2026-07-29T00:00:00Z", "metrics": {"f": 0.9}}) + "\n")
            detector = DriftDetector(storage_path=path)
            history = detector.load_history()
            assert len(history) == 1
            assert history[0].metrics["f"] == 0.9

    def test_load_skips_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            path.write_text(
                "not json\n" + json.dumps({"timestamp": "2026-07-29T00:00:00Z", "metrics": {"f": 0.9}}) + "\n"
            )
            detector = DriftDetector(storage_path=path)
            assert len(detector.load_history()) == 1


class TestDriftDetectorBaseline:
    def test_no_baseline_first_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = DriftDetector(storage_path=Path(tmpdir) / "history.jsonl")
            assert detector.get_baseline() is None

    def test_baseline_averages_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            detector = DriftDetector(storage_path=path, window_days=30)
            now = datetime.now(UTC)
            for i, val in enumerate([0.8, 0.9, 1.0]):
                ts = (now - timedelta(days=2 - i)).isoformat()
                detector.record(
                    EvalSnapshot(
                        timestamp=ts,
                        metrics={"faithfulness": val},
                    )
                )
            baseline = detector.get_baseline()
            assert baseline is not None
            assert abs(baseline.metrics["faithfulness"] - 0.9) < 0.01


class TestDriftDetectorCompare:
    def test_no_drift_when_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            detector = DriftDetector(storage_path=path, window_days=30)
            now = datetime.now(UTC)
            detector.record(EvalSnapshot(timestamp=(now - timedelta(days=1)).isoformat(), metrics={"confidence": 0.9}))
            current = EvalSnapshot(timestamp=now.isoformat(), metrics={"confidence": 0.88})
            report = detector.compare(current)
            assert not report.drifted

    def test_drift_when_drop_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            detector = DriftDetector(
                storage_path=path,
                window_days=30,
                thresholds={"confidence": 0.1},
            )
            now = datetime.now(UTC)
            detector.record(EvalSnapshot(timestamp=(now - timedelta(days=1)).isoformat(), metrics={"confidence": 0.9}))
            current = EvalSnapshot(timestamp=now.isoformat(), metrics={"confidence": 0.7})
            report = detector.compare(current)
            assert report.drifted
            assert "confidence" in report.drifted_metrics

    def test_no_comparison_when_no_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = DriftDetector(storage_path=Path(tmpdir) / "history.jsonl")
            current = EvalSnapshot(timestamp="2026-07-29T00:00:00Z", metrics={"confidence": 0.5})
            report = detector.compare(current)
            assert not report.drifted
            assert report.comparisons == []


class TestDriftReport:
    def test_drifted_property(self) -> None:
        report = DriftReport(
            snapshot=EvalSnapshot(timestamp="t", metrics={}),
            comparisons=[
                DriftResult(metric="a", baseline=0.9, current=0.85, delta=-0.05, threshold=0.1, drifted=False),
                DriftResult(metric="b", baseline=0.9, current=0.7, delta=-0.2, threshold=0.1, drifted=True),
            ],
        )
        assert report.drifted
        assert report.drifted_metrics == ["b"]


class TestHashEvalDataset:
    def test_deterministic(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"q": "test"}\n')
            f.flush()
            h1 = hash_eval_dataset(f.name)
            h2 = hash_eval_dataset(f.name)
            assert h1 == h2
            assert len(h1) == 16
