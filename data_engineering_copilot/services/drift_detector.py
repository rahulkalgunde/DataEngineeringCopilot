"""Automated drift detection for RAG evaluation metrics.

Stores evaluation snapshots in a JSONL history file and compares
subsequent runs against a baseline to detect metric regression.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.8,
    "context_recall": 0.7,
    "context_precision": 0.6,
    "answer_relevancy": 0.7,
    "overall": 0.7,
    "confidence": 0.5,
}


@dataclass(frozen=True)
class EvalSnapshot:
    """A single evaluation run's metric results."""

    timestamp: str
    metrics: dict[str, float]
    eval_dataset_hash: str = ""
    git_commit: str = ""
    generation: str = ""
    embedding_model: str = ""
    reranker: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    retrieval_top_k: int = 0
    config_fingerprint: str = ""


@dataclass(frozen=True)
class DriftResult:
    """Comparison of a single metric against its baseline."""

    metric: str
    baseline: float
    current: float
    delta: float
    threshold: float
    drifted: bool


@dataclass
class DriftReport:
    """Full drift comparison report for an evaluation run."""

    snapshot: EvalSnapshot
    comparisons: list[DriftResult] = field(default_factory=list)

    @property
    def drifted(self) -> bool:
        return any(c.drifted for c in self.comparisons)

    @property
    def drifted_metrics(self) -> list[str]:
        return [c.metric for c in self.comparisons if c.drifted]


class DriftDetector:
    """Compare current eval metrics against stored baselines.

    History is stored as a JSONL file with one ``EvalSnapshot`` per line.
    Baseline is the average of metrics over the last ``window_days`` days.
    """

    def __init__(
        self,
        storage_path: str | Path = "data/eval_history.jsonl",
        thresholds: dict[str, float] | None = None,
        window_days: int = 7,
    ) -> None:
        self._storage_path = Path(storage_path)
        self._thresholds = thresholds or dict(_DEFAULT_THRESHOLDS)
        self._window_days = window_days

    def _ensure_dir(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, snapshot: EvalSnapshot) -> None:
        """Append a snapshot to the history file and trigger drift hook if needed."""
        self._ensure_dir()
        with open(self._storage_path, "a") as f:
            f.write(json.dumps(asdict(snapshot)) + "\n")
        logger.info("eval_snapshot_recorded metrics=%s", list(snapshot.metrics.keys()))
        try:
            report = self.compare(snapshot)
            if report.drifted:
                self._trigger_drift_hook(report)
        except Exception as exc:
            logger.warning("drift hook failed: %s", exc)

    def _trigger_drift_hook(self, report: DriftReport) -> None:
        try:
            payload = {
                "drifted": report.drifted,
                "drifted_metrics": report.drifted_metrics,
                "snapshot": asdict(report.snapshot),
                "comparisons": [asdict(c) for c in report.comparisons],
            }
            hook = PROJECT_ROOT / "data_engineering_copilot" / "evaluation" / "gates" / "drift_hook.py"
            if hook.exists():
                import subprocess

                subprocess.run(
                    [sys.executable, str(hook)],
                    input=json.dumps(payload, default=str),
                    text=True,
                    start_new_session=True,
                )
                logger.info("drift_hook_triggered metrics=%s", report.drifted_metrics)
        except Exception as exc:
            logger.warning("drift hook unavailable: %s", exc)

    def load_history(self) -> list[EvalSnapshot]:
        """Load all snapshots from the history file."""
        if not self._storage_path.exists():
            return []
        snapshots: list[EvalSnapshot] = []
        with open(self._storage_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    snapshots.append(EvalSnapshot(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return snapshots

    def get_baseline(self) -> EvalSnapshot | None:
        """Compute baseline as average of snapshots within the time window."""
        cutoff = datetime.now(UTC) - timedelta(days=self._window_days)
        history = self.load_history()

        recent: list[EvalSnapshot] = []
        for snap in history:
            try:
                ts = datetime.fromisoformat(snap.timestamp.replace("Z", "+00:00"))
                if ts >= cutoff:
                    recent.append(snap)
            except (ValueError, TypeError):
                continue

        if not recent:
            return None

        # Average metrics across recent snapshots
        all_keys: set[str] = set()
        for s in recent:
            all_keys.update(s.metrics.keys())

        avg_metrics: dict[str, float] = {}
        for key in all_keys:
            values = [s.metrics[key] for s in recent if key in s.metrics]
            avg_metrics[key] = sum(values) / len(values) if values else 0.0

        return EvalSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            metrics=avg_metrics,
            eval_dataset_hash="baseline",
        )

    def compare(self, current: EvalSnapshot) -> DriftReport:
        """Compare current snapshot against the stored baseline.

        Returns a ``DriftReport`` with per-metric drift results.
        """
        baseline = self.get_baseline()
        report = DriftReport(snapshot=current)

        if baseline is None:
            logger.info("No baseline available — first run, skipping drift comparison")
            return report

        for metric, current_val in current.metrics.items():
            baseline_val = baseline.metrics.get(metric)
            if baseline_val is None:
                continue
            threshold = self._thresholds.get(metric, 0.1)
            delta = current_val - baseline_val
            drifted = delta < -threshold
            report.comparisons.append(
                DriftResult(
                    metric=metric,
                    baseline=round(baseline_val, 4),
                    current=round(current_val, 4),
                    delta=round(delta, 4),
                    threshold=threshold,
                    drifted=drifted,
                )
            )

        return report


def hash_eval_dataset(path: str | Path) -> str:
    """Return SHA-256 hash of an evaluation dataset file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
