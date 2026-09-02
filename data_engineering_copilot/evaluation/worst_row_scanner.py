"""Worst-row scanner for majority-judge evaluation results.

Loads evaluation results from ``dec evaluate --output-dir`` (JSONL format with
faithfulness/relevance/rubric scores) and golden QA files (with ``source_name``),
computes per-platform metrics, and surfaces the worst-performing rows by platform.

No LLM calls — purely offline analysis of existing artifacts.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_WORST_N = 3
METRIC_THRESHOLDS = {
    "faithfulness": 0.85,
    "relevance": 0.80,
    "rubric": 4.0,
}


class ScanReport(TypedDict):
    platforms: dict[str, PlatformMetrics]
    worst_rows: dict[str, list[EvalRow]]
    total_rows: int
    total_platforms: int


@dataclass
class EvalRow:
    """One evaluated row with scores and metadata."""

    id: str
    faithfulness: float
    relevance: float
    rubric: float
    source_name: str | None = None
    question: str = ""
    answer: str = ""
    judge_votes: list[dict] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        norm_rubric = self.rubric / 5.0
        return (self.faithfulness + self.relevance + norm_rubric) / 3.0

    def below_threshold(self) -> list[str]:
        flags = []
        if self.faithfulness < METRIC_THRESHOLDS["faithfulness"]:
            flags.append("faithfulness")
        if self.relevance < METRIC_THRESHOLDS["relevance"]:
            flags.append("relevance")
        if self.rubric < METRIC_THRESHOLDS["rubric"]:
            flags.append("rubric")
        return flags


@dataclass
class PlatformMetrics:
    """Aggregated metrics for one platform/source."""

    source_name: str
    rows: list[EvalRow]
    faithfulness_mean: float
    relevance_mean: float
    rubric_mean: float
    composite_mean: float
    total: int


def load_golden_qa(golden_path: str | Path) -> dict[str, str]:
    """Load golden QA file, return {id: source_name} mapping."""
    source_map: dict[str, str] = {}
    with open(golden_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            row_id = row.get("id")
            source_name = row.get("source_name")
            if row_id and source_name:
                source_map[row_id] = source_name
    return source_map


def load_evaluation_results(results_path: str | Path) -> list[EvalRow]:
    """Load evaluation results JSONL into EvalRow objects."""
    rows: list[EvalRow] = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            rows.append(
                EvalRow(
                    id=d.get("id", ""),
                    faithfulness=float(d.get("faithfulness", 0.0)),
                    relevance=float(d.get("relevance", 0.0)),
                    rubric=float(d.get("rubric", 0.0)),
                    question=d.get("question", ""),
                    answer=d.get("answer", ""),
                    judge_votes=d.get("judge_votes", []),
                )
            )
    return rows


def enrich_with_source(rows: list[EvalRow], source_map: dict[str, str]) -> None:
    """Attach source_name to each row in-place using the golden QA map."""
    for row in rows:
        if row.source_name is None:
            row.source_name = source_map.get(row.id, "unknown")


def compute_platform_metrics(rows: list[EvalRow]) -> dict[str, PlatformMetrics]:
    """Group rows by source_name and compute per-platform aggregate metrics."""
    by_source: dict[str, list[EvalRow]] = {}
    for row in rows:
        key = row.source_name or "unknown"
        by_source.setdefault(key, []).append(row)

    results: dict[str, PlatformMetrics] = {}
    for source_name, source_rows in by_source.items():
        faithfulness_vals = [r.faithfulness for r in source_rows]
        relevance_vals = [r.relevance for r in source_rows]
        rubric_vals = [r.rubric for r in source_rows]
        composite_vals = [r.composite_score for r in source_rows]
        results[source_name] = PlatformMetrics(
            source_name=source_name,
            rows=source_rows,
            faithfulness_mean=statistics.fmean(faithfulness_vals) if faithfulness_vals else 0.0,
            relevance_mean=statistics.fmean(relevance_vals) if relevance_vals else 0.0,
            rubric_mean=statistics.fmean(rubric_vals) if rubric_vals else 0.0,
            composite_mean=statistics.fmean(composite_vals) if composite_vals else 0.0,
            total=len(source_rows),
        )
    return results


def find_worst_rows(metrics: dict[str, PlatformMetrics], n: int = DEFAULT_WORST_N) -> dict[str, list[EvalRow]]:
    """Return bottom-n rows by composite score per platform."""
    worst: dict[str, list[EvalRow]] = {}
    for source_name, pm in metrics.items():
        sorted_rows = sorted(pm.rows, key=lambda r: r.composite_score)
        worst[source_name] = sorted_rows[:n]
    return worst


def format_worst_row(row: EvalRow) -> str:
    """Format one worst row as a readable recommendation line."""
    flags = row.below_threshold()
    flag_str = f" [{', '.join(flags)} below threshold]" if flags else ""
    snippet = row.question[:80] + ("..." if len(row.question) > 80 else "")
    return (
        f"  id={row.id} F={row.faithfulness:.3f} R={row.relevance:.3f} "
        f"Rb={row.rubric:.2f} (composite={row.composite_score:.3f}){flag_str}\n"
        f"    Q: {snippet}"
    )


def format_platform_summary(pm: PlatformMetrics) -> str:
    """Format one platform's aggregate metrics."""
    f_ok = "✓" if pm.faithfulness_mean >= METRIC_THRESHOLDS["faithfulness"] else "✗"
    r_ok = "✓" if pm.relevance_mean >= METRIC_THRESHOLDS["relevance"] else "✗"
    rb_ok = "✓" if pm.rubric_mean >= METRIC_THRESHOLDS["rubric"] else "✗"
    return (
        f"[{pm.source_name}] n={pm.total}  "
        f"F={pm.faithfulness_mean:.3f}{f_ok}  "
        f"R={pm.relevance_mean:.3f}{r_ok}  "
        f"Rb={pm.rubric_mean:.2f}{rb_ok}  "
        f"composite={pm.composite_mean:.3f}"
    )


def scan_worst_rows(
    results_path: str | Path,
    golden_paths: Sequence[str | Path],
    n: int = DEFAULT_WORST_N,
) -> ScanReport:
    """Main entry point: load results + golden QA, compute metrics, return report dict.

    Args:
        results_path: Path to evaluation results JSONL (from ``dec evaluate --output-dir``).
        golden_paths: One or more golden QA files (e.g. ``qa_spark.gt.jsonl``).
        n: Number of worst rows to surface per platform.

    Returns:
        A ScanReport with ``platforms`` (dict of PlatformMetrics keyed by source_name) and
        ``worst_rows`` (dict of worst row lists keyed by source_name).
    """
    combined_source_map: dict[str, str] = {}
    for gp in golden_paths:
        combined_source_map.update(load_golden_qa(gp))

    rows = load_evaluation_results(results_path)
    enrich_with_source(rows, combined_source_map)

    metrics = compute_platform_metrics(rows)
    worst = find_worst_rows(metrics, n=n)

    return {
        "platforms": metrics,
        "worst_rows": worst,
        "total_rows": len(rows),
        "total_platforms": len(metrics),
    }


def print_report(report: ScanReport) -> None:
    """Print a human-readable scan report to stdout."""
    metrics = report["platforms"]
    worst = report["worst_rows"]

    print(f"\n=== Worst-Row Scan Report (total={report['total_rows']}, platforms={report['total_platforms']}) ===\n")

    for source_name in sorted(metrics.keys()):
        pm = metrics[source_name]
        print(format_platform_summary(pm))
        wrows = worst.get(source_name, [])
        if wrows:
            print(f"  Worst {len(wrows)} rows:")
            for row in wrows:
                print(format_worst_row(row))
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan evaluation results for worst-performing rows by platform.")
    parser.add_argument("results", help="Path to evaluation results JSONL")
    parser.add_argument("golden", nargs="+", help="One or more golden QA JSONL files")
    parser.add_argument(
        "--n", type=int, default=DEFAULT_WORST_N, help=f"Number of worst rows per platform (default {DEFAULT_WORST_N})"
    )
    args = parser.parse_args()

    report = scan_worst_rows(args.results, args.golden, n=args.n)
    print_report(report)
