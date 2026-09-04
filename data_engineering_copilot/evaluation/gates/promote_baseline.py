"""Atomic promotion of retrieval eval to baseline_recall_all with provenance sidecar.

Copies ``retrieval_eval.json`` → ``baseline_recall_all.json`` via tmp→rename
and writes ``baseline_recall_all.provenance.json`` with git_commit,
generation pinned-d3dbad402105, k, top_k, mrl, timestamp.

This module can be invoked as:
    python -m data_engineering_copilot.evaluation.gates.promote_baseline --source /tmp/new_baseline3/retrieval_eval.json
    python -m data_engineering_copilot.evaluation.gates.promote_baseline --check-gate
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path("/tmp/new_baseline3/retrieval_eval.json")
DEFAULT_DEST = PROJECT_ROOT / "tests/evaluation/benchmarks/baseline_recall_all.json"
DEFAULT_PROVENANCE = PROJECT_ROOT / "tests/evaluation/benchmarks/baseline_recall_all.provenance.json"
DEFAULT_BASELINE = DEFAULT_DEST
GENERATION = "pinned-d3dbad402105"
RELATIVE_TOLERANCE = 0.015  # 1.5% relative drop allowed


def _git_commit(short: bool = True) -> str:
    """Return the current git commit SHA, optionally shortened."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            return ""
        sha = proc.stdout.strip()
        return sha[:12] if short else sha
    except Exception:
        return ""


def _load_metrics(source: pathlib.Path) -> dict:
    """Load eval metrics from a JSON file."""
    data = json.loads(source.read_text(encoding="utf-8"))
    overall = data.get("overall") or {}
    per_intent = data.get("per_intent") or {}
    return {"overall": overall, "per_intent": per_intent, "raw": data}


def _recall_at_k(metrics: dict) -> float | None:
    """Extract recall@k value from loaded metrics."""
    overall = metrics.get("overall") or {}
    if "recall@k" in overall:
        return float(overall["recall@k"])
    raw = metrics.get("raw") or {}
    if "recall@k" in raw:
        return float(raw["recall@k"])
    return None


def check_gate(
    source: pathlib.Path = DEFAULT_SOURCE,
    baseline_path: pathlib.Path = DEFAULT_BASELINE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> int:
    """Validate candidate source against active baseline.

    Exits non-zero when overall recall@k regresses by more than
    ``relative_tolerance`` compared to the active baseline.
    """
    src = pathlib.Path(source)
    if not src.exists():
        print(f"❌ --check-gate: source missing {src}", file=sys.stderr)
        return 2

    candidate = _load_metrics(src)
    candidate_recall = _recall_at_k(candidate)
    if candidate_recall is None:
        print("❌ --check-gate: source missing overall recall@k", file=sys.stderr)
        return 2

    if not baseline_path.exists():
        print(f"⚠️ --check-gate: baseline missing {baseline_path}; skipping regression check")
        print(f"check-gate source exists: {src}")
        return 0

    baseline = _load_metrics(baseline_path)
    baseline_recall = _recall_at_k(baseline)
    if baseline_recall is None:
        print(f"⚠️ --check-gate: baseline missing recall@k at {baseline_path}; skipping regression check")
        print(f"check-gate source exists: {src}")
        return 0

    delta = candidate_recall - baseline_recall
    allowed_drop = baseline_recall * relative_tolerance
    floor = max(0.0, baseline_recall - allowed_drop)

    print(
        f"check-gate candidate_recall={candidate_recall:.6f} baseline_recall={baseline_recall:.6f} delta={delta:+.6f} floor={floor:.6f}"
    )

    if candidate_recall < floor:
        print(
            f"❌ --check-gate: recall regression exceeds tolerance: {candidate_recall:.6f} < {floor:.6f}",
            file=sys.stderr,
        )
        return 1

    print(f"✅ --check-gate passed: {src}")
    return 0


def promote(
    source: pathlib.Path = DEFAULT_SOURCE,
    dest: pathlib.Path = DEFAULT_DEST,
    provenance_path: pathlib.Path = DEFAULT_PROVENANCE,
    generation: str = GENERATION,
) -> int:
    """Promote a retrieval eval JSON to the canonical baseline path with provenance."""
    if not source.exists():
        print(f"❌ source not found: {source}", file=sys.stderr)
        return 2
    try:
        from data_engineering_copilot.config.settings import settings

        k = 10
        top_k = int(getattr(settings, "retrieval_top_k", 50))
        mrl = bool(getattr(settings, "mrl_multistage_enabled", False))
    except Exception:
        k, top_k, mrl = 10, 50, False

    data = json.loads(source.read_text(encoding="utf-8"))
    overall = data.get("overall") or {}
    if "recall@k" not in overall and "recall@k" not in data:
        print(f"⚠️ source missing overall.recall@k: {source}", file=sys.stderr)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(dest)
    print(f"✅ promoted {source} → {dest}")

    metrics = overall if overall else data
    commit = _git_commit(short=True)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    provenance = {
        "generated_at": now,
        "generator": "promote_baseline",
        "generation": generation,
        "git_commit": commit,
        "commit": commit,
        "k": k,
        "retrieval_top_k": top_k,
        "top_k": top_k,
        "mrl_multistage_enabled": mrl,
        "mrl": mrl,
        "metrics": metrics,
        "source": str(source),
    }
    prov_tmp = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    prov_tmp.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    prov_tmp.replace(provenance_path)
    print(f"✅ provenance → {provenance_path} (commit={commit} gen={generation} k={k} top_k={top_k} mrl={mrl})")
    return 0


def main() -> int:
    """CLI entrypoint for baseline promotion."""
    ap = argparse.ArgumentParser(description="Promote retrieval eval to baseline with provenance")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="path to retrieval_eval.json")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="destination baseline_recall_all.json")
    ap.add_argument("--provenance", default=str(DEFAULT_PROVENANCE), help="provenance sidecar path")
    ap.add_argument("--generation", default=GENERATION, help="pinned generation id")
    ap.add_argument("--check-gate", action="store_true", help="run gate check without promoting")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="active baseline path for regression check")
    ap.add_argument(
        "--relative-tolerance", type=float, default=RELATIVE_TOLERANCE, help="relative regression tolerance"
    )
    args = ap.parse_args()

    if args.check_gate:
        return check_gate(
            source=pathlib.Path(args.source),
            baseline_path=pathlib.Path(args.baseline),
            relative_tolerance=args.relative_tolerance,
        )

    return promote(
        source=pathlib.Path(args.source),
        dest=pathlib.Path(args.dest),
        provenance_path=pathlib.Path(args.provenance),
        generation=args.generation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
