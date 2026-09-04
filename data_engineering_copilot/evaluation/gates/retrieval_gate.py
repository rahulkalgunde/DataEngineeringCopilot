"""Thin wrapper for ablation / gate invocation.

Supports ``--ablation`` to run the dense/sparse/hybrid ablation harness
with holdout split (110/110 seed=42) and bootstrap CI, mirroring
``dec eval-retrieval --ablation``. Without ``--ablation`` it forwards to
the standard retrieval gate logic.

This module can be invoked as:
    python -m data_engineering_copilot.evaluation.gates.retrieval_gate --ablation --k 10 --split held
    python -m data_engineering_copilot.evaluation.gates.retrieval_gate --k 10 --dataset tests/evaluation/golden/recall_all.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run_ablation(args: argparse.Namespace) -> int:
    """Delegate to CLI's ablation harness for single-source truth."""
    cmd = [
        sys.executable,
        "-m",
        "data_engineering_copilot.cli",
        "eval-retrieval",
        "--ablation",
        "--k",
        str(args.k),
        "--split",
        args.split,
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    """CLI entrypoint for retrieval gate / ablation harness."""
    parser = argparse.ArgumentParser(description="Retrieval gate / ablation harness")
    parser.add_argument("--dataset", default=None, help="Recall JSONL dataset")
    parser.add_argument("--k", type=int, default=10, help="cutoff k")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compare-baseline", default=None)
    parser.add_argument(
        "--ablation", action="store_true", help="run ablation harness (dense/sparse/hybrid + holdout bootstrap)"
    )
    parser.add_argument("--split", choices=["train", "held", "all"], default="all", help="holdout split seed=42")
    args = parser.parse_args()

    if args.ablation:
        return _run_ablation(args)

    cmd = [
        sys.executable,
        "-m",
        "data_engineering_copilot.cli",
        "eval-retrieval",
        "--k",
        str(args.k),
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.compare_baseline:
        cmd.extend(["--compare-baseline", args.compare_baseline])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
