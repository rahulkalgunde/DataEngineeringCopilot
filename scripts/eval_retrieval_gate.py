#!/usr/bin/env python3
"""Thin wrapper for ablation / gate invocation.

Supports ``--ablation`` to run the dense/sparse/hybrid ablation harness
with holdout split (110/110 seed=42) and bootstrap CI, mirroring
``dec eval-retrieval --ablation``. Without ``--ablation`` it forwards to
the standard retrieval gate logic.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _run_ablation(args: argparse.Namespace) -> int:
    # Delegate to cli's ablation harness for single-source truth
    cmd = [
        "dec_venv/bin/dec",
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

    # Non-ablation gate: forward to dec eval-retrieval --compare-baseline
    cmd = ["dec_venv/bin/dec", "eval-retrieval", "--k", str(args.k)]
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
