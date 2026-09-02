#!/usr/bin/env python3
"""Wrapper for hermetic chunking evaluation (no Qdrant, no LLM).

Runs ``dec eval-chunking --strategy all --gold all --output /tmp/chunking_eval.json``
via :func:`data_engineering_copilot.evaluation.chunking_eval.run_chunking_eval`.

Hermetic: only tokenizer + gold spans on disk, ~30s.  Output is a JSON report
``{strategy: {iou, precision, boundary_similarity, fracture_rate, doc_count}}``
plus ``gates``.

Usage:
    dec_venv/bin/python scripts/run_chunking_eval.py --strategy all --gold all --output /tmp/chunking_eval.json
    dec_venv/bin/dec eval-chunking --strategy all --gold all --output /tmp/chunking_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run hermetic chunking evaluation")
    ap.add_argument(
        "--strategy",
        choices=["all", "recursive", "sentence", "header", "structured"],
        default="all",
        help="Chunking strategy to evaluate (default: all).",
    )
    ap.add_argument(
        "--gold",
        choices=["synthetic", "human", "all"],
        default="all",
        help="Gold dataset source (default: all).",
    )
    ap.add_argument(
        "--output",
        default="/tmp/chunking_eval.json",
        help="Output JSON path (default: /tmp/chunking_eval.json).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from data_engineering_copilot.evaluation.chunking_eval import run_chunking_eval

    try:
        report = run_chunking_eval(args.strategy, args.gold, args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"Chunking evaluation failed: {exc}", file=sys.stderr)
        return 2

    # Pretty-print like cli.py eval_chunking_main
    print(f"{'Strategy':<15} {'IoU':>6} {'Prec':>6} {'B-Sim':>6} {'Fract':>6}")
    gates = report.get("gates") or {}
    for strat, m in report.items():
        if not isinstance(m, dict) or "iou" not in m:
            continue
        print(
            f"{strat:<15} {m['iou']:>6.3f} {m['precision']:>6.3f} "
            f"{m['boundary_similarity']:>6.3f} {m['fracture_rate']:>6.3f}"
        )
    if gates:
        verdict = "PASS" if gates.get("fracture_ok") else "FAIL"
        print(
            f"\n{verdict} fracture gate: worst={gates.get('worst_fracture_rate', 0):.3f} "
            f"threshold<={gates.get('fracture_threshold', 0):.2f}"
        )
        if not gates.get("fracture_ok"):
            return 1
    print(f"\nReport written to {args.output}")
    # Also dump full JSON for tail inspection
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
