"""Pre-flight cost estimates for paid eval commands.

Pure estimation helpers; printing/enforcement lives in cli.py. Estimates are
upper-bound heuristics documented in docs/EVALUATION_GUIDE.md (cost map).
"""

from __future__ import annotations

import os
import sys


def estimate_calls(
    command: str,
    n_rows: int,
    *,
    n_trials: int = 3,
    ragas: bool = False,
    spark: bool = False,
) -> int:
    """Estimate paid LLM calls for an eval run (upper-bound heuristic)."""
    if command == "eval-generation":
        return int(n_rows) * (1 + max(1, int(n_trials)) * 2)
    if command == "evaluate":
        if spark:
            return 0  # recall rows short-circuit generation entirely
        return int(n_rows) * (2 + (19 if ragas else 0))
    return 0


def enforce_cost_gate(command: str, estimate: int) -> None:
    """Non-TTY shells must set FORCE=1 to run paid eval commands."""
    if estimate <= 0:
        return
    if sys.stdin.isatty():
        return
    if os.environ.get("FORCE", "").strip() == "1":
        return
    print(f"❌ {command} would make ~{estimate} paid LLM calls. Re-run with FORCE=1 for non-interactive shells.")
    raise SystemExit(2)
