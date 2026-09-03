"""Soft-gate wrapper for mid-pipeline evaluation checks.

Wraps exploratory eval scripts so they can emit non-blocking warnings
locally while still failing the CI job when a soft-gate condition is met.

This module can be invoked as:
    python -m data_engineering_copilot.evaluation.gates.soft_gate --ci -- <script> [args...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SOFT_GATE_PATTERNS = [
    re.compile(r"decision\s*=\s*keep", re.IGNORECASE),
    re.compile(r"CI\s+includes\s+0", re.IGNORECASE),
    re.compile(r"no\s+significant\s+delta", re.IGNORECASE),
    re.compile(r"warn", re.IGNORECASE),
]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Soft-gate wrapper for eval scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m data_engineering_copilot.evaluation.gates.soft_gate --ci data_engineering_copilot/evaluation/gates/pipeline_ablation.py --k 10 --split held\n"
            "  python -m data_engineering_copilot.evaluation.gates.soft_gate scripts/tune_rrf_k.py --k 10\n"
        ),
    )
    parser.add_argument("--ci", action="store_true", help="Fail CI on soft-gate warnings")
    parser.add_argument("script", type=Path, help="Path to eval script to wrap")
    parser.add_argument("script_args", nargs="*", help="Arguments passed to the wrapped script")
    return parser.parse_args()


def _run(script: Path, script_args: list[str], ci: bool) -> int:
    """Run the wrapped script and scan output for soft-gate conditions."""
    if not script.exists():
        print(f"❌ soft-gate: script not found: {script}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(script), *script_args]
    print(f"soft-gate: running {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=False)

    soft_gate_triggered = False
    if proc.stdout:
        for line in proc.stdout.splitlines():
            for pat in SOFT_GATE_PATTERNS:
                if pat.search(line):
                    soft_gate_triggered = True
                    if ci:
                        print(f"❌ soft-gate CI failure: {line.strip()}", file=sys.stderr)
                    else:
                        print(f"⚠️  soft-gate warning: {line.strip()}")
                    break

    if proc.returncode != 0:
        print(f"❌ soft-gate: wrapped script exited {proc.returncode}", file=sys.stderr)
        return 1

    if ci and soft_gate_triggered:
        return 1

    return 0


def main() -> int:
    """CLI entrypoint for soft-gate wrapper."""
    args = _parse_args()
    script = args.script
    if not script.is_absolute():
        script = Path.cwd() / script
    return _run(script, args.script_args, ci=args.ci)


if __name__ == "__main__":
    raise SystemExit(main())
