#!/usr/bin/env python3
"""Soft-gate wrapper for mid-pipeline evaluation checks.

Wraps exploratory eval scripts so they can emit non-blocking warnings
locally while still failing the CI job when a soft-gate condition is met.

Usage:
    python scripts/eval_soft_gate.py --ci -- script.py [args...]
    python scripts/eval_soft_gate.py script.py [args...]

Exit codes:
    0 - wrapped script succeeded and no soft-gate condition triggered
    1 - wrapped script failed OR soft-gate condition triggered in CI mode
    2 - wrapper usage error
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOFT_GATE_PATTERNS = [
    re.compile(r"decision\s*=\s*keep", re.IGNORECASE),
    re.compile(r"CI\s+includes\s+0", re.IGNORECASE),
    re.compile(r"no\s+significant\s+delta", re.IGNORECASE),
    re.compile(r"warn", re.IGNORECASE),
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Soft-gate wrapper for eval scripts")
    ap.add_argument("--ci", action="store_true", help="Fail CI on soft-gate warnings")
    ap.add_argument("script", help="Path to eval script to wrap")
    ap.add_argument("script_args", nargs="*", help="Arguments passed to the wrapped script")
    return ap.parse_args()


def _run(script: Path, script_args: list[str], ci: bool) -> int:
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
    args = _parse_args()
    script = Path(args.script)
    if not script.is_absolute():
        script = PROJECT_ROOT / script
    return _run(script, args.script_args, ci=args.ci)


if __name__ == "__main__":
    raise SystemExit(main())
