"""Drift-triggered re-evaluation hook.

Reads a drift alert JSON from stdin or a file path argument, writes a
machine-readable alert to ``/tmp/drift_alert.json``, and optionally schedules
background re-evaluation jobs.

This module can be invoked as:
    python -m data_engineering_copilot.evaluation.gates.drift_hook --alert /tmp/drift_alert.json
    cat /tmp/drift_alert.json | python -m data_engineering_copilot.evaluation.gates.drift_hook
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ALERT_PATH = Path("/tmp/drift_alert.json")


def _emit_alert(alert: dict) -> None:
    """Write drift alert JSON to a well-known path."""
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_PATH.write_text(json.dumps(alert, indent=2, default=str), encoding="utf-8")
    print(f"drift_hook: alert written to {ALERT_PATH}")


def _schedule_reeval(alert: dict) -> None:
    """Schedule background re-evaluation jobs for drifted metrics."""
    drifted_metrics = alert.get("drifted_metrics", [])
    if not drifted_metrics:
        return
    cmd = (
        "setsid dec_venv/bin/dec eval-fast > /tmp/drift_reeval_eval_fast.log 2>&1 < /dev/null & "
        "setsid dec_venv/bin/dec eval-retrieval --k 10 > /tmp/drift_reeval_eval_retrieval.log 2>&1 < /dev/null &"
    )
    try:
        subprocess.Popen(cmd, shell=True, start_new_session=True)
        print(f"drift_hook: scheduled background re-eval for metrics: {', '.join(drifted_metrics)}")
    except Exception as exc:
        print(f"drift_hook: failed to schedule re-eval: {exc}")


def _load_alert(path: Path | None) -> dict:
    """Load alert JSON from file or stdin."""
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not sys.stdin.isatty():
        return json.loads(sys.stdin.read())
    return {}


def main() -> int:
    """CLI entrypoint for drift gate hook."""
    ap = argparse.ArgumentParser(description="Drift gate hook")
    ap.add_argument("--alert", type=Path, help="Path to drift alert JSON")
    ap.add_argument("--no-reeval", action="store_true", help="Skip background re-evaluation scheduling")
    args = ap.parse_args()

    alert = _load_alert(args.alert)
    if not alert:
        print("drift_hook: no alert provided", file=sys.stderr)
        return 2

    _emit_alert(alert)
    if not args.no_reeval and alert.get("drifted"):
        _schedule_reeval(alert)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
