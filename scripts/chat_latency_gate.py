#!/usr/bin/env python3
"""Live-stack chat latency gate.

Measures one warm chat turn against /api/v1/chat and fails when the total turn
or the rerank stage exceeds configured budgets. Requires the stack up.

Usage:
    dec_venv/bin/python scripts/chat_latency_gate.py \
        [--base-url http://localhost:8000] \
        [--budget-total 60] [--budget-rerank 20]
Env overrides: DEC_CHAT_BUDGET_TOTAL, DEC_CHAT_BUDGET_RERANK.
"""

import argparse
import json
import os
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("DEC_CHAT_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--budget-total", type=float, default=float(os.environ.get("DEC_CHAT_BUDGET_TOTAL", "60")))
    ap.add_argument("--budget-rerank", type=float, default=float(os.environ.get("DEC_CHAT_BUDGET_RERANK", "20")))
    ap.add_argument("--question", default="What is a data pipeline?")
    args = ap.parse_args()

    started = time.monotonic()
    stage_durations: dict[str, float] = {}
    first_token_at = None
    total: float | None = None

    timeout = httpx.Timeout(connect=5, read=args.budget_total + 30, write=5, pool=5)
    with (
        httpx.Client(timeout=timeout) as client,
        client.stream("POST", f"{args.base_url}/api/v1/chat", json={"message": args.question, "stream": True}) as resp,
    ):
        resp.raise_for_status()
        last_event_ts = started
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            now = time.monotonic()
            etype = ev.get("type")
            if etype == "status":
                stage_durations[ev.get("message", "?")] = now - last_event_ts
            elif etype == "token":
                if first_token_at is None:
                    first_token_at = now
            elif etype == "done":
                total = now - started
                if "stage_times_seconds" in ev:
                    stage_durations.setdefault("stage_times_seconds", 0.0)
            elif etype == "error":
                print(json.dumps({"ok": False, "error": ev.get("message")}))
                return 1
            last_event_ts = now

    total = total or (time.monotonic() - started)
    rerank = stage_durations.get("Reranking", 0.0)
    report = {
        "ok": total <= args.budget_total and rerank <= args.budget_rerank,
        "total_seconds": round(total, 2),
        "rerank_seconds": round(rerank, 2),
        "first_token_seconds": round(first_token_at - started, 2) if first_token_at else None,
        "budget_total": args.budget_total,
        "budget_rerank": args.budget_rerank,
        "stages": {k: round(v, 2) for k, v in stage_durations.items() if v > 0},
    }
    print(json.dumps(report, indent=2))
    if total > args.budget_total:
        print(f"FAIL: total {total:.1f}s > budget {args.budget_total:g}s")
    if rerank > args.budget_rerank:
        print(f"FAIL: rerank {rerank:.1f}s > budget {args.budget_rerank:g}s")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
