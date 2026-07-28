from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_INTERVAL = 30

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED"}

STATUS_ICONS = {
    "DISPATCHED": "[ ]",
    "PROCESSING": "[>]",
    "COMPLETED": "[OK]",
    "FAILED": "[!!]",
    "CANCELLED": "[--]",
}

W = 160


def fetch_status(api_url: str, task_id: str | None = None) -> dict | None:
    url = f"{api_url}/api/v1/ingest/status/{task_id}" if task_id else f"{api_url}/api/v1/ingest/latest"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except (ConnectionRefusedError, TimeoutError, OSError):
        return None


def _fmt_delta(value: int) -> str:
    return f"+{value:,}" if value > 0 else str(value)


def _fmt_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_dashboard(
    current: dict,
    prev: dict | None,
    poll_ts: float,
    first_poll_ts: float,
) -> None:
    os.system("clear" if os.name != "nt" else "cls")

    status = current.get("status", "UNKNOWN")
    icon = STATUS_ICONS.get(status, "[?]")
    task_id = current.get("task_id", "N/A")[:8]
    elapsed = _fmt_elapsed(time.time() - first_poll_ts)
    sources = ", ".join(current.get("source_names", []))

    pages = current.get("pages_fetched", 0)
    chunks = current.get("chunks_indexed", 0)
    skipped = current.get("pages_skipped", 0)
    error_count = len([s for s in current.get("source_stats", {}).values() if s.get("errors", 0) > 0])
    current_url = current.get("current_url", "")
    error_msg = current.get("error")

    interval = poll_ts - (prev.get("_poll_ts", first_poll_ts) if prev else first_poll_ts)
    if prev and interval > 0:
        dp = pages - prev.get("pages_fetched", 0)
        dc = chunks - prev.get("chunks_indexed", 0)
        ds = skipped - prev.get("pages_skipped", 0)
        pages_per_s = dp / interval
        chunks_per_s = dc / interval
    else:
        dp = dc = ds = 0
        pages_per_s = chunks_per_s = 0.0

    print("=" * W)
    print("  INGESTION MONITOR")
    print("=" * W)
    print()
    print(f"  {icon} Status:  {status:<12}  Task: {task_id}  Elapsed: {elapsed}")
    if sources:
        print(f"  Sources: {sources}")
    print()

    print("  Metric          Count       Delta (30s)    Rate")
    print("  " + "-" * (W - 4))
    print(f"  Pages fetched   {pages:>10,}   {_fmt_delta(dp):>10}   {pages_per_s:>7.1f} p/s")
    print(f"  Chunks indexed  {chunks:>10,}   {_fmt_delta(dc):>10}   {chunks_per_s:>7.1f} c/s")
    print(f"  Pages skipped   {skipped:>10,}   {_fmt_delta(ds):>10}")
    print(f"  Errors          {error_count:>10,}")
    print()

    if current_url:
        print(f"  Crawling: {current_url}")
    print()

    source_stats = current.get("source_stats", {})
    if source_stats:
        print("  Per-Source")
        print("  " + "-" * (W - 4))
        hdr = f"  {'Source':<32} {'Pages':>9} {'Chunks':>9} {'Errors':>7}   Current URL"
        print(hdr)
        for name, stats in source_stats.items():
            sp = stats.get("pages_fetched", 0)
            sc = stats.get("chunks_indexed", 0)
            se = stats.get("errors", 0)
            cu = stats.get("current_url", "")
            print(f"  {name:<32} {sp:>9,} {sc:>9,} {se:>7}   {cu}")
        print()

    events = current.get("recent_events", [])
    if events:
        print("  Recent Events")
        print("  " + "-" * (W - 4))
        for ev in events[-10:]:
            ts = ev.get("ts", 0)
            t = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "??:??:??"
            etype = ev.get("type", "?")
            url = ev.get("url", "")
            cd = ev.get("chunks", 0) or 0
            err = ev.get("error", "")
            if err:
                print(f"  {t}  {etype:<22} ERR: {err}")
            elif cd:
                print(f"  {t}  {etype:<22} {url}  +{cd} ch")
            else:
                print(f"  {t}  {etype:<22} {url}")
        print()

    if error_msg:
        print("  !!! ERROR !!!")
        print(f"  {error_msg}")
        print()

    print("=" * W)
    if status in TERMINAL_STATES:
        if status == "COMPLETED":
            print(f"  DONE \u2014 {pages:,} pages, {chunks:,} chunks indexed")
        elif status == "FAILED":
            print(f"  FAILED \u2014 {error_msg or 'unknown error'}")
        elif status == "CANCELLED":
            print("  CANCELLED")
    else:
        print(f"  Next refresh in {DEFAULT_INTERVAL}s  (Ctrl+C to quit)")
    print("=" * W)


def main(api_url: str = DEFAULT_API_URL, task_id: str | None = None, interval: int = DEFAULT_INTERVAL) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    print("Connecting to ingestion API...")

    status = fetch_status(api_url, task_id)
    if status is None:
        print(f"No ingestion task found at {api_url}")
        sys.exit(0)

    first_poll_ts = time.time()
    prev = None

    try:
        while True:
            status["_poll_ts"] = time.time()
            render_dashboard(status, prev, status["_poll_ts"], first_poll_ts)

            if status.get("status") in TERMINAL_STATES:
                sys.exit(0)

            prev = status
            time.sleep(interval)
            status = fetch_status(api_url, task_id)
            if status is None:
                print("Lost connection to API or task expired.")
                sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion monitor")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--task-id", default=None, help="Specific task ID to monitor")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Refresh interval in seconds")
    args = parser.parse_args()
    main(api_url=args.api_url, task_id=args.task_id, interval=args.interval)
