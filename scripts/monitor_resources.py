#!/usr/bin/env python3
"""Monitor system and Docker container resources during ingestion.

Usage:
    # Terminal 1: start monitor (runs until Ctrl+C)
    dec_venv/bin/python scripts/monitor_resources.py

    # Terminal 2: run ingestion from UI or CLI
    dec_venv/bin/python main.py ingest --max-pages 40

    # Or auto-stop after N seconds:
    dec_venv/bin/python scripts/monitor_resources.py --duration 120

Output: resources_YYYYMMDD_HHMMSS.csv + summary to stdout.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path


@dataclass
class Sample:
    timestamp: float
    elapsed_s: float

    # Host
    host_cpu_pct: float
    host_mem_used_gb: float
    host_mem_total_gb: float
    host_mem_pct: float
    host_net_rx_mb: float
    host_net_tx_mb: float
    host_disk_read_mb: float
    host_disk_write_mb: float

    # Container totals
    container_cpu_pct: float
    container_mem_used_gb: float
    container_net_rx_mb: float
    container_net_tx_mb: float

    # Per-container CPU%
    cpu_worker: float
    cpu_api: float
    cpu_qdrant: float
    cpu_redis_broker: float
    cpu_redis_cache: float
    cpu_clickhouse: float
    cpu_langfuse: float

    # Per-container Mem (GB)
    mem_worker: float
    mem_api: float
    mem_qdrant: float
    mem_redis_broker: float
    mem_redis_cache: float
    mem_clickhouse: float
    mem_langfuse: float


def _parse_mem(s: str) -> float:
    """Parse '1.153GiB' -> float GB."""
    s = s.strip()
    if s.endswith("GiB"):
        return float(s[:-3])
    if s.endswith("MiB"):
        return float(s[:-3]) / 1024
    if s.endswith("KiB"):
        return float(s[:-3]) / (1024 * 1024)
    if s.endswith("GB"):
        return float(s[:-2])
    if s.endswith("MB"):
        return float(s[:-2]) / 1024
    return 0.0


def _parse_pct(s: str) -> float:
    return float(s.strip().rstrip("%"))


def _read_net_io() -> tuple[float, float]:
    """Read host cumulative network bytes (rx, tx) from /proc/net/dev."""
    rx = tx = 0.0
    with open("/proc/net/dev") as f:
        for line in f:
            if ":" not in line:
                continue
            parts = line.split(":")[1].split()
            if len(parts) >= 10:
                rx += int(parts[0])
                tx += int(parts[8])
    return rx / (1024 * 1024), tx / (1024 * 1024)


def _read_disk_io() -> tuple[float, float]:
    """Read host cumulative disk bytes from /proc/diskstats (sum all devices)."""
    read_bytes = write_bytes = 0.0
    with open("/proc/diskstats") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 14:
                name = parts[2]
                if name.startswith("loop") or name.startswith("ram"):
                    continue
                read_bytes += int(parts[5]) * 512
                write_bytes += int(parts[9]) * 512
    return read_bytes / (1024 * 1024), write_bytes / (1024 * 1024)


def _read_host_cpu() -> float:
    """Read host CPU% from /proc/stat (delta between two reads)."""

    def _read_stat() -> list[int]:
        with open("/proc/stat") as f:
            line = f.readline()
        return [int(x) for x in line.split()[1:]]

    t1 = _read_stat()
    time.sleep(0.2)
    t2 = _read_stat()

    idle1 = t1[3] + (t1[4] if len(t1) > 4 else 0)
    idle2 = t2[3] + (t2[4] if len(t2) > 4 else 0)
    total1 = sum(t1)
    total2 = sum(t2)

    idle_d = idle2 - idle1
    total_d = total2 - total1
    if total_d == 0:
        return 0.0
    return max(0.0, (1.0 - idle_d / total_d) * 100.0)


def _read_host_mem() -> tuple[float, float, float]:
    """Returns (used_gb, total_gb, pct)."""
    info: dict[str, float] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            key = parts[0].rstrip(":")
            val = int(parts[1]) / (1024 * 1024)
            info[key] = val
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", 0)
    used = total - available
    pct = (used / total * 100) if total > 0 else 0
    return used, total, pct


def _docker_stats_once() -> dict[str, dict[str, float]]:
    """Get a single snapshot from `docker stats --no-stream`."""
    result: dict[str, dict[str, float]] = {}
    try:
        out = subprocess.check_output(
            [
                "docker", "stats", "--no-stream",
                "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}",
            ],
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return result

    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name = parts[0].strip()
        cpu = _parse_pct(parts[1])
        mem_parts = parts[2].split("/")
        mem_used = _parse_mem(mem_parts[0]) if mem_parts else 0.0
        net_parts = parts[3].split("/")
        net_rx = _parse_mem(net_parts[0]) if net_parts else 0.0
        net_tx = _parse_mem(net_parts[1]) if len(net_parts) > 1 else 0.0
        result[name] = {"cpu": cpu, "mem": mem_used, "net_rx": net_rx, "net_tx": net_tx}

    return result


def sample(
    prev_net: tuple[float, float],
    prev_disk: tuple[float, float],
    start: float,
) -> tuple[Sample, tuple[float, float], tuple[float, float]]:
    host_cpu = _read_host_cpu()
    host_mem_used, host_mem_total, host_mem_pct = _read_host_mem()
    net_rx, net_tx = _read_net_io()
    disk_r, disk_w = _read_disk_io()
    containers = _docker_stats_once()

    elapsed = time.time() - start

    def _c(name: str, key: str) -> float:
        return containers.get(name, {}).get(key, 0.0)

    return (
        Sample(
            timestamp=time.time(),
            elapsed_s=round(elapsed, 1),
            host_cpu_pct=round(host_cpu, 1),
            host_mem_used_gb=round(host_mem_used, 2),
            host_mem_total_gb=round(host_mem_total, 2),
            host_mem_pct=round(host_mem_pct, 1),
            host_net_rx_mb=round(net_rx, 1),
            host_net_tx_mb=round(net_tx, 1),
            host_disk_read_mb=round(disk_r, 1),
            host_disk_write_mb=round(disk_w, 1),
            container_cpu_pct=round(sum(c["cpu"] for c in containers.values()), 1),
            container_mem_used_gb=round(sum(c["mem"] for c in containers.values()), 2),
            container_net_rx_mb=round(sum(c["net_rx"] for c in containers.values()), 1),
            container_net_tx_mb=round(sum(c["net_tx"] for c in containers.values()), 1),
            cpu_worker=_c("de_copilot_worker", "cpu"),
            cpu_api=_c("de_copilot_api", "cpu"),
            cpu_qdrant=_c("de_copilot_vectorstore", "cpu"),
            cpu_redis_broker=_c("de_copilot_broker", "cpu"),
            cpu_redis_cache=_c("de_copilot_crawler_cache", "cpu"),
            cpu_clickhouse=_c("de_copilot_clickhouse", "cpu"),
            cpu_langfuse=_c("de_copilot_observability", "cpu"),
            mem_worker=_c("de_copilot_worker", "mem"),
            mem_api=_c("de_copilot_api", "mem"),
            mem_qdrant=_c("de_copilot_vectorstore", "mem"),
            mem_redis_broker=_c("de_copilot_broker", "mem"),
            mem_redis_cache=_c("de_copilot_crawler_cache", "mem"),
            mem_clickhouse=_c("de_copilot_clickhouse", "mem"),
            mem_langfuse=_c("de_copilot_observability", "mem"),
        ),
        (net_rx, net_tx),
        (disk_r, disk_w),
    )


def _print_bar(label: str, value: float, max_val: float, width: int = 40, unit: str = "%"):
    filled = int(value / max_val * width) if max_val > 0 else 0
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {label:>12s} |{bar}| {value:6.1f}{unit}")


def print_summary(samples: list[Sample]):
    if not samples:
        return

    print("\n" + "=" * 80)
    print("  RESOURCE MONITORING SUMMARY")
    print("=" * 80)

    duration = samples[-1].elapsed_s - samples[0].elapsed_s
    print(f"\n  Duration: {duration:.0f}s  |  Samples: {len(samples)}")

    def _stats(attr: str) -> tuple[float, float]:
        vals = [getattr(s, attr) for s in samples]
        return sum(vals) / len(vals), max(vals)

    print("\n--- HOST ---")
    for label, attr, mx, unit in [
        ("CPU", "host_cpu_pct", 100, "%"),
        ("Memory", "host_mem_pct", 100, "%"),
    ]:
        avg, mx_val = _stats(attr)
        _print_bar(label, mx_val, mx, unit=unit)
        print(f"             avg={avg:.1f}{unit}  max={mx_val:.1f}{unit}")

    host_mem_avg, _ = _stats("host_mem_used_gb")
    host_mem_mx, _ = _stats("host_mem_used_gb")
    print(f"             Mem used: avg={host_mem_avg:.1f}GB  max={host_mem_mx:.1f}GB / {samples[0].host_mem_total_gb:.1f}GB")

    net_rx_first = samples[0].host_net_rx_mb
    net_rx_last = samples[-1].host_net_rx_mb
    net_tx_first = samples[0].host_net_tx_mb
    net_tx_last = samples[-1].host_net_tx_mb
    if duration > 0:
        rx_rate = (net_rx_last - net_rx_first) / duration
        tx_rate = (net_tx_last - net_tx_first) / duration
    else:
        rx_rate = tx_rate = 0
    print(f"  Network RX: total={net_rx_last - net_rx_first:.1f}MB  avg_rate={rx_rate:.2f}MB/s")
    print(f"  Network TX: total={net_tx_last - net_tx_first:.1f}MB  avg_rate={tx_rate:.2f}MB/s")

    print("\n--- CONTAINERS (CPU%) ---")
    cpu_containers = [
        ("Worker", "cpu_worker"),
        ("API", "cpu_api"),
        ("Qdrant", "cpu_qdrant"),
        ("Redis Broker", "cpu_redis_broker"),
        ("Redis Cache", "cpu_redis_cache"),
        ("ClickHouse", "cpu_clickhouse"),
        ("Langfuse", "cpu_langfuse"),
    ]
    for label, attr in cpu_containers:
        avg, mx = _stats(attr)
        if mx > 0.1:
            _print_bar(label, mx, 100)
            print(f"             avg={avg:.1f}%  max={mx:.1f}%")

    print("\n--- CONTAINERS (Memory GB) ---")
    mem_containers = [
        ("Worker", "mem_worker"),
        ("API", "mem_api"),
        ("Qdrant", "mem_qdrant"),
        ("ClickHouse", "mem_clickhouse"),
        ("Langfuse", "mem_langfuse"),
    ]
    for label, attr in mem_containers:
        avg, mx = _stats(attr)
        if mx > 0.01:
            print(f"  {label:>12s}: avg={avg:.2f}GB  max={mx:.2f}GB")

    print("\n--- BOTTLENECK ANALYSIS ---")
    worker_cpu_avg, worker_cpu_mx = _stats("cpu_worker")
    worker_mem_avg, worker_mem_mx = _stats("mem_worker")
    host_cpu_avg, _ = _stats("host_cpu_pct")
    host_mem_pct_avg, _ = _stats("host_mem_pct")

    print(f"  Host CPU avg:  {host_cpu_avg:.1f}%")
    print(f"  Host Mem avg:  {host_mem_pct_avg:.1f}%")
    print(f"  Worker CPU:    avg={worker_cpu_avg:.1f}%  max={worker_cpu_mx:.1f}%")
    print(f"  Worker Mem:    avg={worker_mem_avg:.2f}GB  max={worker_mem_mx:.2f}GB")

    idle_cpu = 100 - host_cpu_avg
    idle_mem_pct = 100 - host_mem_pct_avg

    print(f"\n  CPU headroom:  {idle_cpu:.1f}%")
    print(f"  Mem headroom:  {idle_mem_pct:.1f}%")

    if idle_cpu > 40 and idle_mem_pct > 30:
        print("\n  >> VERDICT: Significant headroom. Aggressive concurrency increase likely safe.")
        if worker_cpu_mx < 70:
            print(f"     Worker never exceeded {worker_cpu_mx:.0f}% CPU. Can likely 2-3x crawl settings.")
    elif idle_cpu > 20:
        print("\n  >> VERDICT: Moderate headroom. Conservative increase recommended (1.5-2x).")
    else:
        print("\n  >> VERDICT: System near capacity. Increase cautiously or add resources.")

    print("\n  SUGGESTED AGGRESSIVE VALUES (based on current usage):")
    if idle_cpu > 40:
        print("    crawl_concurrency:           20 -> 50")
        print("    crawl_max_concurrency:       40 -> 100")
        print("    crawl_thread_pool_size:       8 -> 16")
        print("    crawl_domain_delay_seconds: 0.5 -> 0.2")
        print("    crawl_delay_seconds:        0.05 -> 0.02")
        print("    ingestion_batch_chunk_size:  256 -> 512")
    elif idle_cpu > 20:
        print("    crawl_concurrency:           20 -> 35")
        print("    crawl_max_concurrency:       40 -> 70")
        print("    crawl_thread_pool_size:       8 -> 12")
        print("    crawl_domain_delay_seconds: 0.5 -> 0.3")
    else:
        print("    (System is saturated - no increase recommended)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monitor resources during ingestion")
    parser.add_argument("--duration", type=int, default=0, help="Stop after N seconds (0 = run until Ctrl+C)")
    parser.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds (default: 2)")
    parser.add_argument("--output", type=str, default="", help="Output CSV path (auto-generated if empty)")
    args = parser.parse_args()

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"resources_{ts}.csv"

    csv_path = Path(args.output)
    field_names = [f.name for f in fields(Sample)]

    print(f"Monitoring resources every {args.interval}s ...")
    if args.duration:
        print(f"Stopping after {args.duration}s")
    print(f"Output: {csv_path}")
    print("Press Ctrl+C to stop early.\n")

    start = time.time()
    prev_net = _read_net_io()
    prev_disk = _read_disk_io()
    samples: list[Sample] = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()

        try:
            while True:
                s, prev_net, prev_disk = sample(prev_net, prev_disk, start)
                samples.append(s)
                writer.writerow({k: getattr(s, k) for k in field_names})
                f.flush()

                print(
                    f"\r  [{s.elapsed_s:6.0f}s] "
                    f"Host CPU: {s.host_cpu_pct:5.1f}% | "
                    f"Mem: {s.host_mem_pct:5.1f}% | "
                    f"Worker CPU: {s.cpu_worker:5.1f}% | "
                    f"Worker Mem: {s.mem_worker:5.2f}GB | "
                    f"Net RX: {s.host_net_rx_mb:7.1f}MB",
                    end="",
                    flush=True,
                )

                if args.duration and s.elapsed_s >= args.duration:
                    break

                sleep_time = max(0, args.interval - 0.2)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nStopped by user.")

    print_summary(samples)
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
