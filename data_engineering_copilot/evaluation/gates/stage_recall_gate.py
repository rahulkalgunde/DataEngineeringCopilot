"""Stage-by-stage recall attribution gate for RAG evaluation.

Runs a representative sample of eval queries through the live pipeline and
computes per-stage Recall@K and chunk survival rates. Exits non-zero when any
stage drops recall below the configured floor, or when the largest recall-loss
stage exceeds the configured drop threshold.

This module can be invoked as:
    python -m data_engineering_copilot.evaluation.gates.stage_recall_gate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from data_engineering_copilot.evaluation.retrieval_metrics import recall_at_k
from data_engineering_copilot.factory import build_rag_service


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage recall attribution gate for RAG evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m data_engineering_copilot.evaluation.gates.stage_recall_gate \\\n"
            "      --dataset tests/evaluation/eval_dataset_spark.jsonl \\\n"
            "      --sample-size 20 --k 10 --recall-floor 0.80 --max-stage-drop 0.30\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/evaluation/golden/recall_all.jsonl"),
        help="JSONL eval dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/stage_recall_gate"),
        help="Directory to write diagnostic JSON output",
    )
    parser.add_argument("--sample-size", type=int, default=20, help="Number of queries to sample")
    parser.add_argument("--k", type=int, default=10, help="Recall@K cutoff")
    parser.add_argument("--recall-floor", type=float, default=0.80, help="Minimum acceptable recall per stage")
    parser.add_argument("--max-stage-drop", type=float, default=0.30, help="Maximum allowed recall drop between stages")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for query sampling")
    return parser.parse_args()


def _load_queries(path: Path, n: int, seed: int) -> list[dict]:
    """Load and sample in-scope queries from a JSONL dataset."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("out_of_scope") and item.get("expected_urls"):
                rows.append(item)
    if not rows:
        raise RuntimeError(f"No in-scope rows with expected_urls in {path}")
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


async def _run(queries: list[dict], k: int) -> tuple[list[dict], dict]:
    """Run stage-by-stage recall attribution for sampled queries."""
    service = build_rag_service()
    records: list[dict] = []
    stage_metrics: dict[str, dict[str, list[float]]] = {}

    for i, item in enumerate(queries, 1):
        q = item.get("question") or ""
        expected = list(item.get("expected_urls", []))
        prov: list[dict] = []
        t0 = time.monotonic()
        try:
            answer = await service.answer(
                q,
                provenance=prov,
                bypass_cache=True,
                retrieval_only=True,
                expected_urls=expected,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(queries)}] ERROR: {exc}")
            continue
        latency = (time.monotonic() - t0) * 1000.0

        prov_record = prov[-1] if prov else {}
        records.append(
            {
                "id": item.get("id", f"q{i}"),
                "question": q,
                "expected_urls": expected,
                "final_recall": recall_at_k([c.url for c in answer.sources], expected, k),
                "latency_ms": latency,
                "provenance": prov_record,
            }
        )

        prev_ids: set[str] = set()
        prev_recall = None
        for snap in prov_record.get("stage_snapshots") or []:
            stage = snap.get("stage", "unknown")
            urls = [u for u in (snap.get("urls") or []) if u]
            ids = [c for c in (snap.get("chunk_ids") or []) if c]
            rec = recall_at_k(urls, expected, k)
            surv = 1.0 if not prev_ids else len(set(ids) & prev_ids) / len(prev_ids)
            drop = 0.0 if prev_recall is None else max(0.0, prev_recall - rec)
            stage_metrics.setdefault(stage, {"recall": [], "survival": [], "drop": []})
            stage_metrics[stage]["recall"].append(rec)
            stage_metrics[stage]["survival"].append(surv)
            stage_metrics[stage]["drop"].append(drop)
            prev_ids = set(ids)
            prev_recall = rec

        print(
            f"[{i}/{len(queries)}] {item.get('id', '')}: "
            f"final_recall={records[-1]['final_recall']:.2f} "
            f"latency={latency:.0f}ms"
        )

    summary = {
        stage: {
            "avg_recall_at_k": sum(v["recall"]) / len(v["recall"]) if v["recall"] else 0.0,
            "avg_survival_rate": sum(v["survival"]) / len(v["survival"]) if v["survival"] else 0.0,
            "avg_recall_drop": sum(v["drop"]) / len(v["drop"]) if v["drop"] else 0.0,
            "max_recall_drop": max(v["drop"]) if v["drop"] else 0.0,
        }
        for stage, v in stage_metrics.items()
    }
    return records, summary


def _report(records: list[dict], summary: dict, recall_floor: float, max_stage_drop: float) -> bool:
    """Report stage recall metrics and determine gate pass/fail."""
    failed = False
    print("\n=== Stage recall attribution ===")
    for stage in sorted(summary):
        m = summary[stage]
        flag = ""
        if m["avg_recall_at_k"] < recall_floor:
            flag = " [BELOW FLOOR]"
            failed = True
        if m["max_recall_drop"] > max_stage_drop:
            flag = " [DROP EXCEEDED]"
            failed = True
        print(
            f"  {stage:20s}: recall={m['avg_recall_at_k']:.3f} "
            f"survival={m['avg_survival_rate']:.3f} "
            f"avg_drop={m['avg_recall_drop']:.3f} "
            f"max_drop={m['max_recall_drop']:.3f}"
            f"{flag}"
        )

    overall = [r["final_recall"] for r in records]
    avg_final = sum(overall) / len(overall) if overall else 0.0
    print(f"\nOverall avg final recall@k={avg_final:.3f} (n={len(records)})")
    if failed:
        print("\n❌ Stage recall gate FAILED")
        return False
    print("\n✅ Stage recall gate PASSED")
    return True


def main() -> int:
    """CLI entrypoint for stage recall attribution gate."""
    args = _parse_args()
    queries = _load_queries(args.dataset, args.sample_size, args.seed)
    records, summary = asyncio.run(_run(queries, args.k))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote diagnostics to {args.output_dir}")
    return 0 if _report(records, summary, args.recall_floor, args.max_stage_drop) else 1


if __name__ == "__main__":
    raise SystemExit(main())
