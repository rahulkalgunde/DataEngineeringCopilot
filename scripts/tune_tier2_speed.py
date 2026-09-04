#!/usr/bin/env python3
"""Grid tune retrieval_top_k × mrl_multistage on train/holdout.

Grid: top_k in {25,50} × mrl in {off,on} = 4 configs.
  For each cfg on train 110 (seed 42):
    set service.config.retrieval_top_k and service.vector_store._mrl_enabled,
    run retrieval_only eval (zero-LLM) and capture recall@k and p95 latency.
  Pick candidate with max recall@k where p95 improves ≥20% vs 50/off
  else keep 50/off.
  Validate winner on held 110 via bootstrap 95% CI for Δ recall (1000 resamples):
    require CI lower bound > -0.01 (settings.py:1342 gate) else revert.
  Emit data/tune_tier2_speed.json.

Reuses tune_rrf_k pattern: service built once, mutated per config,
eval via service.answer(retrieval_only=True) (zero LLM).  Embeddings
are local-hf (2048d) and run per-query; 4×110=440 on train, +110 for
held winner ⇒ ~550 local embeddings, cached in-process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_queries(path: Path) -> list[dict]:
    qs: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                qs.append(json.loads(line))
    return qs


async def _eval_split(
    service,
    queries: list[dict],
    k: int = 10,
) -> dict:
    """Run retrieval_only eval over *queries*; return per-query recall and latencies."""
    from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k

    per_recall: list[float] = []
    per_ndcg: list[float] = []
    latencies_ms: list[float] = []
    # sequential to avoid Qdrant thundering herd; matches CLI batch_size spacing
    for item in queries:
        question = item.get("question") or ""
        expected = [u for u in (item.get("expected_urls") or []) if u]
        if not question:
            per_recall.append(0.0)
            per_ndcg.append(0.0)
            latencies_ms.append(0.0)
            continue
        prov: list[dict] = []
        t0 = time.perf_counter()
        try:
            answer = await service.answer(
                question,
                provenance=prov,
                bypass_cache=True,
                retrieval_only=True,
                expected_urls=expected,
            )
            retrieved_urls = [c.url for c in answer.sources]
        except Exception:
            retrieved_urls = []
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000.0
        latencies_ms.append(lat)
        try:
            rec = recall_at_k(retrieved_urls, expected, k)
            ndc = ndcg_at_k(retrieved_urls, expected, k)
        except Exception:
            rec = 0.0
            ndc = 0.0
        per_recall.append(rec)
        per_ndcg.append(ndc)
    import statistics

    from data_engineering_copilot.evaluation.retrieval_metrics import percentile

    mean_recall = statistics.fmean(per_recall) if per_recall else 0.0
    mean_ndcg = statistics.fmean(per_ndcg) if per_ndcg else 0.0
    p50 = percentile(latencies_ms, 0.5) if latencies_ms else 0.0
    p95 = percentile(latencies_ms, 0.95) if latencies_ms else 0.0
    return {
        "per_recall": per_recall,
        "per_ndcg": per_ndcg,
        "latencies_ms": latencies_ms,
        "mean_recall": mean_recall,
        "mean_ndcg": mean_ndcg,
        "p50": p50,
        "p95": p95,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune retrieval_top_k × mrl on train/holdout")
    ap.add_argument("--dataset", default="tests/evaluation/golden/recall_all.jsonl")
    ap.add_argument("--k", type=int, default=10, help="top_k for recall/ndcg")
    ap.add_argument("--batch-size", type=int, default=55, help="unused, kept for CLI parity")
    ap.add_argument("--split", default="all", help="unused; always evaluates train→held")
    ap.add_argument("--output", default="data/tune_tier2_speed.json")
    args = ap.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    if not dataset_path.exists():
        # fallback to recall_all
        alt = PROJECT_ROOT / "tests/evaluation/golden/recall_all.jsonl"
        if alt.exists():
            dataset_path = alt
        else:
            print(f"❌ dataset not found: {dataset_path}", file=sys.stderr)
            return 2
    queries = _load_queries(dataset_path)
    if len(queries) != 220:
        print(f"⚠️ expected 220 queries, got {len(queries)}")

    from data_engineering_copilot.evaluation.retrieval import bootstrap_delta_ci, split_queries

    train, held = split_queries(queries, seed=42)
    print(f"Dataset {dataset_path.name}: total={len(queries)} train={len(train)} held={len(held)} seed=42")
    print(f"Grid top_k=[50,25] mrl=[off,on] (4 configs) k={args.k}")

    from data_engineering_copilot.cli import _disable_rewrites_for_eval
    from data_engineering_copilot.factory import build_rag_service

    service = build_rag_service(embedding_purpose="evaluation")
    _disable_rewrites_for_eval(service)
    # ensure reranker initialized if present (local cross-encoder is CPU, no LLM)
    if service.reranker is not None:
        try:
            asyncio.run(service.reranker.initialize())
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ reranker init failed: {exc}")

    # grid configs: baseline first for comparison
    grid = [
        (50, False),
        (25, False),
        (50, True),
        (25, True),
    ]
    # map for results
    train_results: dict[tuple[int, bool], dict] = {}
    baseline_key = (50, False)

    import dataclasses

    for top_k, mrl in grid:
        # mutate service config (RagConfig is frozen)
        service.config = dataclasses.replace(service.config, retrieval_top_k=top_k)
        # mrl presupposes hybrid
        hybrid = getattr(service.vector_store, "_hybrid_search", True)
        service.vector_store._mrl_enabled = bool(mrl and hybrid)  # type: ignore[attr-defined]
        # also keep settings._mrl_* dims consistent (store reads its own dims)
        service.vector_store._mrl_small_dim = 256  # type: ignore[attr-defined]
        service.vector_store._mrl_oversample_factor = 4  # type: ignore[attr-defined]
        label = f"top_k={top_k} mrl={'on' if mrl else 'off'} (mrl_enabled={service.vector_store._mrl_enabled})"  # type: ignore[attr-defined]
        print(f"\nEvaluating train split [{label}] ...")
        res = asyncio.run(_eval_split(service, train, k=args.k))
        print(
            f"  train recall@{args.k}={res['mean_recall']:.4f} ndcg={res['mean_ndcg']:.4f} p50={res['p50']:.0f}ms p95={res['p95']:.0f}ms"
        )
        train_results[(top_k, mrl)] = res

    baseline_train = train_results[baseline_key]
    baseline_p95 = baseline_train["p95"]

    # pick candidate with max recall where p95 improves ≥20%
    candidates: list[tuple[tuple[int, bool], dict]] = []
    for key, res in train_results.items():
        if key == baseline_key:
            continue
        p95 = res["p95"]
        improvement = (baseline_p95 - p95) / baseline_p95 if baseline_p95 > 0 else 0.0
        if improvement >= 0.20:
            candidates.append((key, res))
            print(
                f"  candidate {key} qualifies: p95 {p95:.0f} vs baseline {baseline_p95:.0f} improvement {improvement:.1%}"
            )
        else:
            print(
                f"  candidate {key} rejected: p95 improvement {improvement:.1%} <20% (p95 {p95:.0f} vs {baseline_p95:.0f})"
            )

    if candidates:
        # max recall among qualifiers
        best_key, best_train = max(candidates, key=lambda kv: kv[1]["mean_recall"])
        print(
            f"\nTrain winner: top_k={best_key[0]} mrl={'on' if best_key[1] else 'off'} recall={best_train['mean_recall']:.4f} p95={best_train['p95']:.0f}ms"
        )
    else:
        best_key = baseline_key
        best_train = baseline_train
        print("\nNo candidate improves p95 ≥20% — keep baseline 50/off")

    # validate winner on held
    if best_key == baseline_key:
        # no validation needed; still evaluate held for baseline for output
        service.config = dataclasses.replace(service.config, retrieval_top_k=baseline_key[0])
        service.vector_store._mrl_enabled = False  # type: ignore[attr-defined]
        held_baseline = asyncio.run(_eval_split(service, held, k=args.k))
        held_winner = held_baseline
        delta_mean = 0.0
        ci = (0.0, 0.0)
        decision = "keep 50"
        best_held_recall = held_baseline["mean_recall"]
        best_held_p95 = held_baseline["p95"]
        print(
            f"\nHeld baseline (50/off): recall={held_baseline['mean_recall']:.4f} p95={held_baseline['p95']:.0f}ms — decision keep 50"
        )
    else:
        # held baseline
        service.config = dataclasses.replace(service.config, retrieval_top_k=baseline_key[0])
        service.vector_store._mrl_enabled = False  # type: ignore[attr-defined]
        held_baseline = asyncio.run(_eval_split(service, held, k=args.k))
        print(f"\nHeld baseline (50/off): recall={held_baseline['mean_recall']:.4f} p95={held_baseline['p95']:.0f}ms")
        # held winner
        service.config = dataclasses.replace(service.config, retrieval_top_k=best_key[0])
        service.vector_store._mrl_enabled = bool(best_key[1] and getattr(service.vector_store, "_hybrid_search", True))  # type: ignore[attr-defined]
        held_winner = asyncio.run(_eval_split(service, held, k=args.k))
        print(
            f"Held winner top_k={best_key[0]} mrl={'on' if best_key[1] else 'off'}: recall={held_winner['mean_recall']:.4f} p95={held_winner['p95']:.0f}ms"
        )
        # bootstrap CI for Δ recall (winner - baseline) on held
        delta_mean, ci = bootstrap_delta_ci(
            held_winner["per_recall"], held_baseline["per_recall"], n_boot=1000, seed=42
        )
        ci_lo, ci_hi = ci
        print(
            f"Held Δ recall winner - baseline: {delta_mean:+.4f} 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] lower>{-0.01} ? {ci_lo > -0.01}"
        )
        # also p95 gate on held: improvement ≥20% ?
        held_improvement = (
            (held_baseline["p95"] - held_winner["p95"]) / held_baseline["p95"] if held_baseline["p95"] else 0.0
        )
        p95_ok = held_improvement >= 0.20
        print(
            f"Held p95 improvement {held_improvement:.1%} {'≥20% OK' if p95_ok else '<20% FAIL'} (baseline {held_baseline['p95']:.0f} winner {held_winner['p95']:.0f})"
        )
        # gate: CI lower > -0.01 AND p95 improves ≥20%
        if ci_lo > -0.01 and p95_ok:
            decision = "ship"
            print("Decision: ship (held CI> -0.01 and p95 ≥20%)")
        else:
            # revert to baseline
            print("Decision: keep 50 (held gate failed — CI or p95)")
            decision = "keep 50"
            best_key = baseline_key
            best_train = baseline_train
            best_held_recall = held_baseline["mean_recall"]
            best_held_p95 = held_baseline["p95"]
            delta_mean, ci = 0.0, (0.0, 0.0)
            held_winner = held_baseline
        if decision == "ship":
            best_held_recall = held_winner["mean_recall"]
            best_held_p95 = held_winner["p95"]
        else:
            # already set for keep
            pass

    # for keep case we already set; for ship case set deltas; normalize for output
    if best_key != baseline_key and decision == "ship":
        # delta already computed
        delta_lo, delta_hi = ci
    elif best_key == baseline_key and "ci" in locals():
        # ship vs keep: ci from held comparison before revert; if kept, delta 0
        if decision == "keep 50" and "held_baseline" in locals() and "held_winner" in locals():
            # if we reverted, delta is 0
            delta_lo, delta_hi = 0.0, 0.0
            delta_mean = 0.0
        else:
            delta_lo, delta_hi = ci
    else:
        delta_lo, delta_hi = ci if "ci" in locals() else (0.0, 0.0)
        delta_mean = delta_mean if "delta_mean" in locals() else 0.0

    # ensure held metrics reflect final best
    if decision == "keep 50":
        final_held_recall = (
            held_baseline["mean_recall"] if "held_baseline" in locals() else baseline_train["mean_recall"]
        )
        final_held_p95 = held_baseline["p95"] if "held_baseline" in locals() else baseline_train["p95"]
        best_top_k = 50
        best_mrl = False
    else:
        final_held_recall = held_winner["mean_recall"]
        final_held_p95 = held_winner["p95"]
        best_top_k, best_mrl = best_key

    out = {
        "best_top_k": best_top_k,
        "mrl": best_mrl,
        "held_recall": final_held_recall,
        "held_p95": final_held_p95,
        "delta_recall_ci": [delta_lo, delta_hi],
        "delta_recall_mean": delta_mean,
        "decision": decision,
        "k": args.k,
        "n_train": len(train),
        "n_held": len(held),
        "baseline_train_recall": baseline_train["mean_recall"],
        "baseline_train_p95": baseline_train["p95"],
        "baseline_held_recall": held_baseline["mean_recall"] if "held_baseline" in locals() else None,
        "baseline_held_p95": held_baseline["p95"] if "held_baseline" in locals() else None,
        "winner_train_recall": best_train["mean_recall"],
        "winner_train_p95": best_train["p95"],
        "grid": {
            f"top{tk}_mrl{'on' if m else 'off'}": {
                "train_recall": v["mean_recall"],
                "train_p95": v["p95"],
                "train_p50": v["p50"],
            }
            for (tk, m), v in train_results.items()
        },
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
