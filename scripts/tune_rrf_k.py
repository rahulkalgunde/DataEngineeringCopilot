#!/usr/bin/env python3
"""Grid tune RRF k + prefetch on train/holdout.

Grid: k in {2,5,20,61} , L in {40,100} (8 configs). Pick max nDCG@10 on train 110,
validate winner on held 110 via bootstrap 95% CI for Δ = hybrid_best - dense.
Emit ``data/tune_rrf_k.json`` with best_k, best_prefetch, train_ndcg, held_ndcg, CI.
If CI includes 0 -> keep k=60 prefetch 40 (settings.py unchanged).
If CI excludes 0 and Δ>0 -> ship best_k + retrieval_prefetch_limit=100.

Uses ``tests/evaluation/golden/recall_inscope.jsonl`` 220q, deterministic split seed=42.
Reuses embedding cache (embed each query once, reuse across 8 configs) to avoid
880 embedding calls. No LLM calls (rewriter detached).
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
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


async def _embed_queries(service, queries: list[dict]) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    for q in queries:
        qid = str(q.get("id", ""))
        question = q.get("question") or ""
        if not question or not qid:
            continue
        if qid not in cache:
            vec = await service.embedder.embed_query(question)
            cache[qid] = list(vec)
    return cache


async def _eval_configs(
    service,
    queries: list[dict],
    embedded: dict[str, list[float]],
    grid_k: list[int],
    grid_l: list[int],
    top_k: int = 10,
) -> dict[tuple[int, int], dict]:
    from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k
    from data_engineering_copilot.services.query_signals import SearchMode

    results: dict[tuple[int, int], dict] = {}
    orig_k = getattr(service.vector_store, "_hybrid_rrf_k", 60)
    for k in grid_k:
        for lim in grid_l:
            # set RRF k for this config
            if hasattr(service.vector_store, "_hybrid_rrf_k"):
                service.vector_store._hybrid_rrf_k = k  # type: ignore[attr-defined]
            per_ndcg: list[float] = []
            per_recall: list[float] = []
            # dense baseline for same queries not needed here; captured separately
            for item in queries:
                qid = str(item.get("id", ""))
                qtext = item.get("question") or ""
                expected = [u for u in (item.get("expected_urls") or []) if u]
                emb = embedded.get(qid)
                if emb is None or not qtext:
                    per_ndcg.append(0.0)
                    per_recall.append(0.0)
                    continue
                retrieved = await service.vector_store.query(
                    emb,
                    top_k=top_k,
                    query_text=qtext,
                    search_mode=SearchMode.HYBRID_EQUAL,
                    fused_limit=lim,
                )
                urls = [r.chunk.url for r in retrieved]
                per_ndcg.append(ndcg_at_k(urls, expected, top_k))
                per_recall.append(recall_at_k(urls, expected, top_k))
            results[(k, lim)] = {
                "k": k,
                "prefetch": lim,
                "ndcg": statistics.fmean(per_ndcg) if per_ndcg else 0.0,
                "recall": statistics.fmean(per_recall) if per_recall else 0.0,
                "per_ndcg": per_ndcg,
                "per_recall": per_recall,
            }
    # restore
    if hasattr(service.vector_store, "_hybrid_rrf_k"):
        service.vector_store._hybrid_rrf_k = orig_k  # type: ignore[attr-defined]
    return results


async def _eval_dense(
    service,
    queries: list[dict],
    embedded: dict[str, list[float]],
    top_k: int = 10,
) -> dict:
    from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k
    from data_engineering_copilot.services.query_signals import SearchMode

    per_ndcg: list[float] = []
    per_recall: list[float] = []
    for item in queries:
        qid = str(item.get("id", ""))
        qtext = item.get("question") or ""
        expected = [u for u in (item.get("expected_urls") or []) if u]
        emb = embedded.get(qid)
        if emb is None or not qtext:
            per_ndcg.append(0.0)
            per_recall.append(0.0)
            continue
        retrieved = await service.vector_store.query(
            emb,
            top_k=top_k,
            query_text=qtext,
            search_mode=SearchMode.DENSE_ONLY,
        )
        urls = [r.chunk.url for r in retrieved]
        per_ndcg.append(ndcg_at_k(urls, expected, top_k))
        per_recall.append(recall_at_k(urls, expected, top_k))
    return {
        "ndcg": statistics.fmean(per_ndcg) if per_ndcg else 0.0,
        "recall": statistics.fmean(per_recall) if per_recall else 0.0,
        "per_ndcg": per_ndcg,
        "per_recall": per_recall,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Tune RRF k + prefetch on train/holdout")
    ap.add_argument("--dataset", default="tests/evaluation/golden/recall_inscope.jsonl")
    ap.add_argument("--k", type=int, default=10, help="top_k for nDCG/recall")
    ap.add_argument("--output", default="data/tune_rrf_k.json")
    args = ap.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    if not dataset_path.exists():
        print(f"❌ dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    queries = _load_queries(dataset_path)
    if len(queries) != 220:
        print(f"⚠️ expected 220 queries, got {len(queries)}")

    from data_engineering_copilot.evaluation.retrieval import bootstrap_delta_ci, split_queries

    train, held = split_queries(queries, seed=42)
    print(f"Dataset {dataset_path.name}: total={len(queries)} train={len(train)} held={len(held)} seed=42")
    print(f"Grid k=[2,5,20,61] L=[40,100] (8 configs) top_k={args.k}")

    from data_engineering_copilot.cli import _disable_rewrites_for_eval
    from data_engineering_copilot.factory import build_rag_service

    service = build_rag_service(embedding_purpose="evaluation")
    _disable_rewrites_for_eval(service)

    # embed once for reuse across configs
    all_q = train + held
    print("Embedding queries (once, reused across 8 configs)...")
    embedded = asyncio.run(_embed_queries(service, all_q))
    print(f"Embedded {len(embedded)} queries")

    grid_k = [2, 5, 20, 61]
    grid_l = [40, 100]

    # evaluate train grid
    print("Evaluating train split...")
    train_results = asyncio.run(_eval_configs(service, train, embedded, grid_k, grid_l, top_k=args.k))
    # pick best by nDCG on train
    best_key = max(train_results, key=lambda kk: train_results[kk]["ndcg"])
    best_train = train_results[best_key]
    print("\nTrain results (nDCG@10):")
    for (k, lim), res in sorted(train_results.items()):
        marker = " ← best" if (k, lim) == best_key else ""
        print(f"  k={k:>2} L={lim:>3} ndcg={res['ndcg']:.4f} recall={res['recall']:.4f}{marker}")

    # dense baseline on train and held
    print("\nEvaluating dense baselines...")
    dense_train = asyncio.run(_eval_dense(service, train, embedded, top_k=args.k))
    dense_held = asyncio.run(_eval_dense(service, held, embedded, top_k=args.k))
    print(f"  dense train ndcg={dense_train['ndcg']:.4f} recall={dense_train['recall']:.4f}")
    print(f"  dense held  ndcg={dense_held['ndcg']:.4f} recall={dense_held['recall']:.4f}")

    # validate best config on held
    best_k, best_lim = best_key
    # re-eval held for winner (reuse embedded, query with best k/L)
    print(f"\nValidating winner k={best_k} L={best_lim} on held...")
    # set k and eval held hybrid
    if hasattr(service.vector_store, "_hybrid_rrf_k"):
        service.vector_store._hybrid_rrf_k = best_k  # type: ignore[attr-defined]
    from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k
    from data_engineering_copilot.services.query_signals import SearchMode

    async def _held_hybrid() -> dict:
        per_ndcg: list[float] = []
        per_recall: list[float] = []
        for item in held:
            qid = str(item.get("id", ""))
            qtext = item.get("question") or ""
            expected = [u for u in (item.get("expected_urls") or []) if u]
            emb = embedded.get(qid)
            if emb is None or not qtext:
                per_ndcg.append(0.0)
                per_recall.append(0.0)
                continue
            retrieved = await service.vector_store.query(
                emb,
                top_k=args.k,
                query_text=qtext,
                search_mode=SearchMode.HYBRID_EQUAL,
                fused_limit=best_lim,
            )
            urls = [r.chunk.url for r in retrieved]
            per_ndcg.append(ndcg_at_k(urls, expected, args.k))
            per_recall.append(recall_at_k(urls, expected, args.k))
        return {
            "ndcg": statistics.fmean(per_ndcg) if per_ndcg else 0.0,
            "recall": statistics.fmean(per_recall) if per_recall else 0.0,
            "per_ndcg": per_ndcg,
            "per_recall": per_recall,
        }

    held_hybrid = asyncio.run(_held_hybrid())
    # restore
    if hasattr(service.vector_store, "_hybrid_rrf_k"):
        service.vector_store._hybrid_rrf_k = 60  # type: ignore[attr-defined]

    print(f"  held hybrid ndcg={held_hybrid['ndcg']:.4f} recall={held_hybrid['recall']:.4f}")

    # bootstrap CI on held nDCG delta hybrid - dense
    mean_delta, (ci_lo, ci_hi) = bootstrap_delta_ci(
        held_hybrid["per_ndcg"], dense_held["per_ndcg"], n_boot=1000, seed=42
    )
    includes_zero = ci_lo <= 0 <= ci_hi
    print(
        f"\nHeld Δ nDCG hybrid - dense: {mean_delta:+.4f} 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] includes_0={includes_zero}"
    )
    # also recall delta for context
    mean_delta_r, (ci_lo_r, ci_hi_r) = bootstrap_delta_ci(
        held_hybrid["per_recall"], dense_held["per_recall"], n_boot=1000, seed=42
    )
    print(f"Held Δ recall hybrid - dense: {mean_delta_r:+.4f} 95% CI [{ci_lo_r:+.4f}, {ci_hi_r:+.4f}]")

    decision = "ship" if (not includes_zero and mean_delta > 0) else "keep k=60 prefetch 40"
    if decision == "ship":
        print(f"Decision: hybrid wins on held (CI excludes 0, Δ>0) -> ship k={best_k} prefetch={best_lim}")
    else:
        print("Decision: inconclusive — keep k=60 prefetch 40 (settings.py unchanged)")

    out = {
        "best_k": best_k,
        "best_prefetch": best_lim,
        "train_ndcg": best_train["ndcg"],
        "train_recall": best_train["recall"],
        "held_ndcg": held_hybrid["ndcg"],
        "held_recall": held_hybrid["recall"],
        "dense_train_ndcg": dense_train["ndcg"],
        "dense_held_ndcg": dense_held["ndcg"],
        "dense_train_recall": dense_train["recall"],
        "dense_held_recall": dense_held["recall"],
        "held_delta_ndcg": mean_delta,
        "held_ci_ndcg": [ci_lo, ci_hi],
        "held_delta_recall": mean_delta_r,
        "held_ci_recall": [ci_lo_r, ci_hi_r],
        "ci_includes_zero": includes_zero,
        "decision": decision,
        "grid": {f"k{k}_L{lim}": {"ndcg": v["ndcg"], "recall": v["recall"]} for (k, lim), v in train_results.items()},
        "k": args.k,
        "n_train": len(train),
        "n_held": len(held),
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
