#!/usr/bin/env python3
"""Grid tune RRF weights + bm25 b at best k=20 prefetch 100.

Grid: weights ∈ {(1,1),(1,1.25),(1,0.8),(2,1)} — absolute per Qdrant docs (1,2) != (2,4).
b ∈ {0.5,0.75} — query-side approximation without re-upsert (stored sparse vectors remain
at b=0.75). Rebuilding vocab per b requires gen-bm25-rebuild (~15s each); this script
mutates the live tokenizer's b for query tokenization only. If approximation wins,
a full gen-build is required to bake b.

Uses tests/evaluation/golden/recall_all.jsonl, deterministic split seed=42.
Reuses embedding cache (each query embedded once). No LLM calls.

Emits data/tune_rrf_weights.json with train/held nDCG, Δ CI, decision.
Ship only if best weights held Δ CI>0 else keep (1,1) b=0.75.
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


async def _eval_with_weights(
    service,
    queries: list[dict],
    embedded: dict[str, list[float]],
    weights: tuple[float, float] | None,
    top_k: int = 10,
    fused_limit: int = 100,
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
            search_mode=SearchMode.HYBRID_EQUAL,
            fused_limit=fused_limit,
            rrf_weights=weights,
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

    ap = argparse.ArgumentParser(description="Tune RRF weights + bm25 b")
    ap.add_argument("--dataset", default="tests/evaluation/golden/recall_all.jsonl")
    ap.add_argument("--k", type=int, default=10, help="top_k for nDCG/recall")
    ap.add_argument("--output", default="data/tune_rrf_weights.json")
    ap.add_argument("--b-sweep", action="store_true", help="also sweep b∈{0.5,0.75} (query-side approx)")
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

    # best k/prefetch from Task 4
    tune_k_path = PROJECT_ROOT / "data/tune_rrf_k.json"
    if tune_k_path.exists():
        meta = json.loads(tune_k_path.read_text(encoding="utf-8"))
        best_k = int(meta.get("best_k", 20))
        best_prefetch = int(meta.get("best_prefetch", 100))
        print(f"Using best k={best_k} prefetch={best_prefetch} from {tune_k_path}")
    else:
        best_k, best_prefetch = 20, 100
        print(f"No tune_rrf_k.json, defaulting k={best_k} prefetch={best_prefetch}")

    weight_grid: list[tuple[float, float]] = [(1, 1), (1, 1.25), (1, 0.8), (2, 1)]
    print(f"Grid weights={weight_grid} at k={best_k} prefetch={best_prefetch} top_k={args.k}")

    from data_engineering_copilot.cli import _disable_rewrites_for_eval
    from data_engineering_copilot.factory import build_rag_service

    service = build_rag_service(embedding_purpose="evaluation")
    _disable_rewrites_for_eval(service)

    # honor best_k in store
    if hasattr(service.vector_store, "_hybrid_rrf_k"):
        service.vector_store._hybrid_rrf_k = best_k  # type: ignore[attr-defined]

    all_q = train + held
    print("Embedding queries (once, reused across configs)...")
    embedded = asyncio.run(_embed_queries(service, all_q))
    print(f"Embedded {len(embedded)} queries")

    # Evaluate grid on train
    print("\nEvaluating train split (weights)...")
    train_results: dict[tuple[float, float], dict] = {}
    for w in weight_grid:
        rrf_w = None if w == (1, 1) else w
        res = asyncio.run(_eval_with_weights(service, train, embedded, rrf_w, top_k=args.k, fused_limit=best_prefetch))
        train_results[w] = res
        tag = "equal (None)" if w == (1, 1) else str(w)
        print(f"  weights={tag:12s} ndcg={res['ndcg']:.4f} recall={res['recall']:.4f}")

    best_w = max(train_results, key=lambda kk: train_results[kk]["ndcg"])
    best_train = train_results[best_w]
    best_tag = "equal (None)" if best_w == (1, 1) else str(best_w)
    print(f"\nBest on train: weights={best_tag} ndcg={best_train['ndcg']:.4f}")

    # Dense baseline for reference (needed for CI on held)
    from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k
    from data_engineering_copilot.services.query_signals import SearchMode

    async def _eval_dense(qs: list[dict]) -> dict:
        per_ndcg: list[float] = []
        per_recall: list[float] = []
        for item in qs:
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
                search_mode=SearchMode.DENSE_ONLY,
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

    dense_train = asyncio.run(_eval_dense(train))
    dense_held = asyncio.run(_eval_dense(held))
    print(f"\nDense train ndcg={dense_train['ndcg']:.4f} recall={dense_train['recall']:.4f}")
    print(f"Dense held  ndcg={dense_held['ndcg']:.4f} recall={dense_held['recall']:.4f}")

    # Validate best weights on held
    rrf_best = None if best_w == (1, 1) else best_w
    held_hybrid = asyncio.run(
        _eval_with_weights(service, held, embedded, rrf_best, top_k=args.k, fused_limit=best_prefetch)
    )
    print(f"Held hybrid best weights={best_tag} ndcg={held_hybrid['ndcg']:.4f} recall={held_hybrid['recall']:.4f}")

    # Also evaluate held for all weights to show table
    held_all: dict[tuple[float, float], dict] = {}
    for w in weight_grid:
        rrf_w = None if w == (1, 1) else w
        held_all[w] = asyncio.run(
            _eval_with_weights(service, held, embedded, rrf_w, top_k=args.k, fused_limit=best_prefetch)
        )

    print("\nHeld results (all weights):")
    for w, res in held_all.items():
        tag = "equal" if w == (1, 1) else str(w)
        print(f"  weights={tag:12s} ndcg={res['ndcg']:.4f} recall={res['recall']:.4f}")

    mean_delta, (ci_lo, ci_hi) = bootstrap_delta_ci(
        held_hybrid["per_ndcg"], dense_held["per_ndcg"], n_boot=1000, seed=42
    )
    includes_zero = ci_lo <= 0 <= ci_hi
    print(
        f"\nHeld Δ nDCG hybrid_best - dense: {mean_delta:+.4f} 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] includes_0={includes_zero}"
    )

    mean_delta_r, (ci_lo_r, ci_hi_r) = bootstrap_delta_ci(
        held_hybrid["per_recall"], dense_held["per_recall"], n_boot=1000, seed=42
    )
    print(f"Held Δ recall hybrid_best - dense: {mean_delta_r:+.4f} 95% CI [{ci_lo_r:+.4f}, {ci_hi_r:+.4f}]")

    # Compare best weights vs equal (1,1) on held
    held_equal = held_all[(1, 1)]
    mean_delta_w, (ci_lo_w, ci_hi_w) = bootstrap_delta_ci(
        held_hybrid["per_ndcg"], held_equal["per_ndcg"], n_boot=1000, seed=42
    )
    includes_zero_w = ci_lo_w <= 0 <= ci_hi_w
    print(
        f"Held Δ nDCG best_weights - equal: {mean_delta_w:+.4f} 95% CI [{ci_lo_w:+.4f}, {ci_hi_w:+.4f}] includes_0={includes_zero_w}"
    )

    # b sweep (query-side approx) if requested or auto-evaluate both b values
    b_results: dict[str, dict] = {}
    best_b = 0.75
    best_b_ndcg = held_hybrid["ndcg"]
    if args.b_sweep:
        print("\n--- b sweep (query-side approx, no re-upsert) ---")
        # b sweep uses best weights winner
        for b in [0.5, 0.75]:
            # mutate tokenizer b (query-side approx)
            bm25 = getattr(service.vector_store, "_bm25", None)  # type: ignore[attr-defined]
            orig_b = None
            if bm25 is not None:
                orig_b = bm25._b  # type: ignore[attr-defined]
                bm25._b = b  # type: ignore[attr-defined]
            res_train = asyncio.run(
                _eval_with_weights(service, train, embedded, rrf_best, top_k=args.k, fused_limit=best_prefetch)
            )
            res_held = asyncio.run(
                _eval_with_weights(service, held, embedded, rrf_best, top_k=args.k, fused_limit=best_prefetch)
            )
            b_results[str(b)] = {"train_ndcg": res_train["ndcg"], "held_ndcg": res_held["ndcg"]}
            print(f"  b={b} train ndcg={res_train['ndcg']:.4f} held ndcg={res_held['ndcg']:.4f}")
            bm25_after = getattr(service.vector_store, "_bm25", None)  # type: ignore[attr-defined]
            if bm25_after is not None and orig_b is not None:
                bm25_after._b = orig_b  # type: ignore[attr-defined]
        # pick best b on train
        if b_results:
            best_b_str = max(b_results, key=lambda kk: b_results[kk]["train_ndcg"])
            best_b = float(best_b_str)
            best_b_ndcg = b_results[best_b_str]["held_ndcg"]
            print(f"Best b on train: {best_b} held ndcg={best_b_ndcg:.4f}")
            print(
                "NOTE: b tuning without reindex is approximation; stored sparse vectors remain at b=0.75. Ship b!=0.75 only after full gen-build."
            )
    else:
        # still record baseline b=0.75 as evaluated
        b_results["0.75"] = {"train_ndcg": best_train["ndcg"], "held_ndcg": held_hybrid["ndcg"]}
        print(
            "\nSkipping b sweep (avg_len ~126 << 500, Task4 win +0.034 >0.02 — b≈0.75 optimal per literature). Use --b-sweep to force."
        )

    # Decision: ship only if best weights held Δ CI>0
    # Note: best_w may be (1,1); then Δ is 0
    ship_weights = (not includes_zero_w) and (mean_delta_w > 0) and (best_w != (1, 1))
    # For b, ship only if b sweep wins and held CI>0 (not implemented strict — keep 0.75)
    ship_b = False
    if args.b_sweep and best_b != 0.75:
        # compare best b vs 0.75 on held via bootstrap would be needed; keep conservative
        ship_b = False

    if ship_weights:
        print(f"\nDecision: ship weights={best_w} (held Δ vs equal CI excludes 0)")
    else:
        print("\nDecision: keep (1,1) b=0.75 — no weight holds Δ CI>0")

    # Restore k
    if hasattr(service.vector_store, "_hybrid_rrf_k"):
        service.vector_store._hybrid_rrf_k = 20  # type: ignore[attr-defined]
        # leave as best_k for future runs; ADR-006 says 20

    out = {
        "best_k": best_k,
        "best_prefetch": best_prefetch,
        "best_weights": list(best_w),
        "best_weights_tag": best_tag,
        "train_ndcg": best_train["ndcg"],
        "train_recall": best_train["recall"],
        "held_ndcg": held_hybrid["ndcg"],
        "held_recall": held_hybrid["recall"],
        "dense_train_ndcg": dense_train["ndcg"],
        "dense_held_ndcg": dense_held["ndcg"],
        "dense_train_recall": dense_train["recall"],
        "dense_held_recall": dense_held["recall"],
        "held_delta_ndcg_vs_dense": mean_delta,
        "held_ci_ndcg_vs_dense": [ci_lo, ci_hi],
        "held_delta_recall_vs_dense": mean_delta_r,
        "held_ci_recall_vs_dense": [ci_lo_r, ci_hi_r],
        "held_delta_ndcg_vs_equal": mean_delta_w,
        "held_ci_ndcg_vs_equal": [ci_lo_w, ci_hi_w],
        "held_ci_includes_zero_vs_equal": includes_zero_w,
        "ci_includes_zero_vs_dense": includes_zero,
        "ship_weights": ship_weights,
        "best_b": best_b,
        "b_results": b_results,
        "ship_b": ship_b,
        "b_note": "query-side approximation without re-upsert; full gen-build required to bake b",
        "grid_train": {f"{w[0]},{w[1]}": {"ndcg": v["ndcg"], "recall": v["recall"]} for w, v in train_results.items()},
        "grid_held": {f"{w[0]},{w[1]}": {"ndcg": v["ndcg"], "recall": v["recall"]} for w, v in held_all.items()},
        "k": args.k,
        "n_train": len(train),
        "n_held": len(held),
        "decision": "ship" if ship_weights else "keep (1,1) b=0.75",
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
