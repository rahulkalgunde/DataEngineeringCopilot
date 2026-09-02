#!/usr/bin/env python3
"""Pipeline downstream ablation: guardrails / sibling / dedup (ADR-011).

Grid 110/110 held vs train with vs without per stage, Δ recall/nDCG + 95% CI
bootstrap 1000 (same as scripts/tune_rrf_k.py). Toggles via monkeypatched
settings before building AsyncRagService:

- sibling: assembly_enable_sibling_merge True vs False + patch _rejoin_sibling_chunks max_blocks 3→0
- guardrails: input_guardrails_enabled True vs False (InputGuardrails(enabled=False)/None)
- dedup: assembly_content_hash_dedup + context_compression_enabled True vs False

Reuses embedding cache via CachedEmbedder (batch-size 55). Emits
data/pipeline_ablation.json with
{"guardrails": {"with":..., "without":..., "delta":..., "ci":[lo,hi]}, ...}
+ decision ship = ci excludes 0 and delta>0 else keep off.

Run: dec_venv/bin/python scripts/eval_pipeline_ablation.py --k 10 --split held
"""

from __future__ import annotations

import argparse
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


def _build_service_for_stage(stage: str, enabled: bool):
    """Build AsyncRagService with toggle for *stage*.

    Monkeypatches settings before building, matching plan: toggle via
    monkeypatch settings before building AsyncRagService, sibling max_blocks 3→0,
    input_guardrails_enabled False, dedup False.
    """
    from data_engineering_copilot.cli import _disable_rewrites_for_eval
    from data_engineering_copilot.config.settings import settings as _settings
    from data_engineering_copilot.factory import build_rag_service

    overrides: dict[str, object] = {}
    if stage == "guardrails":
        overrides["input_guardrails_enabled"] = enabled
    elif stage == "sibling":
        overrides["assembly_enable_sibling_merge"] = enabled
    elif stage == "dedup":
        overrides["assembly_content_hash_dedup"] = enabled
        overrides["context_compression_enabled"] = enabled
    else:
        raise ValueError(f"unknown stage {stage!r}")

    base = _settings
    new_settings = base.model_copy(update=overrides)
    svc = build_rag_service(app_settings=new_settings)
    _disable_rewrites_for_eval(svc)

    if stage == "sibling" and not enabled:

        async def _noop_rejoin(chunks, max_sibling_blocks: int = 0):  # type: ignore[no-untyped-def]
            return chunks

        svc._rejoin_sibling_chunks = _noop_rejoin  # type: ignore[method-assign,assignment]

    if stage == "guardrails" and not enabled:
        svc.input_guardrails = None  # type: ignore[assignment]

    return svc


async def _eval_service(
    svc,
    queries: list[dict],
    k: int,
    batch_size: int | None,
) -> dict[str, object]:
    from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k

    per_recall: list[float] = []
    per_ndcg: list[float] = []

    if batch_size is not None and batch_size > 0:
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]

            async def _single(item: dict) -> tuple[float, float]:
                q = item.get("question") or ""
                expected = [u for u in (item.get("expected_urls") or []) if u]
                if not q:
                    return 0.0, 0.0
                try:
                    ans = await svc.answer(
                        q, provenance=None, bypass_cache=True, retrieval_only=True, expected_urls=expected
                    )
                    urls = [c.url for c in ans.sources]
                    return recall_at_k(urls, expected, k), ndcg_at_k(urls, expected, k)
                except Exception:
                    return 0.0, 0.0

            rows = await asyncio.gather(*[_single(it) for it in batch])
            for r, n in rows:
                per_recall.append(r)
                per_ndcg.append(n)
            if start + batch_size < len(queries):
                await asyncio.sleep(0.05)
    else:
        for item in queries:
            q = item.get("question") or ""
            expected = [u for u in (item.get("expected_urls") or []) if u]
            if not q:
                per_recall.append(0.0)
                per_ndcg.append(0.0)
                continue
            try:
                ans = await svc.answer(
                    q, provenance=None, bypass_cache=True, retrieval_only=True, expected_urls=expected
                )
                urls = [c.url for c in ans.sources]
                per_recall.append(recall_at_k(urls, expected, k))
                per_ndcg.append(ndcg_at_k(urls, expected, k))
            except Exception:
                per_recall.append(0.0)
                per_ndcg.append(0.0)

    return {
        "recall": statistics.fmean(per_recall) if per_recall else 0.0,
        "ndcg": statistics.fmean(per_ndcg) if per_ndcg else 0.0,
        "per_recall": per_recall,
        "per_ndcg": per_ndcg,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Pipeline ablation guardrails/sibling/dedup with holdout CI (ADR-011)")
    ap.add_argument("--dataset", default="tests/evaluation/golden/recall_inscope.jsonl")
    ap.add_argument("--k", type=int, default=10, help="top_k for recall/nDCG")
    ap.add_argument("--split", choices=["train", "held", "all"], default="held", help="holdout split (default: held)")
    ap.add_argument("--batch-size", type=int, default=55, help="batch size (default: 55)")
    ap.add_argument("--stage", choices=["guardrails", "sibling", "dedup", "all"], default="all", help="stage to ablate")
    ap.add_argument("--output", default="data/pipeline_ablation.json")
    args = ap.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    if not dataset_path.exists():
        alt = PROJECT_ROOT / "tests/evaluation/golden/recall_all.jsonl"
        if alt.exists():
            dataset_path = alt
        else:
            print(f"❌ dataset not found: {dataset_path}", file=sys.stderr)
            return 2

    queries = _load_queries(dataset_path)
    if len(queries) != 220:
        print(f"⚠️ expected 220 queries, got {len(queries)}")

    from data_engineering_copilot.evaluation.retrieval import (
        PIPELINE_ABLATION_STAGES,
        bootstrap_delta_ci,
        pipeline_stage_decision,
        split_queries,
    )

    train, held = split_queries(queries, seed=42)
    if args.split == "train":
        selected = train
        label = "train (110)"
    elif args.split == "held":
        selected = held
        label = "held (110)"
    else:
        selected = queries
        label = f"all ({len(queries)})"

    stages: tuple[str, ...] = PIPELINE_ABLATION_STAGES if args.stage == "all" else (args.stage,)  # type: ignore[assignment]

    print(
        f"Dataset {dataset_path.name}: total={len(queries)} train={len(train)} held={len(held)} selected={len(selected)} [{label}] seed=42"
    )
    print(f"k={args.k} stages={list(stages)} batch_size={args.batch_size}")

    results: dict[str, dict] = {}

    async def _run() -> None:
        for st in stages:
            print(f"\n— Stage {st}: with vs without —")
            svc_with = _build_service_for_stage(st, True)
            svc_without = _build_service_for_stage(st, False)
            # Embed cache is per-service but CachedEmbedder reuses Redis; first pass populates.
            with_res = await _eval_service(svc_with, selected, args.k, args.batch_size)
            without_res = await _eval_service(svc_without, selected, args.k, args.batch_size)

            delta_r, ci_r = bootstrap_delta_ci(with_res["per_recall"], without_res["per_recall"], n_boot=1000, seed=13)  # type: ignore[arg-type]
            delta_n, ci_n = bootstrap_delta_ci(with_res["per_ndcg"], without_res["per_ndcg"], n_boot=1000, seed=13)  # type: ignore[arg-type]

            dec_r = pipeline_stage_decision(delta_r, ci_r)
            dec_n = pipeline_stage_decision(delta_n, ci_n)
            ship = bool(dec_r["ship"] or dec_n["ship"])

            print(
                f"  {st:10s} with recall={with_res['recall']:.4f} ndcg={with_res['ndcg']:.4f} "
                f"vs without recall={without_res['recall']:.4f} ndcg={without_res['ndcg']:.4f} "
                f"Δ recall={delta_r:+.4f} CI [{ci_r[0]:+.4f},{ci_r[1]:+.4f}] excludes_0={dec_r['excludes_zero']} "
                f"Δ ndcg={delta_n:+.4f} CI [{ci_n[0]:+.4f},{ci_n[1]:+.4f}] -> {dec_r['decision']}"
            )
            results[st] = {
                "with": with_res["recall"],
                "without": without_res["recall"],
                "delta": delta_r,
                "ci": [ci_r[0], ci_r[1]],
                "with_ndcg": with_res["ndcg"],
                "without_ndcg": without_res["ndcg"],
                "delta_ndcg": delta_n,
                "ci_ndcg": [ci_n[0], ci_n[1]],
                "ndcg_with": with_res["ndcg"],
                "ndcg_without": without_res["ndcg"],
                "ship": ship,
                "decision": dec_r["decision"],
                "excludes_zero": dec_r["excludes_zero"],
                # verbose
                "with_recall": with_res["recall"],
                "without_recall": without_res["recall"],
            }

    asyncio.run(_run())

    print("\n" + "=" * 72)
    print(f"Pipeline ablation summary k={args.k} split={args.split} n={len(selected)}")
    for st in stages:
        r = results[st]
        print(
            f"  {st:10s} with {r['with']:.3f}/{r['with_ndcg']:.3f} "
            f"without {r['without']:.3f}/{r['without_ndcg']:.3f} "
            f"Δ {r['delta']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] decision={r['decision']}"
        )

    out: dict[str, object] = {"k": args.k, "split": args.split, "n_selected": len(selected), **results}
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("\nJSON:")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
