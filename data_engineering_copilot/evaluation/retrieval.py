"""Ablation harness helpers for dense / sparse / hybrid evaluation.

Provides deterministic holdout split and bootstrap CI for hybrid vs best-single.
Used by ``dec eval-retrieval --ablation`` (cli.py) and the scripts gate.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence

from data_engineering_copilot.evaluation.stats import _percentile_rank


def split_queries(
    queries: Sequence[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Deterministic 50/50 shuffle split (seed=42).

    Mirrors plan Task 3: 220q baseline_inscope → 110 train / 110 held-out.
    When n is odd, train gets the larger half.
    """
    qs = list(queries)
    rnd = random.Random(seed)
    rnd.shuffle(qs)
    mid = (len(qs) + 1) // 2
    # Plan expects exactly 110/110 for 220; keep generic for other sizes.
    if len(qs) == 220:
        return qs[:110], qs[110:]
    return qs[:mid], qs[mid:]


def bootstrap_delta_ci(
    hybrid_scores: Sequence[float],
    best_single_scores: Sequence[float],
    n_boot: int = 1000,
    seed: int = 13,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap 95% CI for mean(hybrid - best_single).

    Paired bootstrap: same resampled indices applied to both lists.
    Returns (mean_delta, (ci_low, ci_high)).
    """
    if not hybrid_scores or not best_single_scores:
        return 0.0, (0.0, 0.0)
    n = min(len(hybrid_scores), len(best_single_scores))
    h = list(hybrid_scores)[:n]
    b = list(best_single_scores)[:n]
    deltas = [hv - bv for hv, bv in zip(h, b, strict=False)]
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    # bootstrap mean deltas
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.fmean(sample))
    lo = _percentile_rank(boot_means, 0.025)
    hi = _percentile_rank(boot_means, 0.975)
    return mean_delta, (lo, hi)


def per_query_best(recalls_dense: Sequence[float], recalls_sparse: Sequence[float]) -> list[float]:
    """Per-query best-single recall (max of dense / sparse)."""
    return [max(d, s) for d, s in zip(recalls_dense, recalls_sparse, strict=False)]


# Search mode loop descriptor for cli ablation (keeps cli import-light).
ABLATION_MODES = ("dense", "sparse", "hybrid_rrf_k60", "hybrid_rrf_k5", "dbsf")

# Reranker as 2nd stage (Task 6): evaluate CrossEncoder on fused top 50 -> top 10
# vs fused top 10. Model cross-encoder/ms-marco-MiniLM-L-6-v2 locally, gated by
# availability; skip if not installed (fail-open). p95 budget <250ms for k=10.
RERANK_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_FUSED_POOL = 50
RERANK_TOP_K = 10
RERANK_P95_BUDGET_MS = 250


def is_cross_encoder_available() -> bool:
    """Return True iff sentence_transformers CrossEncoder can be imported.

    Fail-open: absence skips rerank evaluation without error.
    """
    try:
        import sentence_transformers  # noqa: F401
        from sentence_transformers import CrossEncoder  # noqa: F401

        return True
    except Exception:
        return False


def rerank_fused_with_cross_encoder(
    query: str,
    fused_chunks: Sequence[object],
    top_k: int = RERANK_TOP_K,
    model_name: str = RERANK_CROSS_ENCODER_MODEL,
) -> tuple[list[object], float] | None:
    """Rerank fused top 50 -> top_k with CrossEncoder if available.

    Returns (reranked_top_k, latency_ms) or None when CrossEncoder is not
    installed. Candidates are (query, chunk.text) pairs scored by the
    cross-encoder. Timetaken includes model inference only.

    Fail-open: ImportError/ runtime errors return None — caller falls back to
    fused top_k.
    """
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    import time

    if not fused_chunks:
        return [], 0.0
    # Lazy load per call; caller may cache CrossEncoder externally for batch runs.
    try:
        ce = CrossEncoder(model_name)
    except Exception:
        return None
    # Prepare pairs
    texts: list[str] = []
    for c in fused_chunks:  # type: ignore[assignment]
        # RetrievedChunk has .chunk.text; fallback to str(c)
        try:
            texts.append(c.chunk.text)  # type: ignore[union-attr]
        except Exception:
            texts.append(str(getattr(c, "text", str(c))))
    pairs = [(query, t) for t in texts]
    t0 = time.perf_counter()
    try:
        scores = ce.predict(pairs)  # type: ignore[call-arg]
    except Exception:
        return None
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # Pair scores with chunks and sort descending
    scored = sorted(zip(fused_chunks, scores, strict=False), key=lambda x: float(x[1]), reverse=True)
    reranked = [c for c, _ in scored[:top_k]]
    return reranked, latency_ms


def evaluate_rerank_gain(
    ndcg_fused: Sequence[float],
    ndcg_reranked: Sequence[float],
    latencies_ms: Sequence[float],
    p95_budget_ms: float = RERANK_P95_BUDGET_MS,
) -> dict[str, object]:
    """Compare reranked top 10 vs fused top 10 on held set.

    Returns dict with delta, CI, p95, and gated decision (ship only if
    Δ NDCG CI>0 and p95 < budget). Mirrors Task 6 reranker slice.
    """
    delta, ci = bootstrap_delta_ci(list(ndcg_reranked), list(ndcg_fused))
    p95 = _percentile_rank(list(latencies_ms), 0.95) if latencies_ms else 0.0
    ship = bool(ci[0] > 0 and p95 < p95_budget_ms)
    return {
        "delta_ndcg": delta,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p95_ms": p95,
        "p95_budget_ms": p95_budget_ms,
        "ship": ship,
    }
