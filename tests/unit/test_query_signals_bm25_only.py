"""Task 3: api_lookup → HYBRID_DENSE_BIAS routing (sparse 0.006 drags RRF; ADR-012 updated).

Proves that api_lookup意图始终使用 HYBRID_DENSE_BIAS，即使 rewrite 漂移
dense_rank() → dense ranking。BM25_ONLY recall 0.006 hurts.
"""

from __future__ import annotations

from data_engineering_copilot.services.query_signals import SearchMode, classify_query_signals, select_search_mode


def test_api_lookup_routes_bm25_only():
    assert select_search_mode(intent="api_lookup", query="spark.sql.functions.col") == SearchMode.HYBRID_DENSE_BIAS


def test_rewrite_drift_still_bm25_only():
    # rewrite may change "dense_rank()" → "dense ranking" but original still api_lookup
    assert select_search_mode(intent="api_lookup", query="dense_rank() over window") == SearchMode.HYBRID_DENSE_BIAS


def test_api_lookup_with_signals_still_bm25_only():
    # Signal-based call path (used by _compute_search_mode) must also be hard HYBRID_DENSE_BIAS
    signals = classify_query_signals("dense_rank() over window")
    assert select_search_mode("api_lookup", signals) == SearchMode.HYBRID_DENSE_BIAS


def test_code_example_routes_bm25_only():
    assert select_search_mode(intent="code_example", query="spark.sql.functions.col") == SearchMode.HYBRID_DENSE_BIAS


def test_non_api_intents_not_forced_bm25():
    # factual/how_to must NOT be forced to BM25_ONLY
    assert select_search_mode(intent="factual", query="spark.sql.functions.col") != SearchMode.BM25_ONLY
    assert select_search_mode(intent="how_to", query="spark.sql.functions.col") != SearchMode.BM25_ONLY
