"""Frozen dark flags — ADR-010.

All experimental retrieval/pipeline flags stay False/rrf until
store recall@10 ≥0.35 on held 110 (seed 42) with CI excludes 0.
Provenance: RRF k=20 L100 shipped 74236d9 — ADR-010.
"""

from tests.conftest import make_settings


def test_dark_flags_frozen_until_recall_035() -> None:
    s = make_settings()
    assert s.namespace_bm25_enabled is False
    assert s.identifier_sparse_rrf_enabled is False
    assert s.retrieval_fusion == "rrf"
    assert s.llm_rerank_enabled is False
    assert s.context_compression_enabled is False
    # threshold documented in ADR-010
    assert s.retrieval_prefetch_limit == 100  # shipped k20 win stays, but no new flags
    # shipped tuning stays frozen — RRF k=20 L100 from 74236d9
    assert s.hybrid_rrf_k == 20
    assert s.retrieval_fusion == "rrf"  # DBSF dark until gate
