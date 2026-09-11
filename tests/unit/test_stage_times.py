def _mk():
    # 100s is t0; each stage lands 5s later than the previous.
    return {"query_rewrite": 105.0, "retrieval": 110.0, "rerank": 115.0, "generation": 120.0}, 100.0


def test_summarize_stage_times_computes_deltas():
    from data_engineering_copilot.services.async_rag import _summarize_stage_times

    marks, t0 = _mk()
    times = _summarize_stage_times(marks, t0, now=120.0)
    assert set(times) == {"query_rewrite", "retrieval", "rerank", "generation", "total"}
    assert times["query_rewrite"] == 5.0
    assert times["retrieval"] == 5.0
    assert times["rerank"] == 5.0
    assert times["generation"] == 5.0
    assert times["total"] == 20.0


def test_summarize_stage_times_defaults_missing_mark_to_now():
    from data_engineering_copilot.services.async_rag import _summarize_stage_times

    times = _summarize_stage_times({"query_rewrite": 105.0}, 100.0, now=150.0)
    assert times["retrieval"] == 45.0  # missing mark falls back to now
    assert times["total"] == 50.0
