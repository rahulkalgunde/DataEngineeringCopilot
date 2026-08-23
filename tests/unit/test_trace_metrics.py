from data_engineering_copilot.evaluation.trace_metrics import (
    _token_overlap,
    chunk_is_used,
    trace_completeness,
    trace_utilization,
)


def _chunks():
    return [
        {"text": "spark dataframe api reference select filter", "cited": True},
        {"text": "unrelated storage pricing details here", "cited": False},
    ]


def test_token_overlap_bounds():
    ans = "use the spark dataframe select api"
    assert _token_overlap(_chunks()[0]["text"], ans) > 0.3
    assert _token_overlap(_chunks()[1]["text"], ans) < 0.3


def test_chunk_used_via_citation_or_overlap():
    ans = "use the spark dataframe select api"
    assert chunk_is_used(_chunks()[0], ans) is True
    assert chunk_is_used(_chunks()[1], ans) is False


def test_utilization_fraction():
    ans = "use the spark dataframe select api"
    util = trace_utilization(_chunks(), ans)
    assert util > 0.0
    assert util <= 1.0


def test_utilization_equal_length_chunks():
    chunks = [
        {"text": "a b c", "cited": True},
        {"text": "d e f", "cited": False},
    ]
    assert abs(trace_utilization(chunks, "a b c") - 0.5) < 1e-9


def test_utilization_empty_inputs():
    assert trace_utilization([], "ans") == 0.0
    assert trace_utilization(_chunks(), "") == 0.0


def test_completeness_needs_relevant_marks():
    chunks = [
        {"text": "gold fact one", "relevant": True, "used": True},
        {"text": "gold fact two", "relevant": True, "used": False},
        {"text": "filler", "relevant": False},
    ]
    assert abs(trace_completeness(chunks) - 0.5) < 1e-9
    assert trace_completeness([{"text": "x"}]) == 0.0
