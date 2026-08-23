from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, percentile, recall_at_k


def test_ndcg_perfect_ranking():
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0


def test_ndcg_hand_computed_single_hit_at_rank2():
    import math

    got = ndcg_at_k(["x", "a"], ["a"], 2)
    assert abs(got - 1 / math.log2(3)) < 1e-9


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], ["a"], 2) == 0.0


def test_recall_at_k_counts_expected_fraction():
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], 3) == 2 / 3
    assert recall_at_k(["a"], ["a", "b"], 1) == 0.5


def test_percentile_endpoints_and_median():
    assert percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0], 1.0) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
