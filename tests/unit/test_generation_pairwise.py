from data_engineering_copilot.evaluation.generation_eval import pairwise_verdict


def test_both_orderings_prefer_a():
    """A scores higher in BOTH orderings → A wins."""
    assert pairwise_verdict(0.9, 0.4, 0.9, 0.4) == "A"


def test_both_orderings_prefer_b():
    """B scores higher in BOTH orderings → B wins."""
    assert pairwise_verdict(0.4, 0.9, 0.4, 0.9) == "B"


def test_mixed_is_tie():
    """A wins ord1, B wins ord2 (position bias) → TIE."""
    assert pairwise_verdict(0.9, 0.4, 0.4, 0.9) == "TIE"


def test_exact_equal_scores_are_tie():
    assert pairwise_verdict(0.5, 0.5, 0.5, 0.5) == "TIE"
