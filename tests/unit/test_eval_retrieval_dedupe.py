from data_engineering_copilot.cli import _eval_retrieval_row
from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k


def test_recall_dedupes_duplicate_urls():
    # 7 chunks from the same expected page must count ONCE
    row = _eval_retrieval_row("q", "how_to", ["https://x/a"], ["https://x/a"] * 7, 10)
    assert row["recall"] == 1.0


def test_precision_dedupes_duplicate_urls():
    row = _eval_retrieval_row("q", "how_to", ["https://x/a"], ["https://x/a"] * 7 + ["https://y/b"], 10)
    assert row["precision"] == 0.1  # 1 unique hit / k=10


def test_ndcg_dedupes_duplicate_urls():
    dup = ["https://x/a"] * 5
    assert ndcg_at_k(dup, ["https://x/a"], 10) == 1.0


def test_recall_module_matches_row_semantics():
    got = ["u1", "u1", "u2"]
    assert recall_at_k(got, ["u1", "u2"], 3) == 1.0
