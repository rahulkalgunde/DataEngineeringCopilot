import json

from data_engineering_copilot.evaluation.candidate_pool import load_pool, rank_from_pool, save_pool


def _pool():
    return {
        "q1": [
            {"url": "u2", "dense_score": 0.9, "sparse_score": 0.1, "fused_score": 0.5},
            {"url": "u1", "dense_score": 0.5, "sparse_score": 0.9, "fused_score": 0.8},
        ]
    }


def test_roundtrip(tmp_path):
    p = tmp_path / "pool.json"
    save_pool(p, _pool())
    assert load_pool(p) == _pool()


def test_load_missing_returns_empty(tmp_path):
    assert load_pool(tmp_path / "nope.json") == {}


def test_rank_by_fused_desc_and_cutoff(tmp_path):
    ranked = rank_from_pool(_pool()["q1"], 1)
    assert ranked == ["u1"]


def test_save_creates_parent_dirs(tmp_path):
    p = tmp_path / "sub" / "pool.json"
    save_pool(p, _pool())
    assert json.loads(p.read_text()) == _pool()
