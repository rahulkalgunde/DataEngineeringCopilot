import json

import pytest

from data_engineering_copilot.evaluation.majority_label import label_row, majority_vote


def test_majority_2_of_1_wins():
    assert majority_vote([1, 1, 0]) == (1, "majority")
    assert majority_vote([0, 0, 1]) == (0, "majority")


def test_unanimous():
    assert majority_vote([1, 1, 1]) == (1, "unanimous")
    assert majority_vote([0, 0, 0]) == (0, "unanimous")


def test_even_length_ties_break_to_zero():
    # Not used in production (3 voters), but pinned for determinism.
    assert majority_vote([1, 0]) == (0, "majority")
    with pytest.raises(ValueError):
        majority_vote([])
    with pytest.raises(ValueError):
        majority_vote([2, 0, 1])


class _FakeJudge:
    def __init__(self, score: str):
        self.score = score
        self.calls = 0

    async def generate(self, _prompt: str) -> str:
        self.calls += 1
        return self.score


async def test_label_row_majority_across_judges():
    judges = [_FakeJudge('{"score": 1}'), _FakeJudge('{"score": 0.9}'), _FakeJudge('{"score": 0}')]
    row = {"question": "q", "contexts": ["c"], "answer": "a"}
    out = await label_row(row, judges)
    assert out["human_faithfulness"] == 1
    assert out["needs_label"] is False
    assert out["label_note"] == "llm_majority_3way_majority"


async def test_label_row_unanimous_zero_single_call_per_judge():
    judges = [_FakeJudge('{"score": 0}') for _ in range(3)]
    row = {"question": "q", "contexts": ["c"], "answer": "insufficient information."}
    out = await label_row(row, judges)
    assert out["human_faithfulness"] == 0
    assert out["label_note"] == "llm_majority_3way_unanimous"
    assert all(j.calls == 1 for j in judges)  # legit zero must not retry


async def test_label_dataset_in_place(tmp_path):
    from data_engineering_copilot.evaluation.majority_label import label_dataset

    ds = tmp_path / "cal.jsonl"
    rows = [
        {"id": "r1", "question": "q", "contexts": ["c"], "answer": "a"},
        {"id": "r2", "question": "q", "contexts": ["c"], "answer": "a"},
    ]
    ds.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    judges = [_FakeJudge('{"score": 1}'), _FakeJudge("1"), _FakeJudge("0")]
    summary = await label_dataset(str(ds), judges=judges)
    assert summary == {"rows": 2, "faithful": 2, "unanimous": 0}
    saved = [json.loads(line) for line in ds.read_text().splitlines() if line.strip()]
    assert all(not r["needs_label"] for r in saved)
