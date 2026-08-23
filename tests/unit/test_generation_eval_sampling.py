from data_engineering_copilot.evaluation.generation_eval import (
    GenerationEvalRow,
    stratified_sample,
)


def _rows():
    return [
        GenerationEvalRow(
            question=f"q{i}", contexts=["c"], ground_truth="g", id=f"r{i:03d}", intent=("a" if i % 2 else "b")
        )
        for i in range(10)
    ]


def test_stratified_round_robin_deterministic():
    rows = stratified_sample(_rows(), 6, lambda r: r.intent)
    assert [r.id for r in rows] == sorted(r.id for r in rows) or len(rows) == 6
    assert len(rows) == 6
    intents = {r.intent for r in rows}
    assert intents == {"a", "b"}


def test_stratified_is_deterministic():
    a = [r.id for r in stratified_sample(_rows(), 6, lambda r: r.intent)]
    b = [r.id for r in stratified_sample(_rows(), 6, lambda r: r.intent)]
    assert a == b


def test_sample_n_zero_returns_empty():
    assert stratified_sample(_rows(), 0, lambda r: r.intent) == []


def test_sample_n_exceeding_total_returns_all():
    assert len(stratified_sample(_rows(), 99, lambda r: r.intent)) == 10
