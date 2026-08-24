"""Hermetic tests for the generation-layer evaluation harness.

No live LLM/judge calls: a fake generator and fake judge are injected so the
scoring pipeline (prompt build -> generate -> parse -> aggregate -> gate) is
exercised deterministically.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.generation_eval import (
    FAITHFULNESS_GATE,
    RELEVANCE_GATE,
    RUBRIC_GATE,
    GenerationEvalReport,
    evaluate_generation,
    load_generation_dataset,
    score_faithfulness,
    score_relevance,
    score_rubric,
)


class FakeGenerator:
    async def generate(self, prompt: str) -> str:
        return (
            "Apache Spark is a unified analytics engine for large-scale data "
            "processing with APIs in Scala, Java, Python, and R."
        )


class FakeJudgeHigh:
    """Returns strong scores so all gates should pass."""

    async def generate(self, prompt: str) -> str:
        if "faithfulness grader" in prompt:
            return '{"score": 0.95}'
        if "answer relevance" in prompt:
            return '{"score": 0.88}'
        if "answer-quality judge" in prompt:
            return '{"score": 5}'
        return '{"score": 0.5}'


class FakeJudgeLow:
    """Returns weak scores so gates should fail."""

    async def generate(self, prompt: str) -> str:
        if "faithfulness grader" in prompt:
            return '{"score": 0.40}'
        if "answer relevance" in prompt:
            return '{"score": 0.50}'
        if "answer-quality judge" in prompt:
            return '{"score": 2}'
        return '{"score": 0.0}'


def test_parse_score_clamps_and_extracts():
    from data_engineering_copilot.evaluation.generation_eval import _parse_score

    assert _parse_score('{"score": 0.9}', 0.0, 1.0) == 0.9
    assert _parse_score("some text 3.2 more", 1.0, 5.0) == 3.2
    # clamps out-of-range
    assert _parse_score('{"score": 9}', 0.0, 1.0) == 1.0
    assert _parse_score('{"score": -1}', 0.0, 1.0) == 0.0
    # no-match is None (distinct from a legit zero score)
    assert _parse_score("", 0.0, 1.0) is None


def test_load_generation_dataset_reads_frozen_rows():
    rows = load_generation_dataset("tests/evaluation/eval_dataset.jsonl")
    assert rows, "dataset should be non-empty"
    first = rows[0]
    assert first.question
    assert first.contexts  # frozen gold context present
    assert first.ground_truth


@pytest.mark.asyncio
async def test_evaluate_generation_passes_gates_with_strong_judge():
    report = await evaluate_generation(
        "tests/evaluation/eval_dataset.jsonl",
        None,  # settings unused when generator/judge injected
        generator=FakeGenerator(),
        judge=FakeJudgeHigh(),
        n_trials=3,
    )
    assert isinstance(report, GenerationEvalReport)
    assert report.faithfulness_mean == pytest.approx(0.95)
    assert report.relevance_mean == pytest.approx(0.88)
    assert report.rubric_mean == pytest.approx(5.0)
    assert report.passed is True


@pytest.mark.asyncio
async def test_evaluate_generation_fails_gates_with_weak_judge():
    report = await evaluate_generation(
        "tests/evaluation/eval_dataset.jsonl",
        None,
        generator=FakeGenerator(),
        judge=FakeJudgeLow(),
        n_trials=1,
    )
    assert report.faithfulness_mean < FAITHFULNESS_GATE
    assert report.relevance_mean < RELEVANCE_GATE
    assert report.rubric_mean < RUBRIC_GATE
    assert report.passed is False


@pytest.mark.asyncio
async def test_score_functions_parse_judge_output():
    judge = FakeJudgeHigh()
    faith = await score_faithfulness(judge, "q", "a", ["ctx"], n_trials=1)
    rel = await score_relevance(judge, "q", "a", n_trials=1)
    rubric = await score_rubric(judge, "q", "a", "gold", ["ctx"], n_trials=3)
    assert faith == pytest.approx(0.95)
    assert rel == pytest.approx(0.88)
    # rubric averaged over 3 trials, all returning 5
    assert rubric == pytest.approx(5.0)


def test_report_markdown_includes_gates():
    report = GenerationEvalReport(
        rows=[{"id": "x", "faithfulness": 0.9, "relevance": 0.85, "rubric": 4.5}],
        faithfulness_mean=0.9,
        relevance_mean=0.85,
        rubric_mean=4.5,
        passed=True,
    )
    md = report.to_markdown()
    assert "Faithfulness" in md
    assert "0.900" in md
    assert "Passed gates" in md
    assert "True" in md


class _StubJudge:
    """Judge double returning a fixed rubric score."""

    def __init__(self, score: float) -> None:
        self.score = score

    async def generate(self, prompt: str, **kwargs: object) -> str:
        import json as _json

        return _json.dumps({"score": self.score, "reason": "stub"})


class _StubGenerator:
    async def generate(self, prompt: str, **kwargs: object) -> str:
        return "Grounded answer restating the context."


def _write_dataset(tmp_path, rows=3):
    import json as _json

    path = tmp_path / "gen_eval.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(rows):
            fh.write(
                _json.dumps(
                    {
                        "id": f"row-{i}",
                        "question": f"What is topic {i}?",
                        "contexts": ["Context sentence for the topic."],
                        "ground_truth": "Gold answer.",
                    }
                )
                + "\n"
            )
    return str(path)


@pytest.mark.asyncio
async def test_dual_judge_agreement_reported(tmp_path):
    """Task 7: judge_b within ±1 of judge_a on every row -> agreement 1.0."""
    from data_engineering_copilot.evaluation.generation_eval import evaluate_generation

    dataset = _write_dataset(tmp_path)
    report = await evaluate_generation(
        dataset,
        generator=_StubGenerator(),
        judge=_StubJudge(4.0),
        judge_b=_StubJudge(5.0),
        n_trials=1,
    )
    assert report.judge_agreement == 1.0
    assert "judge_agreement" in report.to_dict()


@pytest.mark.asyncio
async def test_dual_judge_disagreement_detected(tmp_path):
    from data_engineering_copilot.evaluation.generation_eval import evaluate_generation

    dataset = _write_dataset(tmp_path)
    report = await evaluate_generation(
        dataset,
        generator=_StubGenerator(),
        judge=_StubJudge(1.0),
        judge_b=_StubJudge(5.0),
        n_trials=1,
    )
    assert report.judge_agreement == 0.0


@pytest.mark.asyncio
async def test_no_second_judge_leaves_agreement_none(tmp_path):
    from data_engineering_copilot.evaluation.generation_eval import evaluate_generation

    dataset = _write_dataset(tmp_path)
    report = await evaluate_generation(
        dataset,
        generator=_StubGenerator(),
        judge=_StubJudge(4.0),
        n_trials=1,
    )
    assert report.judge_agreement is None


class TestRowConcurrency:
    async def test_concurrent_matches_serial_and_preserves_order(self, tmp_path):
        import asyncio

        from data_engineering_copilot.evaluation.generation_eval import evaluate_generation

        dataset = _write_dataset(tmp_path, rows=6)

        async def run(conc):
            return await evaluate_generation(
                str(dataset),
                settings=None,
                generator=FakeGenerator(),
                judge=FakeJudgeHigh(),
                n_trials=1,
                row_concurrency=conc,
            )

        serial, parallel = await asyncio.gather(run(1), run(4))
        assert [r["id"] for r in parallel.rows] == [r["id"] for r in serial.rows]
        assert parallel.faithfulness_mean == pytest.approx(serial.faithfulness_mean)
        assert parallel.rubric_mean == pytest.approx(serial.rubric_mean)
        assert parallel.passed == serial.passed

    async def test_semaphore_actually_overlaps_rows(self, tmp_path):
        import asyncio

        from data_engineering_copilot.evaluation.generation_eval import evaluate_generation

        state = {"in_flight": 0, "max_seen": 0}

        class SlowGen:
            async def generate(self, prompt: str) -> str:
                state["in_flight"] += 1
                state["max_seen"] = max(state["max_seen"], state["in_flight"])
                await asyncio.sleep(0.05)
                state["in_flight"] -= 1
                return "answer"

        dataset = _write_dataset(tmp_path, rows=6)
        await evaluate_generation(
            str(dataset),
            settings=None,
            generator=SlowGen(),
            judge=FakeJudgeHigh(),
            n_trials=1,
            row_concurrency=3,
        )
        assert state["max_seen"] >= 2


class TestMajorityJudges:
    async def test_median_suppresses_outlier_and_reports_votes(self, tmp_path):
        from data_engineering_copilot.evaluation.generation_eval import evaluate_generation

        class JudgeMid:
            model_id = "mid"

            async def generate(self, prompt: str) -> str:
                if "faithfulness" in prompt:
                    return '{"score": 0.70}'
                if "relevance" in prompt:
                    return '{"score": 0.60}'
                return '{"score": 4}'

        class JudgeOutlier:
            model_id = "outlier"

            async def generate(self, prompt: str) -> str:
                return (
                    '{"score": 0.10}'
                    if "faithfulness" in prompt
                    else ('{"score": 1}' if "answer-quality judge" in prompt else '{"score": 0.10}')
                )

        dataset = _write_dataset(tmp_path, rows=2)
        report = await evaluate_generation(
            str(dataset),
            settings=None,
            generator=FakeGenerator(),
            judges=[FakeJudgeHigh(), JudgeMid(), JudgeOutlier()],
            n_trials=1,
        )
        # votes per row: faith [.95,.70,.10] -> median .70 ; rubric [5,4,1] -> 4
        assert report.rows[0]["judge_votes"]
        assert len(report.rows[0]["judge_votes"]) == 3
        assert report.faithfulness_mean == pytest.approx(0.70)
        assert report.rubric_mean == pytest.approx(4.0)
        # pairwise rubric agreement: (5,4)->1, (4,1)->0, (5,1)->0 => 1/3
        assert report.judge_agreement == pytest.approx(1 / 3)
