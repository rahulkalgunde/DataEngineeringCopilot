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
    assert _parse_score("", 0.0, 1.0) == 0.0


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
