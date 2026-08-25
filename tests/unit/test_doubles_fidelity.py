"""Fidelity contracts for test doubles: output must derive from input.

Every entry pins a paired-call contract `(input_a -> out_a, input_b -> out_b)`
mirroring what the REAL component does. A double that returns constants while
the real contract is input-dependent is a lying double — it lets integration
defects pass green (see 2026-08-24 ragas reference-column incident).

Rules:
- New production-relevant doubles get an entry here in the same commit as the
  consumer that depends on their input-sensitivity.
- Constant-output doubles are exempt only with a comment stating why the real
  contract is genuinely constant.
"""

from __future__ import annotations

import json

import pytest

from data_engineering_copilot.services.ragas_evaluation import RagasEvaluator


class _RecordingJudge:
    """Judge double whose scores derive from prompt keywords — mirrors the
    real rubric/faithfulness/relevance graders' input-sensitive behavior."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    async def generate(self, prompt: str) -> str:
        for needle, score in self._mapping.items():
            if needle in prompt:
                return json.dumps({"score": score})
        raise AssertionError(f"unrecognized prompt shape — double must not guess: {prompt[:80]}")


@pytest.mark.unit
class TestJudgeDoubleFidelity:
    async def test_scores_track_prompt_kind(self):
        judge = _RecordingJudge({"faithfulness grader": "0.9", "answer relevance": "0.6", "answer-quality judge": "4"})
        faith = await judge.generate("You are a faithfulness grader. Context: x Answer: y")
        rel = await judge.generate("Rate answer relevance. Q: q A: a")
        rub = await judge.generate("You are an answer-quality judge. Rate 1-5.")
        assert (
            float(json.loads(faith)["score"]),
            float(json.loads(rel)["score"]),
            float(json.loads(rub)["score"]),
        ) == (
            0.9,
            0.6,
            4,
        )

    async def test_unrecognized_prompt_fails_loud(self):
        judge = _RecordingJudge({"faithfulness grader": "0.9"})
        with pytest.raises(AssertionError, match="must not guess"):
            await judge.generate("totally different prompt")


@pytest.mark.unit
class TestRagasResultSelectionFidelity:
    """The ragas evaluate() double must report ONLY the metrics it ran —
    mirroring real ragas, which omits unselected metrics (KeyError path)."""

    async def test_result_reflects_selected_metrics(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from tests.unit.test_ragas_evaluation import MOCK_METRICS

        ev = RagasEvaluator()
        _ = {}  # captured unused but required by closure

        class NamedMetric:
            def __init__(self, name: str):
                self.name = name
                self.llm = None
                self.embeddings = None

        all_metrics = [NamedMetric(n) for n in MOCK_METRICS]

        def fake_evaluate(dataset, metrics=None, llm=None, **kwargs):
            selected = {m.name for m in (metrics or [])}
            result = MagicMock()
            result.__getitem__ = MagicMock(
                side_effect=lambda k: _raise_keyerror(k) if k not in selected else [MOCK_METRICS[k]]
            )
            return result

        def _raise_keyerror(k):
            raise KeyError(k)

        with (
            patch.object(ev, "_lazy_init", return_value=True),
            patch.object(ev, "_evaluate", side_effect=fake_evaluate),
            patch.object(ev, "_build_runtime", return_value=(MagicMock(), MagicMock())),
        ):
            ev._metrics = all_metrics
            full = ev.evaluate(questions=["q"], answers=["a"], contexts=[["c"]], ground_truth=["ref"])
            partial = ev.evaluate(questions=["q"], answers=["a"], contexts=[["c"]])

        assert full.context_recall == pytest.approx(MOCK_METRICS["context_recall"])
        # selection excluded recall -> KeyError path -> wrapper scores 0.0
        assert partial.context_recall == 0.0
