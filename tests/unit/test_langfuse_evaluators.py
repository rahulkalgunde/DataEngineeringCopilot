"""Tests for Phase 7 LLM-as-a-judge evaluators (langfuse_evaluators.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.evaluation import langfuse_evaluators as mod


def test_parse_float_score_extracts_json_score() -> None:
    assert mod._parse_float_score('{"score": 0.8, "reason": "ok"}') == 0.8
    assert mod._parse_float_score('{"score": 1, "reason": "full"}') == 1.0


def test_parse_float_score_extracts_bare_number() -> None:
    assert mod._parse_float_score("0.42") == 0.42


def test_parse_float_score_out_of_range_treated_as_zero() -> None:
    # The scorer only accepts 0-1; out-of-range numbers are not matched.
    assert mod._parse_float_score('{"score": 7}') == 0.0
    assert mod._parse_float_score("2") == 0.0


def test_parse_float_score_empty() -> None:
    assert mod._parse_float_score("") == 0.0
    assert mod._parse_float_score("no numbers here") == 0.0


def test_parse_bool_extracts_flag() -> None:
    assert mod._parse_bool('{"out_of_scope": true}') is True
    assert mod._parse_bool('{"out_of_scope": false}') is False


def test_parse_bool_loose_match() -> None:
    assert mod._parse_bool("definitely true") is True
    assert mod._parse_bool("nope") is False
    assert mod._parse_bool("") is False


def test_coerce_answer_plain_string_passthrough() -> None:
    assert mod._coerce_answer("Spark is an engine") == "Spark is an engine"


def test_coerce_answer_dict_prefers_answer_key() -> None:
    out = {"status": "OK", "answer": "real answer", "missing_info": None}
    assert mod._coerce_answer(out) == "real answer"


def test_coerce_answer_dict_falls_back_to_text() -> None:
    assert mod._coerce_answer({"text": "fallback text"}) == "fallback text"


def test_coerce_answer_dict_without_text_returns_dict() -> None:
    out = {"status": "INSUFFICIENT_CONTEXT", "answer": None}
    assert mod._coerce_answer(out) is out


async def _fake_judge_response(text: str):
    with patch.object(mod, "_run_judge", AsyncMock(return_value=text)):
        from langfuse import Evaluation

        result = await mod.faithfulness_judge(input="q", output="a", metadata={"retrieved_contexts": "ctx"})
        assert isinstance(result, Evaluation)
        return result


@pytest.mark.asyncio
async def test_faithfulness_judge_returns_evaluation() -> None:
    result = await _fake_judge_response('{"score": 0.9, "reason": "supported"}')
    assert result.name == "faithfulness"
    assert result.value == 0.9


@pytest.mark.asyncio
async def test_relevance_judge_returns_evaluation() -> None:
    with patch.object(mod, "_run_judge", AsyncMock(return_value='{"score": 0.7}')):
        from langfuse import Evaluation

        result = await mod.relevance_judge(input="q", output="a")
        assert isinstance(result, Evaluation)
        assert result.name == "relevance"
        assert result.value == 0.7


@pytest.mark.asyncio
async def test_out_of_scope_judge_returns_boolean() -> None:
    with patch.object(mod, "_run_judge", AsyncMock(return_value='{"out_of_scope": true}')):
        from langfuse import Evaluation

        result = await mod.out_of_scope_judge(input="q", output="a")
        assert isinstance(result, Evaluation)
        assert result.name == "out_of_scope"
        assert result.value is True
        assert result.data_type == "BOOLEAN"


def test_trace_mapper_extracts_retrieval_context() -> None:
    from langfuse.batch_evaluation import EvaluatorInputs

    item = MagicMock()
    item.id = "trace-123"
    item.input = "question"
    item.output = {"answer": "answer text"}
    item.metadata = {"intent": "factual"}
    retrieval = MagicMock()
    retrieval.name = "retrieval"
    retrieval.output = ["chunk one", "chunk two"]
    other = MagicMock()
    other.name = "llm-generation"
    other.output = "irrelevant"
    item.observations = [other, retrieval]

    inputs = mod._trace_mapper(item=item)
    assert isinstance(inputs, EvaluatorInputs)
    assert inputs.input == "question"
    assert inputs.output == "answer text"
    metadata = inputs.metadata or {}
    assert metadata["retrieved_contexts"] == "chunk one\nchunk two"
    assert metadata["trace_id"] == "trace-123"


def test_trace_mapper_truncates_huge_context() -> None:
    item = MagicMock()
    item.id = "t"
    item.input = "q"
    item.output = "a"
    item.metadata = {}
    retrieval = MagicMock()
    retrieval.name = "retrieval"
    retrieval.output = ["x" * mod.MAX_JUDGE_CONTEXT_CHARS * 2]
    item.observations = [retrieval]
    inputs = mod._trace_mapper(item=item)
    assert len((inputs.metadata or {})["retrieved_contexts"]) <= mod.MAX_JUDGE_CONTEXT_CHARS


def test_run_batched_trace_evaluation_sampled_out() -> None:
    with patch.object(mod.random, "random", return_value=1.0), patch.object(mod, "get_langfuse_instance") as mock_get:
        mock_get.return_value = None
        result = mod.run_batched_trace_evaluation()
    assert result is None


def test_run_batched_trace_evaluation_no_client() -> None:
    with (
        patch.object(mod.random, "random", return_value=0.0),
        patch.object(mod, "get_langfuse_instance", return_value=None),
    ):
        result = mod.run_batched_trace_evaluation()
    assert result is None


def test_run_batched_trace_evaluation_explicit_max_items_bypasses_sampling() -> None:
    fake_result = object()
    fake_client = MagicMock()
    fake_client._client.run_batched_evaluation.return_value = fake_result
    with (
        patch.object(mod.random, "random", return_value=1.0),
        patch.object(mod, "get_langfuse_instance", return_value=fake_client),
    ):
        result = mod.run_batched_trace_evaluation(max_items=5)
    assert result is fake_result
    fake_client._client.run_batched_evaluation.assert_called_once()
    kwargs = fake_client._client.run_batched_evaluation.call_args.kwargs
    assert kwargs["scope"] == "traces"
    assert [e.__name__ for e in kwargs["evaluators"]] == [
        "faithfulness_judge",
        "relevance_judge",
        "out_of_scope_judge",
    ]
