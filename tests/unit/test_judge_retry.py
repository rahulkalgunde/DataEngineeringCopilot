from unittest.mock import AsyncMock, MagicMock

from data_engineering_copilot.evaluation.generation_eval import (
    _judge_call_with_retry,
    _parse_score,
)


def test_parse_score_json():
    assert _parse_score('{"score": 3.5}', 1.0, 5.0) == 3.5


def test_parse_score_bare_number():
    assert _parse_score("0.82", 0.0, 1.0) == 0.82


def test_parse_score_empty():
    assert _parse_score("", 0.0, 1.0) is None


def test_parse_score_legit_zero_is_zero_not_none():
    assert _parse_score('{"score": 0}', 0.0, 1.0) == 0.0


def test_parse_score_no_numbers():
    assert _parse_score("no numbers here at all", 0.0, 1.0) is None


def test_parse_score_clamped():
    assert _parse_score("99", 1.0, 5.0) == 5.0


async def test_retry_succeeds_on_second_call():
    judge = MagicMock()
    judge.generate = AsyncMock(side_effect=["unparseable!", '{"score": 4.2}'])
    s = await _judge_call_with_retry(judge, "prompt", 0.0, 5.0, max_retries=3)
    assert s == 4.2
    assert judge.generate.call_count == 2


async def test_retry_exhausted_returns_zero():
    judge = MagicMock()
    judge.generate = AsyncMock(return_value="garbage")
    s = await _judge_call_with_retry(judge, "prompt", 0.0, 1.0, max_retries=2)
    assert s == 0.0
    assert judge.generate.call_count == 2


async def test_retry_first_call_ok():
    judge = MagicMock()
    judge.generate = AsyncMock(return_value='{"score": 0.9}')
    s = await _judge_call_with_retry(judge, "prompt", 0.0, 1.0, max_retries=3)
    assert s == 0.9
    assert judge.generate.call_count == 1


async def test_legit_zero_score_returns_immediately_no_retry():
    """A parsed 0 is a valid verdict — must NOT be retried (3x spend bug)."""
    judge = MagicMock()
    judge.generate = AsyncMock(return_value='{"score": 0}')
    s = await _judge_call_with_retry(judge, "prompt", 0.0, 1.0, max_retries=3)
    assert s == 0.0
    assert judge.generate.call_count == 1
