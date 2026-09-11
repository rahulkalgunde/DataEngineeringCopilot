import pytest

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import RagConfig


def test_default_budgets():
    s = AppSettings(_env_file=None, reranker_enabled=True)
    assert s.reranker_init_budget_seconds == 15.0
    assert s.chat_turn_budget_seconds == 90.0
    assert s.reranker_eager_warmup is True


def test_ragconfig_receives_rerank_budget():
    cfg = RagConfig(reranker_enabled=True, reranker_init_budget_seconds=7.0)
    assert cfg.reranker_init_budget_seconds == 7.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reranker_init_budget_seconds": 0.0},
        {"reranker_init_budget_seconds": -1.0},
        {"chat_turn_budget_seconds": 0.0},
        {"chat_turn_budget_seconds": 601.0},
    ],
)
def test_invalid_budgets_raise(kwargs):
    # AppSettings is frozen — construct with the bad value, don't setattr.
    s = AppSettings(_env_file=None, **kwargs)
    with pytest.raises(ValueError):
        s.validate_all()


def test_budget_must_say_under_turn_budget():
    s = AppSettings(_env_file=None, reranker_init_budget_seconds=95.0, chat_turn_budget_seconds=90.0)
    with pytest.raises(ValueError):
        s.validate_all()
