"""Unsampled production judging (rate >= 1.0, no --max-items) logs a warning."""

import logging

from data_engineering_copilot.evaluation import langfuse_evaluators as le


async def test_warn_when_unsampled(caplog):
    with caplog.at_level(logging.WARNING, logger="data_engineering_copilot.evaluation.langfuse_evaluators"):
        le.warn_if_unsampled(sample_rate=1.0, max_items=None)
    assert any("unsampled" in r.message.lower() for r in caplog.records)


async def test_silent_when_sampled_or_capped(caplog):
    with caplog.at_level(logging.WARNING, logger="data_engineering_copilot.evaluation.langfuse_evaluators"):
        le.warn_if_unsampled(sample_rate=0.1, max_items=None)
        le.warn_if_unsampled(sample_rate=1.0, max_items=25)
    assert not [r for r in caplog.records if "unsampled" in r.message.lower()]
