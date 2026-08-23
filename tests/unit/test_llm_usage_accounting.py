from data_engineering_copilot.infrastructure.provider_fallback import UsageLedger


def test_record_and_snapshot():
    UsageLedger.reset()
    UsageLedger.record("judge", {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5})
    UsageLedger.record("judge", {"calls": 1, "prompt_tokens": 7, "completion_tokens": 3})
    UsageLedger.record("generator", {"calls": 1, "prompt_tokens": 100, "completion_tokens": 50})
    snap = UsageLedger.snapshot()
    assert snap["judge"] == {"calls": 2, "prompt_tokens": 17, "completion_tokens": 8}
    assert snap["generator"] == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 50}


def test_reset_clears():
    UsageLedger.record("x", {"calls": 1, "prompt_tokens": 1, "completion_tokens": 1})
    UsageLedger.reset()
    assert UsageLedger.snapshot() == {}
