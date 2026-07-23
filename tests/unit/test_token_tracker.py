"""Tests for LLM token usage tracker."""

from __future__ import annotations

from data_engineering_copilot.observability.token_tracker import TokenTracker


class TestTokenTracker:
    def test_initial_state(self):
        tt = TokenTracker()
        usage = tt.get_usage()
        assert usage.total_prompt_tokens == 0
        assert usage.total_completion_tokens == 0
        assert usage.total_calls == 0

    def test_record_single_call(self):
        tt = TokenTracker()
        tt.record(prompt_tokens=100, completion_tokens=50, model="llama3.2:3b")
        usage = tt.get_usage()
        assert usage.total_prompt_tokens == 100
        assert usage.total_completion_tokens == 50
        assert usage.total_calls == 1

    def test_record_multiple_calls(self):
        tt = TokenTracker()
        tt.record(prompt_tokens=100, completion_tokens=50)
        tt.record(prompt_tokens=200, completion_tokens=100)
        usage = tt.get_usage()
        assert usage.total_prompt_tokens == 300
        assert usage.total_completion_tokens == 150
        assert usage.total_calls == 2

    def test_reset(self):
        tt = TokenTracker()
        tt.record(prompt_tokens=100, completion_tokens=50)
        tt.reset()
        usage = tt.get_usage()
        assert usage.total_prompt_tokens == 0
        assert usage.total_calls == 0

    def test_by_model(self):
        tt = TokenTracker()
        tt.record(prompt_tokens=100, completion_tokens=50, model="llama3.2:3b")
        tt.record(prompt_tokens=200, completion_tokens=100, model="llama3.2:3b")
        tt.record(prompt_tokens=80, completion_tokens=20, model="nomic-embed-text")
        usage = tt.get_usage()
        assert usage.by_model["llama3.2:3b"]["prompt_tokens"] == 300
        assert usage.by_model["nomic-embed-text"]["prompt_tokens"] == 80
