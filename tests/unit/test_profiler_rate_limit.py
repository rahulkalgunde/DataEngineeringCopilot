"""Tests for RateLimitTracker and RateLimitEvent."""

from __future__ import annotations

from data_engineering_copilot.profiler.rate_limit_tracker import (
    RateLimitTracker,
    _extract_ratelimit_headers,
    _parse_retry_after,
)


class TestParseRetryAfter:
    def test_none_when_missing(self):
        assert _parse_retry_after({}) is None

    def test_parses_float(self):
        assert _parse_retry_after({"retry-after": "5.0"}) == 5.0

    def test_parses_retry_after_case_insensitive(self):
        assert _parse_retry_after({"Retry-After": "30"}) == 30.0

    def test_invalid_returns_none(self):
        assert _parse_retry_after({"retry-after": "not-a-number"}) is None


class TestExtractRatelimitHeaders:
    def test_openrouter_headers_extracted(self):
        headers = {
            "x-ratelimit-remaining": "10",
            "x-ratelimit-limit": "100",
            "x-ratelimit-reset": "60",
        }
        result = _extract_ratelimit_headers("openrouter", headers)
        assert result["remaining"] == 10
        assert result["limit"] == 100

    def test_openai_headers_extracted(self):
        headers = {
            "x-ratelimit-remaining-requests": "5",
            "x-ratelimit-limit-requests": "50",
            "x-ratelimit-reset-requests": "30",
        }
        result = _extract_ratelimit_headers("openai", headers)
        assert result["remaining"] == 5
        assert result["limit"] == 50

    def test_unknown_provider_returns_empty(self):
        result = _extract_ratelimit_headers("stripe", {"x-ratelimit-remaining": "10"})
        assert result == {}

    def test_invalid_header_values_ignored(self):
        headers = {"x-ratelimit-remaining": "abc"}
        result = _extract_ratelimit_headers("openrouter", headers)
        assert "remaining" not in result


class TestRateLimitTracker:
    def test_empty_on_init(self):
        rlt = RateLimitTracker()
        assert rlt.total_429s == 0
        assert rlt.events == []

    def test_tracks_200_events(self):
        rlt = RateLimitTracker()
        rlt.record_response("ollama", 200)
        assert rlt.total_429s == 0
        assert len(rlt.events) == 1

    def test_tracks_429_events(self):
        rlt = RateLimitTracker()
        rlt.record_response("openrouter", 429)
        assert rlt.total_429s == 1
        assert rlt.events[0].status_code == 429

    def test_provider_429_count(self):
        rlt = RateLimitTracker()
        rlt.record_response("openrouter", 429)
        rlt.record_response("openrouter", 200)
        rlt.record_response("ollama", 429)
        assert rlt.provider_429_count("openrouter") == 1
        assert rlt.provider_429_count("ollama") == 1

    def test_get_provider_saturation_zero_when_no_events(self):
        rlt = RateLimitTracker()
        assert rlt.get_provider_saturation("openrouter") == 0.0

    def test_saturation_from_ratelimit_headers(self):
        rlt = RateLimitTracker()
        rlt.record_response(
            "openrouter",
            200,
            headers={"x-ratelimit-remaining": "20", "x-ratelimit-limit": "100"},
        )
        sat = rlt.get_provider_saturation("openrouter")
        assert 0.79 < sat < 0.81  # 1 - (20/100) = 0.8

    def test_saturation_from_429_ratio(self):
        rlt = RateLimitTracker()
        for _ in range(9):
            rlt.record_response("openrouter", 200)
        rlt.record_response("openrouter", 429)
        sat = rlt.get_provider_saturation("openrouter")
        assert 0.09 < sat < 0.11  # 1/10 = 0.1

    def test_is_throttled_above_threshold(self):
        rlt = RateLimitTracker()
        for _ in range(5):
            rlt.record_response("openrouter", 429)
        assert rlt.is_throttled("openrouter", threshold=0.1) is True

    def test_is_not_throttled_below_threshold(self):
        rlt = RateLimitTracker()
        rlt.record_response("openrouter", 429)
        rlt.record_response("openrouter", 200)
        rlt.record_response("openrouter", 200)
        assert rlt.is_throttled("openrouter", threshold=0.5) is False
