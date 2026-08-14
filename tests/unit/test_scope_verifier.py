"""Tests for ScopeVerifier — topic-scope gate (fail-open) parsing and behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock

from data_engineering_copilot.services.scope_verifier import ScopeVerifier, _parse_verdict


class TestParseVerdict:
    def test_covers(self):
        assert _parse_verdict('{"verdict": "covers", "reason": "docs present"}') is True

    def test_partially_is_fail_open(self):
        assert _parse_verdict('{"verdict": "partially", "reason": "tangential"}') is True

    def test_does_not_cover_refuses(self):
        assert _parse_verdict('{"verdict": "does_not_cover", "reason": "different product"}') is False

    def test_fenced_json(self):
        assert _parse_verdict('```json\n{"verdict": "does_not_cover"}\n```') is False

    def test_case_insensitive_verdict(self):
        assert _parse_verdict('{"verdict": "DOES_NOT_COVER"}') is False
        assert _parse_verdict('{"verdict": "Partially"}') is True

    def test_legacy_covered_bool_backcompat(self):
        assert _parse_verdict('{"covered": true}') is True
        assert _parse_verdict('{"covered": false}') is False

    def test_regex_verdict_fallback(self):
        assert _parse_verdict('"verdict": "does_not_cover"') is False
        assert _parse_verdict('text "verdict": "covers" more') is True

    def test_unparseable_returns_none(self):
        assert _parse_verdict("maybe, I am not sure") is None
        assert _parse_verdict("") is None
        assert _parse_verdict("[1, 2, 3]") is None


class TestVerify:
    async def test_disabled_returns_true(self):
        gv = ScopeVerifier(llm_client=AsyncMock(), enabled=False)
        assert await gv.verify("question", "context") is True

    async def test_no_llm_client_returns_true(self):
        gv = ScopeVerifier(llm_client=None, enabled=True)
        assert await gv.verify("question", "context") is True

    async def test_empty_question_returns_true(self):
        gv = ScopeVerifier(llm_client=AsyncMock(), enabled=True)
        assert await gv.verify("", "context") is True

    async def test_covers(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"verdict": "covers", "reason": "docs cover it"}'
        gv = ScopeVerifier(llm_client=mock_llm, enabled=True)
        assert await gv.verify("how does Spark work", "Spark is a unified engine.") is True

    async def test_partially_does_not_refuse(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"verdict": "partially", "reason": "tangential"}'
        gv = ScopeVerifier(llm_client=mock_llm, enabled=True)
        assert await gv.verify("how does Spark work", "Spark is a unified engine.") is True

    async def test_does_not_cover_refuses(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"verdict": "does_not_cover", "reason": "different product"}'
        gv = ScopeVerifier(llm_client=mock_llm, enabled=True)
        assert await gv.verify("how does React work", "Spark is a unified engine.") is False

    async def test_unparseable_fails_open(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "I think the context kind of covers it."
        gv = ScopeVerifier(llm_client=mock_llm, enabled=True)
        assert await gv.verify("question", "context") is True

    async def test_llm_error_fails_open(self):
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        gv = ScopeVerifier(llm_client=mock_llm, enabled=True)
        assert await gv.verify("question", "context") is True

    async def test_prompt_contains_question_and_context(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"verdict": "covers"}'
        gv = ScopeVerifier(llm_client=mock_llm, enabled=True)
        await gv.verify("unique-question-token", "unique-context-token")
        prompt_arg = mock_llm.generate.call_args[0][0]
        assert "unique-question-token" in prompt_arg
        assert "unique-context-token" in prompt_arg
