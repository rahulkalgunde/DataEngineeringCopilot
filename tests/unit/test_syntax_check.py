"""Tests for _validate_and_fix_code_syntax in AsyncRagService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.domain.models import RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService


@pytest.fixture
def service():
    return AsyncRagService(
        config=RagConfig(),
        vector_store=MagicMock(query=AsyncMock(return_value=[])),
        llm_client=MagicMock(generate=AsyncMock(return_value="answer")),
        embedder=MagicMock(embed_query=AsyncMock(return_value=[0.1] * 768)),
    )


@pytest.mark.asyncio
async def test_non_code_intent_returns_unchanged(service):
    result = await service._validate_and_fix_code_syntax("some answer", "factual", service.llm_client)
    assert result == "some answer"


@pytest.mark.asyncio
async def test_no_code_block_returns_unchanged(service):
    result = await service._validate_and_fix_code_syntax(
        "Here is some text without code.", "code_example", service.llm_client
    )
    assert result == "Here is some text without code."


@pytest.mark.asyncio
async def test_valid_syntax_returns_unchanged(service):
    answer = "Here is the code:\n```python\nx = 1\ny = x + 2\n```"
    result = await service._validate_and_fix_code_syntax(answer, "code_example", service.llm_client)
    assert result == answer


@pytest.mark.asyncio
async def test_invalid_syntax_triggers_fix(service):
    code_llm = MagicMock()
    code_llm.generate = AsyncMock(return_value="x = 1")
    broken_answer = "Here is the code:\n```python\nx = \n```"
    result = await service._validate_and_fix_code_syntax(broken_answer, "api_lookup", code_llm)
    assert "x = 1" in result
    code_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_fix_failure_returns_original(service):
    code_llm = MagicMock()
    code_llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
    broken_answer = "Here is the code:\n```python\nx = \n```"
    result = await service._validate_and_fix_code_syntax(broken_answer, "code_example", code_llm)
    assert result == broken_answer
