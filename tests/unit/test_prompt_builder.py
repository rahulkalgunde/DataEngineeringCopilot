"""Tests for PromptBuilder service."""

from __future__ import annotations

import pytest

from data_engineering_copilot.services.prompt_builder import CODE_INTENTS, PromptBuilder


def test_prompt_builder_constructs_system_and_context():
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(
        context="Pandas is a Python data analysis library.",
        question="What is Pandas?",
    )
    assert "Pandas is a Python data analysis library." in prompt
    assert "What is Pandas?" in prompt
    assert "SYSTEM" in prompt or "DataEngineeringCopilot" in prompt


def test_code_example_intent_uses_code_instructions():
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(
        context="Some context.",
        question="Show me a code example",
        intent="code_example",
    )
    assert "code block" in prompt
    assert "appropriate language tag" in prompt
    assert "runnable code example" in prompt


def test_api_lookup_intent_uses_code_instructions():
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(
        context="Some context.",
        question="What is the API for SparkSession?",
        intent="api_lookup",
    )
    assert "code block" in prompt
    assert "API signatures" in prompt


@pytest.mark.parametrize("intent", ["factual", "how_to", "comparative", "debugging", "unknown"])
def test_non_code_intents_use_documentation_instructions(intent: str):
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(
        context="Some context.",
        question="Some question",
        intent=intent,
    )
    assert "factual questions: State facts" in prompt
    assert "runnable code example" not in prompt


def test_code_intents_are_defined():
    assert "code_example" in CODE_INTENTS
    assert "api_lookup" in CODE_INTENTS
    assert len(CODE_INTENTS) == 2


class TestSafetyNet:
    """Tests for safety net allowing code blocks when query contains code keywords."""

    def test_factual_with_code_keyword_allows_code_blocks(self):
        """Query with code keywords should allow code blocks even for factual intent."""
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Spark DataFrame API documentation.",
            question="Give me code to read from delta lake",
            intent="factual",
        )
        assert "include a complete, runnable code example" in prompt
        assert "code block" in prompt

    def test_factual_without_code_keyword_no_code_blocks(self):
        """Query without code keywords should not allow code blocks."""
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Spark DataFrame API documentation.",
            question="What is the best practice for data processing?",
            intent="factual",
        )
        assert "include a complete, runnable code example" not in prompt
        assert "factual questions: State facts" in prompt

    @pytest.mark.parametrize(
        "keyword",
        ["code", "script", "function", "implement", "snippet", "sample", "example", "pyspark", "scala", "python"],
    )
    def test_code_keywords_trigger_safety_net(self, keyword: str):
        """All code-related keywords should trigger the safety net."""
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Some documentation.",
            question=f"Show me a {keyword} for data processing",
            intent="factual",
        )
        assert "include a complete, runnable code example" in prompt

    def test_code_intent_uses_full_code_instructions(self):
        """Code intent should use full code instructions, not safety net."""
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Some documentation.",
            question="Show me code to read from delta lake",
            intent="code_example",
        )
        # Should use _CODE_INSTRUCTIONS, not _DOCUMENTATION_INSTRUCTIONS_WITH_CODE
        assert "Provide a brief explanation (1-3 sentences) followed by a complete, runnable code example" in prompt
        assert "Match the language requested by the user" in prompt
