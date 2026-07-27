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
    assert "runnable Python/PySpark code" in prompt
    assert "type hints" in prompt


def test_api_lookup_intent_uses_code_instructions():
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(
        context="Some context.",
        question="What is the API for SparkSession?",
        intent="api_lookup",
    )
    assert "runnable Python/PySpark code" in prompt
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
    assert "runnable Python/PySpark code" not in prompt


def test_code_intents_are_defined():
    assert "code_example" in CODE_INTENTS
    assert "api_lookup" in CODE_INTENTS
    assert len(CODE_INTENTS) == 2
