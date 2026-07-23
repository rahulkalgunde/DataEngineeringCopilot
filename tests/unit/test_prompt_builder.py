"""Tests for PromptBuilder service."""

from __future__ import annotations

from data_engineering_copilot.services.prompt_builder import PromptBuilder


def test_prompt_builder_constructs_system_and_context():
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(
        context="Pandas is a Python data analysis library.",
        question="What is Pandas?",
    )
    assert "Pandas is a Python data analysis library." in prompt
    assert "What is Pandas?" in prompt
    assert "SYSTEM" in prompt or "DataEngineeringCopilot" in prompt
