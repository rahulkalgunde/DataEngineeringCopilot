"""Tests for PromptBuilder service."""

from __future__ import annotations

import re

import pytest

from data_engineering_copilot.services.prompt_builder import (
    CODE_INTENTS,
    TRAILING_INSTRUCTIONS,
    PromptBuilder,
)


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


def test_mode_confusion_constraint_present(monkeypatch):
    """The rag-answer prompt must carry the mode/entity isolation constraint.

    Hermetic: force the offline fallback template so the test does not depend
    on a live Langfuse server prompt (which may lag the seeded template).
    """
    from data_engineering_copilot.observability.langfuse_prompts import _FallbackPrompt
    from data_engineering_copilot.services import prompt_builder as pb_module

    monkeypatch.setattr(
        pb_module,
        "get_langfuse_prompt",
        lambda *a, **k: _FallbackPrompt(pb_module._RAG_PROMPT_TEMPLATE),
    )
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(
        context="Some context.",
        question="How does dynamic allocation work on YARN or Kubernetes?",
        intent="comparative",
    )
    assert "Mode/Entity Isolation" in prompt
    assert "YARN vs Kubernetes" in prompt
    assert "do NOT transfer behavior from one mode to another" in prompt


class TestHistoryInjection:
    def test_no_history_matches_legacy_prompt(self):
        """Without history, no history section appears (salted tags break byte-identity)."""
        builder = PromptBuilder()
        plain = builder.build_rag_prompt(
            context="Some context.",
            question="What is Spark?",
            intent="factual",
        )
        no_history = builder.build_rag_prompt(
            context="Some context.",
            question="What is Spark?",
            intent="factual",
            history=None,
        )
        assert "## CONVERSATION HISTORY" not in plain
        assert "## CONVERSATION HISTORY" not in no_history
        # Both contain the context and question regardless of salt.
        assert "Some context." in plain
        assert "Some context." in no_history

    def test_history_injected_before_context(self):
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="RETRIEVED_CONTEXT_MARKER",
            question="What about its API?",
            intent="factual",
            history="User: How does filter work?\nAssistant: It filters arrays.",
        )
        assert "## CONVERSATION HISTORY" in prompt
        assert "User: How does filter work?" in prompt
        assert "Assistant: It filters arrays." in prompt
        # History appears before the retrieved context.
        assert prompt.index("User: How does filter work?") < prompt.index("RETRIEVED_CONTEXT_MARKER")
        assert prompt.index("## CONVERSATION HISTORY") < prompt.index("RETRIEVED_CONTEXT_MARKER")

    def test_history_budget_evicts_oldest_first(self):
        builder = PromptBuilder()
        # Many turns of identical size; a tiny budget keeps only the most recent.
        history = "\n".join(f"User: question {i}\nAssistant: answer {i}" for i in range(20))
        budgeted = builder._budget_history(history, max_history_tokens=20)
        assert "question 19" in budgeted  # most recent turn kept
        assert "question 0" not in budgeted  # oldest evicted

    def test_history_under_budget_is_unchanged(self):
        builder = PromptBuilder()
        history = "User: hi\nAssistant: hello"
        assert builder._budget_history(history, max_history_tokens=100) == history

    def test_eviction_never_splits_a_turn(self):
        """The budget must drop whole turns, never split User/Assistant pairs."""
        builder = PromptBuilder()
        history = "\n".join(f"User: q{i}\nAssistant: a{i}" for i in range(10))
        budgeted = builder._budget_history(history, max_history_tokens=12)
        lines = budgeted.splitlines()
        # Any kept User line must have its Assistant line present.
        for i, line in enumerate(lines):
            if line.startswith("User:"):
                assert i + 1 < len(lines)
                assert lines[i + 1].startswith("Assistant:")

    def test_system_role_separator_preserved_with_history(self):
        """History injection must not break the SYSTEM_BLOCK_SEPARATOR split."""
        from data_engineering_copilot.infrastructure.llm_client import (
            SYSTEM_BLOCK_SEPARATOR,
            build_chat_messages,
        )

        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Some context.",
            question="Follow up?",
            intent="factual",
            history="User: first\nAssistant: reply",
        )
        assert SYSTEM_BLOCK_SEPARATOR in prompt
        messages = build_chat_messages(prompt)
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # History block landed in the user message, before the context.
        assert "## CONVERSATION HISTORY" in messages[1]["content"]
        assert "Some context." in messages[1]["content"]


class TestSaltedXmlTags:
    """Per-request salted context tags for prompt-injection hardening."""

    def test_salted_xml_tags_are_unique_per_request(self):
        """Each call to build_rag_prompt must produce a different salt."""
        builder = PromptBuilder()
        p1 = builder.build_rag_prompt(context="ctx", question="q1")
        p2 = builder.build_rag_prompt(context="ctx", question="q2")
        tag1 = re.search(r"<context_data_(\w+)>", p1)
        tag2 = re.search(r"<context_data_(\w+)>", p2)
        assert tag1 is not None, "Expected <context_data_...> open tag"
        assert tag2 is not None, "Expected <context_data_...> open tag"
        assert tag1.group(1) != tag2.group(1)

    def test_salted_tags_contain_context(self):
        """Context text must appear between the salted open and close tags."""
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(context="Hello world", question="q")
        match = re.search(r"<context_data_\w+>.*Hello world.*</context_data_\w+>", prompt, re.DOTALL)
        assert match is not None, "Context not found inside salted tags"

    def test_salted_tag_format_is_8char_hex(self):
        """The salt portion must be exactly 8 hex characters."""
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(context="x", question="q")
        salt = re.search(r"<context_data_([0-9a-f]{8})>", prompt)
        assert salt is not None, "Salt did not match 8-char hex pattern"

    def test_old_chunk_tags_are_gone(self):
        """Static <chunk> tags must no longer appear in the prompt."""
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(context="x", question="q")
        assert "<chunk>" not in prompt
        assert "</chunk>" not in prompt

    def test_old_chunk_tags_gone_with_history(self):
        """Salted tags must be used even when history is injected."""
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Some context.",
            question="q",
            history="User: hi\nAssistant: hello",
        )
        assert "<chunk>" not in prompt
        assert re.search(r"<context_data_\w+>", prompt) is not None


class TestTrailingInstructions:
    """Instruction sandwiching: trailing reminders exploit recency bias."""

    def test_trailing_instructions_present_by_default(self, monkeypatch):
        from data_engineering_copilot.observability.langfuse_prompts import _FallbackPrompt
        from data_engineering_copilot.services import prompt_builder as pb_module

        monkeypatch.setattr(
            pb_module,
            "get_langfuse_prompt",
            lambda *a, **k: _FallbackPrompt(pb_module._RAG_PROMPT_TEMPLATE),
        )
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(context="ctx", question="q")
        assert "CRITICAL REMINDERS" in prompt
        assert "Every factual claim MUST cite" in prompt

    def test_trailing_instructions_absent_when_disabled(self):
        builder = PromptBuilder(prompt_trailing_instructions=False)
        prompt = builder.build_rag_prompt(context="ctx", question="q")
        assert "CRITICAL REMINDERS" not in prompt

    def test_trailing_instructions_before_answer_section(self, monkeypatch):
        """Trailing block must appear between context and YOUR ANSWER (recency bias)."""
        from data_engineering_copilot.observability.langfuse_prompts import _FallbackPrompt
        from data_engineering_copilot.services import prompt_builder as pb_module

        monkeypatch.setattr(
            pb_module,
            "get_langfuse_prompt",
            lambda *a, **k: _FallbackPrompt(pb_module._RAG_PROMPT_TEMPLATE),
        )
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(context="ctx", question="q")
        idx_reminders = prompt.index("CRITICAL REMINDERS")
        idx_answer = prompt.index("## YOUR ANSWER")
        assert idx_reminders < idx_answer

    def test_trailing_instructions_constant_content(self):
        assert "Never fabricate citations" in TRAILING_INSTRUCTIONS
        assert "INSUFFICIENT_CONTEXT" in TRAILING_INSTRUCTIONS


class TestCitationEnforcement:
    """Citation examples baked into code and documentation instructions."""

    def test_citation_examples_in_code_instructions(self):
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Some context.",
            question="Show me code for Spark joins",
            intent="code_example",
        )
        assert "Doc-1" in prompt
        assert "cite the source" in prompt.lower() or "cite" in prompt.lower()

    def test_citation_examples_in_documentation_instructions(self):
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Some context.",
            question="What is Spark?",
            intent="factual",
        )
        assert "Doc-" in prompt

    def test_no_fabrication_rule_in_code_instructions(self):
        builder = PromptBuilder()
        prompt = builder.build_rag_prompt(
            context="Some context.",
            question="Show me code",
            intent="code_example",
        )
        assert "do not fabricate" in prompt.lower() or "never fabricate" in prompt.lower()
