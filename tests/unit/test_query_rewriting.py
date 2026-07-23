"""Tests for QueryRewriter — multi-step decomposition, HyDE, intent classification."""

from __future__ import annotations

from data_engineering_copilot.services.query_rewriting import QueryRewriter


class TestIntentClassification:
    def test_simple_factual_returns_single_step(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("What is Spark SQL?")
        assert result == "factual"

    def test_comparative_returns_multi_step(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Compare Spark DataFrame API vs Spark SQL syntax")
        assert result == "comparative"

    def test_procedural_returns_how_to(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("How to configure Delta Lake with Spark")
        assert result == "how_to"

    def test_debugging_returns_debug(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Why is my Spark job failing with OOM error?")
        assert result == "debugging"

    def test_disabled_returns_factual(self):
        rw = QueryRewriter(llm_client=None, enabled=False)
        result = rw.classify_intent("anything goes here")
        assert result == "factual"


class TestDecomposeQuery:
    def test_single_step_for_factual(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        steps = rw.decompose("What is Spark SQL?", intent="factual")
        assert len(steps) == 1
        assert steps[0] == "What is Spark SQL?"

    def test_multi_step_for_comparative(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        steps = rw.decompose(
            "Compare Spark DataFrame API vs Spark SQL",
            intent="comparative",
        )
        assert len(steps) >= 2
        assert any("DataFrame" in s for s in steps)

    def test_disabled_returns_original(self):
        rw = QueryRewriter(llm_client=None, enabled=False)
        steps = rw.decompose("Compare X and Y", intent="comparative")
        assert steps == ("Compare X and Y",)


class TestRewrite:
    def test_disabled_returns_passthrough(self):
        rw = QueryRewriter(llm_client=None, enabled=False)
        result = rw.rewrite("What is Spark?")
        assert result.original_query == "What is Spark?"
        assert result.intent == "factual"
        assert result.hyde_query == ""
        assert result.decomposed_steps == ("What is Spark?",)

    def test_enabled_classifies_and_decomposes(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.rewrite("How to configure Delta Lake with Spark")
        assert result.intent == "how_to"
        assert len(result.decomposed_steps) >= 1
        assert result.original_query == "How to configure Delta Lake with Spark"

    def test_hyde_disabled_by_default(self):
        rw = QueryRewriter(llm_client=None, enabled=True, hyde_enabled=False)
        result = rw.rewrite("What is Spark?")
        assert result.hyde_query == ""

    def test_hyde_enabled_no_client_returns_empty(self):
        rw = QueryRewriter(llm_client=None, enabled=True, hyde_enabled=True)
        result = rw.rewrite("What is Spark?")
        assert result.hyde_query == ""

    def test_hyde_enabled_with_client(self):
        class FakeLLM:
            async def generate(self, prompt: str, **kwargs: object) -> str:  # noqa: ARG001
                return "Spark SQL is a module for structured data."

        rw = QueryRewriter(llm_client=FakeLLM(), enabled=True, hyde_enabled=True)
        result = rw.rewrite("What is Spark SQL?")
        assert "Spark SQL" in result.hyde_query
        assert result.original_query == "What is Spark SQL?"
