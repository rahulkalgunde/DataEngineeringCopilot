"""Tests for QueryRewriter — multi-step decomposition, HyDE, intent classification."""

from __future__ import annotations

import pytest

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


class TestCodeIntentClassification:
    """Tests for expanded code_example regex patterns."""

    def test_give_me_code_to_read(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Give me code to read from delta lake table")
        assert result == "code_example"

    def test_write_code_for_spark(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Write code for Spark streaming job")
        assert result == "code_example"

    def test_show_me_code_to_implement(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Show me code to implement custom transformer")
        assert result == "code_example"

    def test_generate_sample_code(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Generate sample code for data pipeline")
        assert result == "code_example"

    def test_code_to_connect(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Code to connect to Kafka")
        assert result == "code_example"

    def test_code_for_etl(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Code for ETL process")
        assert result == "code_example"

    def test_write_a_script(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Write a script to clean data")
        assert result == "code_example"

    def test_how_to_write_code(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("How to write code for streaming")
        assert result == "code_example"

    def test_how_to_implement(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("How to implement window functions")
        assert result == "code_example"

    def test_provide_code_example(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Provide code example for joins")
        assert result == "code_example"

    def test_send_me_sample_code(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Send me sample code for aggregation")
        assert result == "code_example"

    def test_get_code_for(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Get code for reading parquet files")
        assert result == "code_example"

    def test_code_snippet(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Code snippet for DataFrame operations")
        assert result == "code_example"

    def test_code_sample(self):
        rw = QueryRewriter(llm_client=None, enabled=True)
        result = rw.classify_intent("Code sample for Spark SQL queries")
        assert result == "code_example"


class TestHybridIntentClassification:
    """Tests for LLM fallback in intent classification."""

    def test_llm_fallback_disabled_by_default(self):
        rw = QueryRewriter(llm_client=None, enabled=True, intent_llm_enabled=False)
        result = rw.classify_intent("Give me something to read")
        assert result == "factual"

    def test_llm_fallback_enabled_no_client(self):
        rw = QueryRewriter(llm_client=None, enabled=True, intent_llm_enabled=True)
        result = rw.classify_intent("Give me something to read")
        assert result == "factual"

    def test_llm_fallback_success(self):
        class FakeLLM:
            async def generate(self, prompt: str, **kwargs: object) -> str:  # noqa: ARG001
                return '{"intent": "code_example"}'

        rw = QueryRewriter(llm_client=FakeLLM(), enabled=True, intent_llm_enabled=True)
        result = rw.classify_intent("Give me something to read")
        assert result == "code_example"

    def test_llm_fallback_failure_returns_factual(self):
        class FailingLLM:
            async def generate(self, prompt: str, **kwargs: object) -> str:  # noqa: ARG001
                raise RuntimeError("LLM unavailable")

        rw = QueryRewriter(llm_client=FailingLLM(), enabled=True, intent_llm_enabled=True)
        result = rw.classify_intent("Give me something to read")
        assert result == "factual"

    def test_llm_fallback_invalid_json_returns_factual(self):
        class FakeLLM:
            async def generate(self, prompt: str, **kwargs: object) -> str:  # noqa: ARG001
                return "not valid json"

        rw = QueryRewriter(llm_client=FakeLLM(), enabled=True, intent_llm_enabled=True)
        result = rw.classify_intent("Give me something to read")
        assert result == "factual"

    def test_llm_fallback_invalid_intent_returns_factual(self):
        class FakeLLM:
            async def generate(self, prompt: str, **kwargs: object) -> str:  # noqa: ARG001
                return '{"intent": "unknown_intent"}'

        rw = QueryRewriter(llm_client=FakeLLM(), enabled=True, intent_llm_enabled=True)
        result = rw.classify_intent("Give me something to read")
        assert result == "factual"

    def test_regex_fast_path_takes_priority(self):
        """Regex match should take priority over LLM fallback."""

        class FakeLLM:
            async def generate(self, prompt: str, **kwargs: object) -> str:  # noqa: ARG001
                return '{"intent": "factual"}'  # LLM would say factual

        rw = QueryRewriter(llm_client=FakeLLM(), enabled=True, intent_llm_enabled=True)
        result = rw.classify_intent("Give me code to read from delta lake")
        assert result == "code_example"  # Regex wins


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


class TestAsyncRewrite:
    @pytest.mark.asyncio
    async def test_async_rewrite_returns_rewritten_query(self):
        class RecordingLLM:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def generate(self, prompt: str, **kwargs: object) -> str:  # noqa: ARG001
                self.calls.append(prompt)
                return "spark sql module structured data"

        llm = RecordingLLM()
        rw = QueryRewriter(llm_client=llm, enabled=True, hyde_enabled=True)

        result = await rw.async_rewrite("What is Spark SQL?")

        assert result.original_query == "What is Spark SQL?"
        assert result.intent == "factual"
        assert result.decomposed_steps == ("spark sql module structured data",)
        assert "spark sql" in result.hyde_query
        assert len(llm.calls) == 2  # rewrite + hyde

    @pytest.mark.asyncio
    async def test_async_rewrite_disabled_returns_passthrough(self):
        rw = QueryRewriter(llm_client=None, enabled=False, hyde_enabled=True)

        result = await rw.async_rewrite("What is Spark?")

        assert result.original_query == "What is Spark?"
        assert result.intent == "factual"
        assert result.hyde_query == ""
        assert result.decomposed_steps == ("What is Spark?",)

    @pytest.mark.asyncio
    async def test_async_rewrite_falls_back_to_rule_based_on_llm_error(self):
        class FailingLLM:
            async def generate(self, prompt: str, **kwargs: object) -> str:  # noqa: ARG001
                raise RuntimeError("LLM unavailable")

        rw = QueryRewriter(llm_client=FailingLLM(), enabled=True, hyde_enabled=True)

        result = await rw.async_rewrite("How to configure Delta Lake with Spark")

        assert result.intent == "how_to"
        assert result.original_query == "How to configure Delta Lake with Spark"
