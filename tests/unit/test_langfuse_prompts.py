"""Tests for the Langfuse prompt-management helper and prompt migrations.

Behavioral regression: every migrated prompt must compile byte-identically to
the string the hardcoded template produced before migration — both through the
offline fallback and through the Langfuse-form seed template (via the real SDK
``TemplateParser``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from data_engineering_copilot.domain.models import ParsedDocument

# Importing the service modules registers their fallbacks at import time.
from data_engineering_copilot.evaluation import (
    langfuse_evaluators as _eval_module,  # noqa: F401  (registers judge-* fallbacks)
)
from data_engineering_copilot.infrastructure.llm_client import SYSTEM_BLOCK_SEPARATOR
from data_engineering_copilot.observability import langfuse_prompts as lp_module
from data_engineering_copilot.observability.langfuse_prompts import get_langfuse_prompt
from data_engineering_copilot.services.async_rag import _JSON_RETRY_SUFFIX
from data_engineering_copilot.services.contextual_chunk_enricher import _SUMMARY_PROMPT, LLMContextSummarizer
from data_engineering_copilot.services.groundedness import _NLI_PROMPT
from data_engineering_copilot.services.prompt_builder import (
    _DOC_OUTPUT_FORMAT,
    _DOCUMENTATION_INSTRUCTIONS,
    _RAG_PROMPT_TEMPLATE,
    PromptBuilder,
)
from data_engineering_copilot.services.query_rewriting import (
    _CLASSIFY_INTENT_PROMPT,
    _EXPAND_PROMPT,
    _HYDE_PROMPT,
    _REWRITE_CONTEXTUAL_PROMPT,
    _REWRITE_PROMPT,
    QueryRewriter,
)
from data_engineering_copilot.services.rag_evaluation import _FAITHFULNESS_PROMPT, FaithfulnessEvaluator
from data_engineering_copilot.services.scope_verifier import _SCOPE_PROMPT
from tests.doubles.llm import StubLLM

# nosec: B402 — importing service modules is required to trigger register_fallback.

DEFAULT_SYSTEM_ROLE = "You are DataEngineeringCopilot, an expert data engineering assistant."

_ALL_PROMPTS = {
    "rag-answer",
    "query-intent-classify",
    "query-rewrite",
    "query-rewrite-contextual",
    "query-expand",
    "query-hyde",
    "groundedness-nli",
    "chunk-enrichment-summary",
    "eval-faithfulness",
    "judge-faithfulness",
    "judge-relevance",
    "judge-out-of-scope",
    "scope-check",
    "rag-json-retry-suffix",
}

_COMPILE_CASES = {
    "rag-answer": {
        "system_role": DEFAULT_SYSTEM_ROLE,
        "output_format": _DOC_OUTPUT_FORMAT,
        "instructions": _DOCUMENTATION_INSTRUCTIONS,
        "tagged_context": "<chunk>\n[DENSITY: LOW]\nSome docs.\n</chunk>",
        "question": "What is X?",
        "trailing": "",
    },
    "query-intent-classify": {"query": "show me code to read a parquet file"},
    "query-rewrite": {"question": "what is a spark dataframe"},
    "query-rewrite-contextual": {
        "conversation_history": "User: how does filter work?\nAssistant: it filters arrays.",
        "question": "what about its API?",
        "session_topic": "pyspark filter on arrays",
    },
    "query-expand": {"max_variations": 3, "query": "filter arrays of structs"},
    "query-hyde": {"query": "how does windowing work"},
    "groundedness-nli": {"answer": "Spark SQL is a module.", "context": "Spark SQL is a module for data."},
    "chunk-enrichment-summary": {"max_summary_words": 50, "title": "Test Title", "text": "Content here"},
    "eval-faithfulness": {"answer": "answer text", "context": "context text"},
    "judge-faithfulness": {"output": "the answer", "context": "the context"},
    "judge-relevance": {"input": "the question", "output": "the answer"},
    "judge-out-of-scope": {"input": "the question", "output": "the answer"},
    "scope-check": {"question": "What is X?", "context": "Some docs."},
    "rag-json-retry-suffix": {},
}

_FALLBACK_TEMPLATES = {
    "rag-answer": _RAG_PROMPT_TEMPLATE,
    "query-intent-classify": _CLASSIFY_INTENT_PROMPT,
    "query-rewrite": _REWRITE_PROMPT,
    "query-rewrite-contextual": _REWRITE_CONTEXTUAL_PROMPT,
    "query-expand": _EXPAND_PROMPT,
    "query-hyde": _HYDE_PROMPT,
    "groundedness-nli": _NLI_PROMPT,
    "chunk-enrichment-summary": _SUMMARY_PROMPT,
    "eval-faithfulness": _FAITHFULNESS_PROMPT,
    "judge-faithfulness": (
        "You are a faithfulness judge. Determine whether the answer is "
        "supported by the retrieved documentation context. Score 0 to 1 "
        "(1 = fully supported, 0 = hallucinated or unsupported).\n\n"
        + SYSTEM_BLOCK_SEPARATOR
        + "Context:\n{context}\n\n"
        "Answer:\n{output}\n\n"
        'Reply with ONLY a JSON object: {{"score": <0-1>, "reason": "<brief>"}}'
    ),
    "judge-relevance": (
        "You are a relevance judge. Determine whether the answer actually "
        "addresses the user's question. Score 0 to 1 (1 = directly relevant, "
        "0 = off-topic or evasive).\n\n" + SYSTEM_BLOCK_SEPARATOR + "Question:\n{input}\n\n"
        "Answer:\n{output}\n\n"
        'Reply with ONLY a JSON object: {{"score": <0-1>, "reason": "<brief>"}}'
    ),
    "judge-out-of-scope": (
        "You are an out-of-scope detector. Determine whether the user's "
        "question is answerable from the provided documentation. Reply "
        "true if the question is NOT answerable from the docs, false if it is.\n\n"
        + SYSTEM_BLOCK_SEPARATOR
        + "Question:\n{input}\n\n"
        "Answer:\n{output}\n\n"
        'Reply with ONLY a JSON object: {{"out_of_scope": <true|false>, "reason": "<brief>"}}'
    ),
    "scope-check": _SCOPE_PROMPT,
    "rag-json-retry-suffix": _JSON_RETRY_SUFFIX,
}


@pytest.fixture(autouse=True)
def _reset_prompt_cache():
    lp_module._CACHE.clear()
    yield
    lp_module._CACHE.clear()


class _StubClient:
    def __init__(self, prompt_lookup=None):
        self._prompt_lookup = prompt_lookup

    def get_prompt(self, name, label=None, **kwargs):
        if self._prompt_lookup is None:
            return None
        return self._prompt_lookup(name, label, kwargs)


class _StubInstance:
    def __init__(self, prompt_lookup=None):
        self._client = _StubClient(prompt_lookup)


def test_all_prompts_have_registered_fallbacks():
    assert set(lp_module._FALLBACK) == _ALL_PROMPTS
    assert set(lp_module.SEED_PROMPTS) == _ALL_PROMPTS


@pytest.mark.parametrize("name", sorted(_ALL_PROMPTS))
def test_fallback_compile_matches_legacy_format(monkeypatch, name):
    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    kwargs = _COMPILE_CASES[name]
    assert get_langfuse_prompt(name).compile(**kwargs) == _FALLBACK_TEMPLATES[name].format(**kwargs)


@pytest.mark.parametrize("name", sorted(_ALL_PROMPTS))
def test_seed_template_renders_byte_identical_to_fallback(name):
    """The Langfuse-form seed template compiles (via the real SDK parser) to the fallback output."""
    from langfuse.model import TemplateParser

    kwargs = _COMPILE_CASES[name]
    seeded = lp_module.SEED_PROMPTS[name]
    assert TemplateParser.compile_template(seeded, kwargs) == _FALLBACK_TEMPLATES[name].format(**kwargs)


def test_build_rag_prompt_matches_legacy_template(monkeypatch):
    """Verify structure of the offline-fallback rag-answer prompt.

    Salted context tags break byte-identity, so we check structural
    properties instead of exact string equality.
    """
    import re

    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(context="Some docs.", question="What is X?", intent="factual")
    assert "## SYSTEM" in prompt
    assert DEFAULT_SYSTEM_ROLE in prompt
    assert "## CONSTRAINTS" in prompt
    assert "## OUTPUT FORMAT" in prompt
    assert "## INSTRUCTIONS" in prompt
    assert SYSTEM_BLOCK_SEPARATOR in prompt
    assert "## USER QUESTION AND CONTEXT" in prompt
    assert "## YOUR ANSWER" in prompt
    assert "Some docs." in prompt
    assert "What is X?" in prompt
    # Salted tags present, old static tags gone.
    assert re.search(r"<context_data_[0-9a-f]{8}>", prompt) is not None
    assert "<chunk>" not in prompt


def test_compiled_rag_prompt_splits_into_system_and_user(monkeypatch):
    """The rag-answer prompt must split into system + user chat messages."""
    from data_engineering_copilot.infrastructure.llm_client import build_chat_messages

    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt(context="Some docs.", question="What is X?", intent="factual")
    messages = build_chat_messages(prompt)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "## SYSTEM" in messages[0]["content"]
    assert SYSTEM_BLOCK_SEPARATOR not in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "## USER QUESTION AND CONTEXT" in messages[1]["content"]
    assert "What is X?" in messages[1]["content"]


def test_judge_prompts_carry_system_marker(monkeypatch):
    from data_engineering_copilot.infrastructure.llm_client import build_chat_messages

    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    for name in ("judge-faithfulness", "judge-relevance", "judge-out-of-scope"):
        rendered = get_langfuse_prompt(name).compile(**_COMPILE_CASES[name])
        messages = build_chat_messages(rendered)
        assert len(messages) == 2, f"{name} did not split into system+user"
        assert messages[0]["role"] == "system"


def test_returns_langfuse_prompt_when_available(monkeypatch):
    sentinel = SimpleNamespace(name="rag-answer")
    monkeypatch.setattr(
        lp_module,
        "get_langfuse_instance",
        lambda: _StubInstance(prompt_lookup=lambda name, label, kwargs: sentinel),
    )
    assert get_langfuse_prompt("rag-answer") is sentinel


def test_cache_avoids_repeated_instance_lookup(monkeypatch):
    calls = {"n": 0}

    def fake_instance():
        calls["n"] += 1
        return None

    monkeypatch.setattr(lp_module, "get_langfuse_instance", fake_instance)
    for _ in range(3):
        get_langfuse_prompt("rag-answer").compile(**_COMPILE_CASES["rag-answer"])
    assert calls["n"] == 1


def test_seed_prompts_creates_all_text_prompts(monkeypatch):
    created: dict[str, tuple] = {}

    class _SeedClient:
        def create_prompt(self, *, name, prompt, type, labels, commit_message=None):
            created[name] = (prompt, type, labels, commit_message)
            return SimpleNamespace(name=name, version=1)

    class _SeedInstance:
        _client = _SeedClient()

    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: _SeedInstance())
    lp_module.seed_prompts(label="production", commit_message="seed prompts")

    assert set(created) == _ALL_PROMPTS
    for name, (prompt, prompt_type, labels, commit_message) in created.items():
        assert prompt_type == "text"
        assert labels == ["production"]
        assert commit_message == "seed prompts"
        assert prompt == lp_module.SEED_PROMPTS[name]


def test_seed_prompts_raises_when_langfuse_unavailable(monkeypatch):
    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    with pytest.raises(RuntimeError):
        lp_module.seed_prompts()


async def test_async_rewrite_uses_compiled_rewrite_prompt(monkeypatch):
    """Real QueryRewriter sends the query-rewrite-compiled prompt to the LLM."""
    seen: list[str] = []

    class _CapturingLLM(StubLLM):
        async def generate(self, prompt: str, **kwargs: object) -> str:
            seen.append(prompt)
            return "rewritten query"

    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    rw = QueryRewriter(llm_client=_CapturingLLM(), enabled=True, hyde_enabled=False)
    result = await rw.async_rewrite("What is a Spark DataFrame?")
    assert result.decomposed_steps == ("rewritten query",)
    assert seen[0] == _REWRITE_PROMPT.format(question="What is a Spark DataFrame?")


async def test_expand_queries_uses_compiled_expand_prompt(monkeypatch):
    """Real QueryRewriter sends the query-expand-compiled prompt to the LLM."""
    seen: list[str] = []

    class _CapturingLLM(StubLLM):
        async def generate(self, prompt: str, **kwargs: object) -> str:
            seen.append(prompt)
            return "variant one\nvariant two"

    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    rw = QueryRewriter(llm_client=_CapturingLLM(), enabled=True, hyde_enabled=False)
    variations = await rw.expand_queries("filter arrays", max_variations=2)
    assert seen[0] == _EXPAND_PROMPT.format(max_variations=2, query="filter arrays")
    assert variations[0] == "filter arrays"


async def test_faithfulness_prompt_truncates_answer_and_context(monkeypatch):
    """Real FaithfulnessEvaluator passes truncated answer/context into the compiled prompt."""
    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    llm = AsyncMock()
    llm.generate.return_value = '{"supported": 1, "unsupported": 0}'
    evaluator = FaithfulnessEvaluator(llm_client=llm)
    result = await evaluator.evaluate(answer="A" * 5000, context="B" * 5000)
    assert result.faithfulness_score == 1.0
    prompt = llm.generate.call_args[0][0]
    assert "A" * 2000 in prompt
    assert "A" * 2001 not in prompt
    assert "B" * 3000 in prompt
    assert "B" * 3001 not in prompt


async def test_summarizer_uses_compiled_summary_prompt(monkeypatch):
    """Real LLMContextSummarizer sends the chunk-enrichment-summary-compiled prompt."""
    monkeypatch.setattr(lp_module, "get_langfuse_instance", lambda: None)
    llm = AsyncMock()
    llm.generate.return_value = "a summary"
    summarizer = LLMContextSummarizer(llm_client=llm, max_summary_words=30)
    doc = ParsedDocument(source_name="src", title="Test Title", url="http://x", text="Content here")
    await summarizer.summarize(doc)
    prompt = llm.generate.call_args[0][0]
    assert prompt == _SUMMARY_PROMPT.format(max_summary_words=30, title="Test Title", text="Content here")
