"""Central Langfuse prompt-management helper.

Every prompt module registers its hardcoded fallback template string under the
same ``name`` used to seed Langfuse.  ``get_langfuse_prompt(name)`` returns an
object exposing ``compile(**kwargs) -> str`` that renders the Langfuse-managed
template, or the registered constant when Langfuse is disabled/unreachable, so
prompt behavior is byte-identical in degraded mode.

The Langfuse-form templates (``{{var}}`` style) live in ``SEED_PROMPTS`` and
are created/updated idempotently by ``seed_prompts()`` (wired to
``dec langfuse-seed-prompts``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from data_engineering_copilot.infrastructure.llm_client import SYSTEM_BLOCK_SEPARATOR
from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

logger = logging.getLogger(__name__)

# name -> hardcoded .format()-style template (registered by each prompt module).
_FALLBACK: dict[str, str] = {}

# Resolved-prompt cache keyed by (name, label). Langfuse caches get_prompt for
# 60s; we additionally cache so repeated hot-path calls never re-run the
# instance health/auth checks more than once per TTL.
_CACHE: dict[tuple[str, str], tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 60.0


class _FallbackPrompt:
    """Stand-in for langfuse's ``TextPromptClient`` using a ``.format()`` template."""

    def __init__(self, template: str) -> None:
        self._template = template

    def compile(self, **kwargs: Any) -> str:
        return self._template.format(**kwargs)


def register_fallback(name: str, template: str) -> None:
    """Register the hardcoded ``.format()``-style fallback template for a prompt name."""
    _FALLBACK[name] = template


def get_langfuse_prompt(name: str, label: str = "production") -> Any:
    """Return a prompt client with a ``compile(**kwargs) -> str`` interface.

    Prefers the Langfuse-managed prompt (the SDK caches prompts for 60s); falls
    back to the registered hardcoded template when Langfuse is disabled, the
    fetch fails, or the prompt has not been seeded yet.
    """
    key = (name, label)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    prompt: Any
    client = get_langfuse_instance()
    if client is not None:
        try:
            prompt = client._client.get_prompt(name, label=label)
            _CACHE[key] = (now, prompt)
            return prompt
        except Exception as exc:
            logger.warning("Langfuse prompt fetch failed for %r (label=%r): %s", name, label, exc)

    fallback = _FALLBACK.get(name)
    if fallback is None:
        logger.error("No fallback registered for Langfuse prompt %r", name)
        fallback = ""
    prompt = _FallbackPrompt(fallback)
    _CACHE[key] = (now, prompt)
    return prompt


# --- Langfuse-form templates ({{var}} style) used to seed the server. ---

SEED_PROMPTS: dict[str, str] = {
    "rag-answer": "\n".join(
        [
            "## SYSTEM",
            "{{system_role}}",
            "Your role is to answer questions using ONLY the provided documentation context.",
            "",
            "## CONSTRAINTS",
            "1. Base your answer strictly on the provided context.",
            "2. Do NOT invent, assume, or use external knowledge.",
            "3. If information is missing or unclear, explicitly state the limitation.",
            "4. Cite specific documentation sources when possible.",
            "5. Use precise technical terminology from the context.",
            "6. Sparse/Low-Signal Text: If the context contains only raw code snippets, log lines, boilerplate, or insufficient material — do NOT fabricate. Set status to INSUFFICIENT_CONTEXT and list missing information.",
            "7. Ignore API Boilerplate: Discard standard package imports, memory addresses, and log timestamps when evaluating the context.",
            "8. Out-of-Scope Topics: Answer ONLY from the provided context. If the context does not cover the question's topic — even if it contains substantial material on other topics — do NOT answer from general knowledge. Set status to INSUFFICIENT_CONTEXT and state which topic the provided documentation does not cover.",
            "9. Mode/Entity Isolation: Do NOT conflate execution modes or deployments (e.g., YARN vs Kubernetes vs Standalone vs Mesos) or product variants (e.g., Spark vs Airflow vs Delta Lake). State facts only for the mode the provided context describes. If the context does not explicitly compare or cover a mode, do NOT transfer behavior from one mode to another; say the documentation does not address that mode.",
            "",
            "## OUTPUT FORMAT",
            "{{output_format}}",
            "",
            "## INSTRUCTIONS",
            "{{instructions}}",
            "",
            SYSTEM_BLOCK_SEPARATOR,
            "## USER QUESTION AND CONTEXT",
            "Context:\n{{tagged_context}}\n\nQuestion: {{question}}",
            "",
            "## YOUR ANSWER",
        ]
    ),
    "query-intent-classify": (
        'Classify the user query into exactly one category: "code_example" or "factual".\n'
        '- Choose "code_example" if the user wants code snippets, programming examples, or scripts.\n'
        '- Choose "factual" if the user wants conceptual explanations, documentation text, or theory.\n\n'
        + SYSTEM_BLOCK_SEPARATOR
        + 'Query: "{{query}}"\n'
        'JSON Response: {"intent": "code_example" | "factual"}'
    ),
    "query-rewrite": (
        "You are a search query rewriter. Given a user question, produce a concise, "
        "search-optimized query that would best retrieve relevant documentation.\n"
        "Rules:\n"
        "- Return ONLY the rewritten query, no preamble.\n"
        "- Preserve the user's intent.\n"
        "- Expand abbreviations and jargon where helpful.\n"
        "- Output a single line, no more than 30 words.\n\n"
        + SYSTEM_BLOCK_SEPARATOR
        + "User question: {{question}}\n\nRewritten query:"
    ),
    "query-rewrite-contextual": (
        "You are a search query rewriter for a conversational documentation assistant. "
        "Given a conversation history and the user's latest message, rewrite the latest "
        "message into a standalone search query that would retrieve the relevant "
        "documentation on its own.\n"
        "Rules:\n"
        '- Resolve pronouns and references using the history (e.g. "its API" -> the API '
        "of the previously discussed feature).\n"
        '- Use the conversation topic to disambiguate terse follow-ups (e.g. "give an '
        "example in python\" stays about the conversation's subject).\n"
        "- Preserve the user's intent and expand abbreviations/jargon where helpful.\n"
        "- Return ONLY the rewritten query, no preamble.\n"
        "- Output a single line, no more than 40 words.\n\n"
        + SYSTEM_BLOCK_SEPARATOR
        + "## CONVERSATION TOPIC\n{{session_topic}}\n\n"
        "## CONVERSATION HISTORY\n{{conversation_history}}\n\n"
        "Latest user message: {{question}}\n\nRewritten query:"
    ),
    "query-expand": (
        "Generate {{max_variations}} different search queries that would find "
        "the same information as this question. Return ONLY the queries, "
        "one per line, no numbering.\n\n" + SYSTEM_BLOCK_SEPARATOR + "Original question: {{query}}\n\nVariations:"
    ),
    "query-hyde": (
        "Write a short, authoritative paragraph that would perfectly answer "
        "the following question. Do not address the user directly.\n\n" + SYSTEM_BLOCK_SEPARATOR + "Question: {{query}}"
    ),
    "groundedness-nli": (
        "You are a groundedness verifier. Given an answer and supporting context, "
        "determine which claims in the answer are supported by the context.\n\n"
        "For each claim in the answer, output a JSON array of objects:\n"
        '[{"claim": "...", "supported": true/false, "evidence": "..."}]\n'
        "Return ONLY the JSON array, no preamble.\n\n" + SYSTEM_BLOCK_SEPARATOR + "ANSWER:\n{{answer}}\n\n"
        "CONTEXT (excerpted from documentation):\n{{context}}\n\n"
        "JSON array:"
    ),
    "scope-check": (
        "You are a scope verifier for a documentation RAG system. Given a user question "
        "and the retrieved documentation context, decide whether the context covers the "
        "question's topic.\n"
        "The context COVERS the topic when it contains material needed to answer the "
        "question: prose, API reference entries, function signatures, docstrings, or "
        "code samples for the feature being asked about.\n"
        '- "covers": the context directly addresses the question\'s topic (the docs, '
        "API, or code for the feature asked about is present).\n"
        '- "partially": the context is related to the topic but only tangentially — it '
        "touches the feature without directly documenting it. Treated as acceptable.\n"
        '- "does_not_cover": the context has no material on the question\'s topic — it '
        "documents a DIFFERENT product, technology, or feature, even if it mentions "
        "some related terms.\n"
        "Answer ONLY with a JSON object:\n"
        '{"verdict": "covers" | "partially" | "does_not_cover", "reason": "..."}\n\n'
        + SYSTEM_BLOCK_SEPARATOR
        + "QUESTION:\n{{question}}\n\n"
        "CONTEXT (excerpted from documentation):\n{{context}}\n\n"
        "JSON:"
    ),
    "chunk-enrichment-summary": (
        "You are a technical documentation indexer.\n"
        "Provide a direct 1-2 sentence overview (under {{max_summary_words}} words) "
        "of the documentation page below.\n"
        "State ONLY what main concepts, components, or procedures are documented.\n"
        "INTERNAL STYLE: flat, factual, no introductory fluff.\n"
        "If the page lacks substantive content beyond navigation links, headers, "
        "or index listings, return exactly: NO_CONTENT_TO_SUMMARIZE\n\n" + SYSTEM_BLOCK_SEPARATOR + "Title: {{title}}\n"
        "Content:\n{{text}}\n\n"
        "Summary:"
    ),
    "eval-faithfulness": (
        "Given the answer and context below, count how many claims in the "
        "answer are supported by the context.\n\n"
        + SYSTEM_BLOCK_SEPARATOR
        + "ANSWER: {{answer}}\n\nCONTEXT: {{context}}\n\n"
        'Return JSON: {"supported": N, "unsupported": N}'
    ),
    "judge-faithfulness": (
        "You are a faithfulness judge. Determine whether the answer is "
        "supported by the retrieved documentation context. Score 0 to 1 "
        "(1 = fully supported, 0 = hallucinated or unsupported).\n\n"
        + SYSTEM_BLOCK_SEPARATOR
        + "Context:\n{{context}}\n\n"
        "Answer:\n{{output}}\n\n"
        'Reply with ONLY a JSON object: {"score": <0-1>, "reason": "<brief>"}'
    ),
    "judge-relevance": (
        "You are a relevance judge. Determine whether the answer actually "
        "addresses the user's question. Score 0 to 1 (1 = directly relevant, "
        "0 = off-topic or evasive).\n\n" + SYSTEM_BLOCK_SEPARATOR + "Question:\n{{input}}\n\n"
        "Answer:\n{{output}}\n\n"
        'Reply with ONLY a JSON object: {"score": <0-1>, "reason": "<brief>"}'
    ),
    "judge-out-of-scope": (
        "You are an out-of-scope detector. Determine whether the user's "
        "question is answerable from the provided documentation. Reply "
        "true if the question is NOT answerable from the docs, false if it is.\n\n"
        + SYSTEM_BLOCK_SEPARATOR
        + "Question:\n{{input}}\n\n"
        "Answer:\n{{output}}\n\n"
        'Reply with ONLY a JSON object: {"out_of_scope": <true|false>, "reason": "<brief>"}'
    ),
    "rag-json-retry-suffix": (
        "\n\nIMPORTANT: Your previous response was not valid JSON. "
        "Return ONLY raw JSON with no markdown, no code fences, no preamble."
    ),
}


def seed_prompts(label: str = "production", commit_message: str | None = None) -> dict[str, Any]:
    """Idempotently create/update every managed prompt under ``label``.

    Re-running creates a new version of each prompt. Returns ``{name: created}``.
    """
    client = get_langfuse_instance()
    if client is None:
        raise RuntimeError("Langfuse is unavailable; cannot seed prompts")
    created: dict[str, Any] = {}
    for name, template in SEED_PROMPTS.items():
        created[name] = client._client.create_prompt(
            name=name,
            prompt=template,
            type="text",
            labels=[label],
            commit_message=commit_message,
        )
    return created
