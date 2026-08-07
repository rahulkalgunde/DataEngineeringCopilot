"""LLM-as-a-judge evaluators over production traces (Phase 7).

Defines the three judge evaluators (faithfulness, relevance, out-of-scope)
driven by Langfuse-managed prompts (``judge-*``, with hardcoded fallbacks) and
the repo's purpose-``evaluation`` LLM fallback chain, plus
``run_batched_trace_evaluation`` which feeds sampled ``rag-query-pipeline``
traces through them via the v4 ``run_batched_evaluation`` API and writes the
judged scores back onto each trace.

See ``docs/langfuse_evaluators.md`` for the definitions.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance
from data_engineering_copilot.observability.langfuse_prompts import get_langfuse_prompt, register_fallback

logger = logging.getLogger(__name__)

# Default filter targets the main RAG answer pipeline (streaming variant
# ``rag-query-pipeline-stream`` is excluded). Format: Langfuse public-API
# filter array passed straight through to ``trace.list``.
JUDGE_FILTER_DEFAULT = '[{"type": "string", "column": "name", "operator": "=", "value": "rag-query-pipeline"}]'

# Cap the retrieved-context passed to judges so prompt size stays bounded even
# when many chunks were retrieved.
MAX_JUDGE_CONTEXT_CHARS = 12000


def _coerce_answer(output: Any) -> Any:
    """Extract the real answer text from structured trace outputs.

    Traces may store the answer as a plain string, or as a dict produced by
    ``parse_rag_response`` (e.g. ``{"status": ..., "answer": "..."}``). Prefer
    the ``answer`` key when present so judges see the actual text.
    """
    if isinstance(output, dict):
        answer = output.get("answer")
        if answer is not None:
            return answer
        for key in ("text", "content"):
            if output.get(key):
                return output[key]
    return output


register_fallback(
    "judge-faithfulness",
    "You are a faithfulness judge. Determine whether the answer is "
    "supported by the retrieved documentation context. Score 0 to 1 "
    "(1 = fully supported, 0 = hallucinated or unsupported).\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{output}\n\n"
    'Reply with ONLY a JSON object: {{"score": <0-1>, "reason": "<brief>"}}',
)
register_fallback(
    "judge-relevance",
    "You are a relevance judge. Determine whether the answer actually "
    "addresses the user's question. Score 0 to 1 (1 = directly relevant, "
    "0 = off-topic or evasive).\n\n"
    "Question:\n{input}\n\n"
    "Answer:\n{output}\n\n"
    'Reply with ONLY a JSON object: {{"score": <0-1>, "reason": "<brief>"}}',
)
register_fallback(
    "judge-out-of-scope",
    "You are an out-of-scope detector. Determine whether the user's "
    "question is answerable from the provided documentation. Reply "
    "true if the question is NOT answerable from the docs, false if it is.\n\n"
    "Question:\n{input}\n\n"
    "Answer:\n{output}\n\n"
    'Reply with ONLY a JSON object: {{"out_of_scope": <true|false>, "reason": "<brief>"}}',
)

_LLM_HOLDER: dict[str, Any] = {}


def _evaluation_llm():
    """Lazy-build (and cache) the purpose-``evaluation`` LLM fallback chain.

    No provider is pinned: every call routes through ``llm_fallback_order`` and
    picks the first currently-available provider (local Ollama as last resort),
    matching the RAGAS evaluation behavior.
    """
    llm = _LLM_HOLDER.get("llm")
    if llm is not None:
        return llm
    from data_engineering_copilot.factory import (
        _build_provider_health_registry,
        _build_provider_rate_limiters,
        build_llm_fallback_chain,
    )

    rate_limiters = _build_provider_rate_limiters(settings)
    health_registry = _build_provider_health_registry(settings)
    llm = build_llm_fallback_chain(
        purpose="evaluation",
        app_settings=settings,
        provider_rate_limiters=rate_limiters,
        health_registry=health_registry,
        purpose_provider=settings.evaluation_llm_provider,
        purpose_model=settings.evaluation_llm_model,
    )
    _LLM_HOLDER["llm"] = llm
    return llm


def _parse_float_score(text: str) -> float:
    """Extract a 0-1 score from a judge response (JSON or bare number)."""
    if not text:
        return 0.0
    match = re.search(r'"score"\s*:\s*([01](?:\.\d+)?)', text)
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    match = re.search(r"\b([01](?:\.\d+)?)\b", text)
    return min(1.0, max(0.0, float(match.group(1)))) if match else 0.0


def _parse_bool(text: str) -> bool:
    """Extract a boolean from an out-of-scope judge response."""
    if not text:
        return False
    match = re.search(r'"out_of_scope"\s*:\s*(true|false)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return "true" in text.lower()


async def _run_judge(prompt_name: str, template_kwargs: dict[str, Any]) -> str:
    """Compile the Langfuse-managed judge prompt and run it through the chain."""
    prompt = get_langfuse_prompt(prompt_name)
    rendered = prompt.compile(**template_kwargs)
    llm = _evaluation_llm()
    if llm is None:
        logger.warning("No evaluation LLM available for %r — judging as 0", prompt_name)
        return ""
    return str(await llm.generate(rendered))


async def faithfulness_judge(*, input, output, expected_output=None, metadata=None, **kwargs):
    """Score how well the answer is supported by the retrieved context."""
    from langfuse import Evaluation

    context = ""
    if metadata:
        context = metadata.get("retrieved_contexts") or metadata.get("context") or ""
    response = await _run_judge("judge-faithfulness", {"output": output, "context": context})
    return Evaluation(name="faithfulness", value=_parse_float_score(response), comment="llm-as-judge")


async def relevance_judge(*, input, output, expected_output=None, metadata=None, **kwargs):
    """Score how directly the answer addresses the user's question."""
    from langfuse import Evaluation

    response = await _run_judge("judge-relevance", {"input": input, "output": output})
    return Evaluation(name="relevance", value=_parse_float_score(response), comment="llm-as-judge")


async def out_of_scope_judge(*, input, output, expected_output=None, metadata=None, **kwargs):
    """Return a boolean score marking whether the question is out of scope."""
    from langfuse import Evaluation

    response = await _run_judge("judge-out-of-scope", {"input": input, "output": output})
    return Evaluation(
        name="out_of_scope",
        value=_parse_bool(response),
        data_type="BOOLEAN",
        comment="llm-as-judge",
    )


def _trace_mapper(*, item, **kwargs):
    """Map a v4 trace to ``EvaluatorInputs``.

    Extracts the retrieved contexts from the ``retrieval`` observation (its
    output is the list of chunk texts) so the faithfulness judge has evidence.
    """
    from langfuse.batch_evaluation import EvaluatorInputs

    context = ""
    observations = getattr(item, "observations", None) or []
    for obs in observations:
        if getattr(obs, "name", None) == "retrieval":
            obs_output = getattr(obs, "output", None)
            if isinstance(obs_output, list):
                context = "\n".join(str(c) for c in obs_output)
            elif obs_output:
                context = str(obs_output)
            context = context[:MAX_JUDGE_CONTEXT_CHARS]
            break

    metadata = dict(getattr(item, "metadata", None) or {})
    metadata["retrieved_contexts"] = context
    metadata["trace_id"] = getattr(item, "id", None)
    return EvaluatorInputs(
        input=getattr(item, "input", None),
        output=_coerce_answer(getattr(item, "output", None)),
        metadata=metadata,
    )


def run_batched_trace_evaluation(
    filter: str | None = None,
    max_items: int | None = None,
    max_concurrency: int = 5,
    verbose: bool = False,
):
    """Run the three judges over production ``rag-query-pipeline`` traces.

    Cost is gated by ``settings.langfuse_sample_rate`` when ``max_items`` is not
    explicitly requested — traces are sampled with probability 1 - sample_rate
    skipped entirely (returns None). When ``max_items`` is given the run is
    always executed (explicit operator intent).

    Returns a ``BatchEvaluationResult`` from the v4 SDK, or ``None`` when
    Langfuse is unavailable or the run is sampled out.
    """
    if max_items is None and random.random() >= settings.langfuse_sample_rate:
        logger.info("Production trace evaluation sampled out (sample_rate=%s)", settings.langfuse_sample_rate)
        return None

    client = get_langfuse_instance()
    if client is None:
        logger.warning("Langfuse client not available, cannot run batch trace evaluation")
        return None

    return client._client.run_batched_evaluation(
        scope="traces",
        mapper=_trace_mapper,
        filter=filter or JUDGE_FILTER_DEFAULT,
        fetch_trace_fields="core,io,scores,observations,metrics",
        max_items=max_items,
        evaluators=[faithfulness_judge, relevance_judge, out_of_scope_judge],
        max_concurrency=max_concurrency,
        metadata={"judge": "llm-as-judge", "sample_rate": settings.langfuse_sample_rate},
        verbose=verbose,
    )
