from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any, cast

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.exceptions import LLMGenerationError, RetrievalError
from data_engineering_copilot.domain.models import (
    Answer,
    CachedAnswer,
    CacheScope,
    ChatMessage,
    DocumentChunk,
    LLMUsage,
    RagConfig,
    RetrievalFilters,
    RetrievedChunk,
)
from data_engineering_copilot.domain.protocols import (
    EmbedderProtocol,
    LLMClientProtocol,
    PiiRedactorProtocol,
    RerankerProtocol,
    TelemetryTracerProtocol,
    VectorStoreProtocol,
)
from data_engineering_copilot.infrastructure.llm_client import SYSTEM_BLOCK_SEPARATOR
from data_engineering_copilot.observability.langfuse_prompts import get_langfuse_prompt, register_fallback
from data_engineering_copilot.observability.token_tracker import RetrievalTracker, TokenTracker
from data_engineering_copilot.services.context_assembler import DEFAULT_ITEM_LIMIT_CHARS, ContextAssembler
from data_engineering_copilot.services.context_compression import ContextCompressor
from data_engineering_copilot.services.groundedness import GroundednessVerifier
from data_engineering_copilot.services.input_guardrails import InputGuardrails
from data_engineering_copilot.services.prompt_builder import CODE_INTENTS, PromptBuilder
from data_engineering_copilot.services.query_cache import _NON_ANSWER_MARKERS
from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
from data_engineering_copilot.services.query_rewriting import (
    QueryRewriter,
    is_degenerate_query,
    render_conversation_history,
)
from data_engineering_copilot.services.query_signals import (
    RRF_EQUAL_PROFILE,
    classify_query_signals,
    select_rrf_profile,
)
from data_engineering_copilot.services.scope_verifier import ScopeVerifier
from data_engineering_copilot.services.structured_output import parse_rag_response, verify_citations

# Offline fallback for the Langfuse-managed ``rag-json-retry-suffix`` prompt.
_JSON_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous response was not valid JSON. "
    "Return ONLY raw JSON with no markdown, no code fences, no preamble."
)

register_fallback("rag-json-retry-suffix", _JSON_RETRY_SUFFIX)

# Refusal text emitted by the topic-scope gate. Both the CLI and the Spark eval
# detect INSUFFICIENT_CONTEXT via the "cannot answer" / "Missing information:"
# / "INSUFFICIENT_CONTEXT" markers.
_SCOPE_REFUSAL_TEXT = (
    '{"status": "INSUFFICIENT_CONTEXT", "answer": "", "missing_info": "The retrieved '
    'documentation does not cover the topic of the question."}\n\n'
    "I cannot answer this question because the provided documentation does not cover "
    "its topic.\n\n"
    "Missing information: The retrieved documentation does not contain material on "
    "the topic of the question, so I cannot answer it from the knowledge base."
)

# How long to wait for the cross-encoder model to load before degrading to
# "no reranking". The model is cached locally after the first download, so
# this only bites on a cold cache with a slow network.
_RERANKER_INIT_TIMEOUT_SECONDS = 120.0


def merge_retrieval_results(
    results: list[list[RetrievedChunk]],
    original_query: str,
) -> list[RetrievedChunk]:
    """Merge per-query retrieval results using rank fusion plus an original-query bonus.

    Each result list is ranked in retrieval order. Chunks are scored by the sum
    of ``1 / (rank + 3)`` across all queries that retrieved them. Chunks that
    match the original query (retrieved by the original query variant) receive
    an extra bonus. Chunks whose text defines a Spark function named in the
    query (e.g. ``def filter(``) receive an additional lexical bonus so the
    real docstring implementation surfaces above unrelated function-family
    chunks (e.g. ``inline``, ``array_*``). Returns chunks sorted by fused score
    descending.
    """
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}
    for query_index, result_list in enumerate(results):
        for rank, result in enumerate(result_list):
            cid = result.chunk.chunk_id
            chunk_by_id[cid] = result
            score = 1.0 / (rank + 3.0)
            # Original-query bonus for the first result set.
            if query_index == 0:
                score += 0.5
            scores[cid] = scores.get(cid, 0.0) + score

    merged = sorted(chunk_by_id.values(), key=lambda c: -scores.get(c.chunk.chunk_id, 0.0))
    return merged


logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA_VERSION = "1"


def _chunk_provenance_ref(result: RetrievedChunk, rank: int) -> dict[str, object]:
    """Compact, JSON-safe per-candidate reference for retrieval provenance."""
    return {
        "rank": rank,
        "chunk_id": result.chunk.chunk_id,
        "url": result.chunk.url,
        "source_name": result.chunk.source_name,
        "distance": result.distance,
        "confidence": result.confidence,
    }


def _retrieval_details(retrieved_chunks: list[RetrievedChunk], *, limit: int = 50) -> tuple[dict, ...]:
    """Visualizer-facing per-candidate details (rank, scores, snippet)."""
    details: list[dict] = []
    for rank, result in enumerate(retrieved_chunks[:limit]):
        details.append(
            {
                "rank": rank,
                "chunk_id": result.chunk.chunk_id,
                "source_name": result.chunk.source_name,
                "title": result.chunk.title,
                "url": result.chunk.url,
                "distance": result.distance,
                "confidence": result.confidence,
                "word_count": result.chunk.word_count,
                "text_snippet": result.chunk.text[:240],
            }
        )
    return tuple(details)


def _usage_to_dict(usage: LLMUsage | None) -> dict[str, object]:
    """Serialize an ``LLMUsage`` dataclass for the ``Answer.token_usage`` field."""
    if usage is None:
        return {}
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "model": usage.model,
        "duration_ms": usage.duration_ms,
        "tokens_per_second": usage.tokens_per_second,
    }


def _l2_norm(vector: list[float]) -> float:
    return sum(v * v for v in vector) ** 0.5


_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
]


def _scrub_pii(text: str) -> str:
    """Redact common PII patterns (email, phone, SSN) from trace input."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


_INTENT_CONFIG_ID_UNSET = "__unset__"
_intent_config_id_cache: str | None = _INTENT_CONFIG_ID_UNSET  # type: ignore[assignment]


def _get_intent_config_id() -> str | None:
    """Lazily resolve (and cache) the ``intent`` score-config id for categorical scoring.

    Returns ``None`` when it cannot be resolved so the caller can fall back.
    """
    global _intent_config_id_cache
    if _intent_config_id_cache is not _INTENT_CONFIG_ID_UNSET:
        return _intent_config_id_cache
    from data_engineering_copilot.evaluation.langfuse_score_configs import get_score_config_id

    _intent_config_id_cache = get_score_config_id("intent")
    return _intent_config_id_cache


def _llm_model_name(llm_client: Any) -> str:
    """Best-effort model name from an LLM client (LLMClient, ProviderFallbackChain)."""
    for attr in ("model", "model_name"):
        value = getattr(llm_client, attr, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def _clean_chat_text(raw: str) -> str:
    """P0-B: unwrap the RAG JSON envelope for user display.

    ``parse_rag_response`` extracts ``answer`` from the
    ``{"status","answer","missing_info"}`` envelope (doc intents); for code
    intents or unparseable text it falls back to the raw text unchanged.
    Handles refusal text that mixes a JSON object with trailing prose.
    Strips a trailing ``Sources: [...]`` line the LLM appends (UI shows
    sources separately).
    """
    if not raw:
        return ""
    result = raw
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        # JSON-prefixed (envelope or refusal-with-prose): strip the leading JSON
        # object, then parse any remaining prose.
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(stripped):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        rest = stripped[i + 1 :].lstrip()
                        result = rest if rest else raw
                        break
    parsed = parse_rag_response(result)
    if parsed.answer:
        result = parsed.answer
    return _strip_sources_line(result)


_SOURCES_LINE_RE = re.compile(r"\n\s*Sources:\s*\[.*\]\s*$", re.IGNORECASE | re.DOTALL)


def _strip_sources_line(text: str) -> str:
    """Remove a trailing ``Sources: [...]`` line from an answer (UI shows sources separately)."""
    if not text:
        return text
    cleaned = _SOURCES_LINE_RE.sub("", text)
    return cleaned.rstrip()


# Identity questions: the assistant's identity is fixed and never answered from
# retrieved documents. These are intercepted before retrieval and answered with
# the chat persona directly.
_IDENTITY_QUESTION_RE = re.compile(
    r"^(who|what)\s+are\s+you|are\s+you\s+(claude|anthropic|gpt|chatgpt|an? (ai )?assistant|a robot)|"
    r"your\s+name|what('s| is) your (name|identity|role)|identify yourself|say\s+your\s+name",
    re.IGNORECASE,
)

_IDENTITY_ANSWER = (
    "I am DataEngineeringCopilot, an expert data engineering assistant for "
    "Apache Spark, Apache Airflow, and Delta Lake documentation."
)


# ChatGPT-style follow-up suggestion prompt. Grounded in the RETRIEVED context
# (the same documentation chunks that grounded the answer) so every suggested
# follow-up is answerable from the knowledge repo — never invented by the LLM.
# Uses the conversation trajectory (history), the query intent, and a logical-
# expansion taxonomy (next step / alternative / edge case) like ChatGPT/Gemini.
_SUGGESTION_PROMPT = (
    "You are a documentation assistant. Given the conversation history, the "
    "retrieved documentation context, the user's question, the answer, and the "
    "question's intent, suggest {count} short follow-up questions the user might "
    "logically ask next.\n"
    "Rules:\n"
    "- ONLY suggest questions that can be answered from the provided context. "
    "If the context does not mention a topic, do not suggest it.\n"
    "- Prefer logical next steps given the conversation trajectory (deepen the "
    "current topic, alternatives, edge cases, usage examples).\n"
    "- Use the intent to steer: for 'how_to'/'code_example' suggest examples or "
    "variations; for 'comparative' suggest the comparison alternatives; for "
    "'api_lookup' suggest related functions/parameters; for 'debugging' suggest "
    "causes/fixes; for 'factual' suggest deeper or related topics.\n"
    "- Vary the kinds of follow-ups (do not repeat the same shape).\n"
    "- Each must be a standalone question, one per line, no numbering, no "
    "preamble, max 12 words each.\n\n" + SYSTEM_BLOCK_SEPARATOR + "## CONVERSATION HISTORY\n{history}\n\n"
    "## RETRIEVED CONTEXT\n{context}\n\n"
    "## QUESTION\n{question}\n\n"
    "## INTENT\n{intent}\n\n"
    "## ANSWER\n{answer}\n\n"
    "Suggested follow-ups:"
)


# P3: deterministic domain-coherence guard. A question that strongly signals a
# data-engineering domain must not be answered from an unrelated domain's docs.
_DE_DOMAIN_TERMS = re.compile(
    r"\b(spark|pyspark|dataframe|delta|airflow|dag|scala|sql|etl|executor|shuffle)\b", re.IGNORECASE
)
_FOREIGN_DOMAIN_MARKERS = re.compile(r"\b(anthropic|claude|prompt cache|mcp connector)\b", re.IGNORECASE)


def _domain_mismatch(chunks) -> bool:
    """Return True when the top retrieved chunks are dominated by a foreign
    domain (e.g. Anthropic/Claude docs) despite a data-engineering query context.

    Inspects the top-ranked chunks (those that actually reach the prompt), not
    the full fused candidate pool, so a wide pool that legitimately mixes
    domains does not falsely trigger the refusal. Used as a cheap fail-safe:
    if the corpus was cross-tainted (mixed-domain index), refuse rather than
    answer from irrelevant docs.
    """
    if not chunks:
        return False
    top = chunks[: min(len(chunks), 8)]
    foreign_score = sum(1 for c in top if _FOREIGN_DOMAIN_MARKERS.search(c.chunk.source_name or ""))
    de_score = sum(1 for c in top if _DE_DOMAIN_TERMS.search(c.chunk.text or ""))
    return foreign_score > 0 and foreign_score >= de_score


def _rerank_pool_size(retrieval_top_k: int, reranker_top_k: int, configured_pool: int = 0) -> int:
    """Candidate pool handed to the cross-encoder reranker.

    The pool is intentionally wider than ``retrieval_top_k`` so that URLs that
    just miss the dense/sparse cutoff can still be rescued by reranking. The
    retrieval multiplier is ``* 4`` (tuned for CPU-only hosts: keeping the pool
    wider than needed costs real cross-encoder inference time on hosts without
    a GPU forcing rerank to the actual dense cutoff), while the reranker
    multiplier of ``* 8`` guarantees a relevant-but-poorly-scored document
    (fused at rank ~175) stays inside the pool without running the cross-encoder
    over the whole corpus.

    When ``configured_pool`` is positive it overrides the formula, giving the
    operator direct control over cross-encoder inference cost.
    """
    if configured_pool > 0:
        return configured_pool
    return max(retrieval_top_k * 4, reranker_top_k * 8)


def _cap_rejoined_block(text: str, limit_chars: int) -> str:
    """Truncate a rejoined sibling block to ``limit_chars``.

    Sibling rejoin restores surrounding context around a matched segment, but a
    large parent (e.g. a whole source file split into dozens of segments) can
    inflate the block far beyond the per-segment item limit. Capping preserves
    the rejoin's coverage benefit without ever producing an over-limit segment,
    which would violate the ``ContextAssembler`` item-limit invariant.
    """
    if len(text) <= limit_chars:
        return text
    truncated = text[:limit_chars].rstrip()
    if not truncated:
        return text[:limit_chars]
    return truncated


class AsyncRagService:
    def __init__(
        self,
        config: RagConfig,
        vector_store: VectorStoreProtocol,
        llm_client: LLMClientProtocol,
        embedder: EmbedderProtocol,
        reranker: RerankerProtocol | None = None,
        telemetry: TelemetryTracerProtocol | None = None,
        cache: TwoTierCache | None = None,
        query_rewriter: QueryRewriter | None = None,
        groundedness_verifier: GroundednessVerifier | None = None,
        scope_verifier: ScopeVerifier | None = None,
        context_compressor: ContextCompressor | None = None,
        token_tracker: TokenTracker | None = None,
        retrieval_tracker: RetrievalTracker | None = None,
        code_llm_client: LLMClientProtocol | None = None,
        evaluation_llm_client: LLMClientProtocol | None = None,
        pii_redactor: PiiRedactorProtocol | None = None,
        input_guardrails: InputGuardrails | None = None,
        review_dataset_hook: Callable[[str, str, str], object] | None = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.code_llm_client = code_llm_client
        self.evaluation_llm_client = evaluation_llm_client
        self.embedder = embedder
        self.reranker = reranker
        self.telemetry = telemetry
        self.cache = cache
        self.query_rewriter = query_rewriter
        self.groundedness_verifier = groundedness_verifier
        self.scope_verifier = scope_verifier
        self.context_compressor = context_compressor
        self.token_tracker = token_tracker
        self.retrieval_tracker = retrieval_tracker
        self._pii_redactor = pii_redactor
        self.input_guardrails = input_guardrails
        self.review_dataset_hook = review_dataset_hook
        self._prompt_builder = PromptBuilder()
        self._rejoin_item_limit_chars = DEFAULT_ITEM_LIMIT_CHARS

    def _rrf_profile_for(self, query: str) -> str:
        """Select the hybrid RRF profile for a query variant.

        Weighted (identifier-sparse) fusion is applied only when enabled in the
        config; otherwise every query uses the default equal-RRF profile.
        """
        if self.config.identifier_sparse_rrf_enabled:
            return select_rrf_profile(classify_query_signals(query))
        return RRF_EQUAL_PROFILE

    async def _retrieve_variant_queries(
        self,
        queries_to_run: list[str],
        variant_labels: list[str],
        *,
        source_filter: list[str] | None,
        chunk_type_filter: str | None,
        metadata_filters: RetrievalFilters | None,
    ) -> tuple[
        list[list[RetrievedChunk]],
        list[dict[str, object]],
        list,
        set[str],
        bool,
        Exception | None,
        int,
        list[float],
    ]:
        """Embed and retrieve every query variant.

        Returns the per-variant result sets, provenance entries, deduplicated
        retrieved chunks (and seen ids), success flag, last error, the observed
        embedding dimension, and the last embedded query vector. The original
        query is always embedded; HyDE variants are additional queries, never
        replacements.
        """
        per_query_results: list[list[RetrievedChunk]] = []
        prov_variants: list[dict[str, object]] = []
        all_retrieved: list = []
        seen_ids: set[str] = set()
        any_success = False
        last_error: Exception | None = None
        query_dim = 0
        q_emb: list[float] = []
        for i, q in enumerate(queries_to_run):
            try:
                profile = self._rrf_profile_for(q)
                q_emb = await self.embedder.embed_query(q)
                if query_dim == 0 and q_emb:
                    query_dim = len(q_emb)
                results = await self.vector_store.query(
                    q_emb,
                    top_k=self.config.retrieval_top_k,
                    query_text=q,
                    source_filter=source_filter,
                    chunk_type_filter=chunk_type_filter,
                    metadata_filters=metadata_filters,
                    fused_limit=_rerank_pool_size(
                        self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
                    ),
                    rrf_profile=profile,
                )
                # Unfiltered fallback: if metadata filters removed everything,
                # retry once without inferred filters.
                if not results and metadata_filters is not None and not metadata_filters.is_empty:
                    results = await self.vector_store.query(
                        q_emb,
                        top_k=self.config.retrieval_top_k,
                        query_text=q,
                        source_filter=source_filter,
                        chunk_type_filter=chunk_type_filter,
                        fused_limit=_rerank_pool_size(
                            self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
                        ),
                        rrf_profile=profile,
                    )
                any_success = True
                per_query_results.append(results)
                prov_variants.append(
                    {
                        "variant": variant_labels[i],
                        "query": q,
                        "rrf_profile": profile,
                        "retrieved": [_chunk_provenance_ref(r, rank) for rank, r in enumerate(results)],
                    }
                )
                for r in results:
                    cid = r.chunk.chunk_id
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_retrieved.append(r)
            except Exception as sub_exc:
                last_error = sub_exc
                logger.warning("Failed to retrieve for sub-query %r: %s", q[:50], sub_exc)
        return (
            per_query_results,
            prov_variants,
            all_retrieved,
            seen_ids,
            any_success,
            last_error,
            query_dim,
            q_emb,
        )

    async def _record_low_confidence_review(self, trace, question: str, answer_text: str) -> None:
        """Phase 6 (Task 6.3): queue low-confidence answers into the review dataset.

        Fail-open and off the request path: the hook runs in a worker thread and
        any failure is logged and ignored.
        """
        if self.review_dataset_hook is None:
            return
        trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None) if trace else None
        if not trace_id:
            return
        try:
            await asyncio.to_thread(self.review_dataset_hook, trace_id, question, answer_text)
        except Exception as exc:
            logger.warning("Failed to create low-confidence review item: %s", exc)

    async def _reject_low_confidence(
        self,
        retrieved_chunks: list[RetrievedChunk],
        *,
        rerank_used: bool,
        trace,
        question: str,
        emit_provenance: Callable[[], None],
    ) -> Answer | None:
        """Gate the top chunk against the confidence threshold and, when it
        fails, return the out-of-repository ``Answer`` (else ``None``).

        Reranker scores are min-max normalized within the candidate pool (see
        ``LLMReranker._apply`` and ``CrossEncoderReranker.rerank``), so the
        ``reranker_confidence_threshold`` has the same meaning across providers:
        a uniformly scaled score well below the threshold indicates the best
        available chunk is weakly relevant. Without a reranker the gate falls
        back to the embedding/fused ``confidence_threshold``.
        """
        if not retrieved_chunks:
            return None
        gate_threshold = self.config.reranker_confidence_threshold if rerank_used else self.config.confidence_threshold
        if retrieved_chunks[0].confidence >= gate_threshold:
            return None
        if trace:
            trace.update(
                output=(f"Low confidence: {retrieved_chunks[0].confidence:.4f} < threshold {gate_threshold:.4f}")
            )
            trace.end()
        emit_provenance()
        await self._record_low_confidence_review(
            trace, question, "I cannot answer this question because it is outside my knowledge repository."
        )
        return Answer(
            text="I cannot answer this question because it is outside my knowledge repository.",
            sources=tuple(),
            confidence=0.0,
        )

    async def _record_cache_hit_trace(self, question: str, cached, cache_scope, *, semantic: bool) -> None:
        """Phase 7 (Task 7.4): lightweight cache-hit trace with a boolean ``cache_hit`` score.

        Cache hits return before the full ``rag-query-pipeline`` trace exists, so
        record them under a distinct name with ``cache_hit=true`` for cache
        analytics. Fail-open and off the hot path (best-effort).
        """
        if not self.telemetry:
            return
        try:
            trace = self.telemetry.start_observation(
                name="rag-query-pipeline-cache-hit",
                input=_scrub_pii(question),
                as_type="trace",
                tags=["app:data-engineering-copilot"],
                metadata={
                    "cache_tier": "semantic" if semantic else "exact",
                    "cache_scope": cache_scope.value if cache_scope else None,
                    "app_env": settings.langfuse_environment,
                },
            )
            trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None)
            if trace_id:
                self.telemetry.score(trace_id=trace_id, name="cache_hit", value=True, data_type="BOOLEAN")
            if hasattr(trace, "end"):
                trace.end()
            await self.telemetry.flush_async()
        except Exception as exc:
            logger.warning("Failed to record cache-hit trace: %s", exc)

    async def answer(
        self,
        question: str,
        on_step: Callable[[str], None] | None = None,
        on_step_detail: Callable[[str, dict], None] | None = None,
        source_filter: list[str] | None = None,
        cache_scope: CacheScope | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        provenance: list[dict] | None = None,
        bypass_cache: bool = False,
        expected_urls: list[str] | None = None,
        retrieval_only: bool = False,
    ) -> Answer:
        _t0 = time.monotonic()
        _stage_times: dict[str, float] = {}

        def _record_stage(name: str) -> None:
            _stage_times[name] = round((time.monotonic() - _t0) * 1000, 1)

        # Opt-in retrieval provenance capture (used by evaluation). Production
        # callers pass nothing and are unaffected.
        effective_query = question
        _prov_cache_hit = False
        _prov_variants: list[dict[str, object]] = []
        _prov_fused: list[dict[str, object]] = []
        _prov_rerank: dict[str, object] | None = None
        _prov_final: list[dict[str, object]] = []
        _prov_dropped: list[dict[str, object]] = []
        _prov_expected_urls = expected_urls or []

        def _emit_provenance() -> None:
            if provenance is None:
                return
            provenance.append(
                {
                    "schema_version": PROVENANCE_SCHEMA_VERSION,
                    "question": question,
                    "effective_query": effective_query,
                    "cache_hit": _prov_cache_hit,
                    "query_variants": _prov_variants,
                    "fused": _prov_fused,
                    "rerank": _prov_rerank,
                    "final_context": _prov_final,
                    "dropped": _prov_dropped,
                    "expected_urls": _prov_expected_urls,
                    "candidate_pool_size": _prov_pool,
                    "stage_times": dict(_stage_times),
                }
            )

        def _emit_detail(kind: str, payload: dict[str, object]) -> None:
            """Forward a per-stage input/output snapshot to the visualizer."""
            if on_step_detail is not None:
                on_step_detail(kind, payload)

        query_emb_for_cache: list[float] | None = None
        query_dim: int = 0
        q_emb: list[float] = []
        cache_active = self.cache is not None and self.config.cache_enabled and not bypass_cache
        if cache_active and self.cache is not None:
            # Exact tier first — no embedding round-trip on an exact hit.
            cached = await self.cache.aget(question, scope=cache_scope)
            if cached is not None:
                logger.info("cache_hit question=%r", question[:80])
                _record_stage("cache_lookup")
                _prov_cache_hit = True
                _emit_provenance()
                await self._record_cache_hit_trace(question, cached, cache_scope, semantic=False)
                return Answer(
                    text=cached.text,
                    sources=cached.sources,
                    confidence=cached.confidence,
                    groundedness_score=cached.groundedness_score,
                    stage_times=_stage_times,
                )
            # Exact miss: only now pay the embedding cost for semantic lookup.
            with contextlib.suppress(Exception):
                query_emb_for_cache = await self.embedder.embed_query(question)
            cached = await self.cache.aget(question, query_embedding=query_emb_for_cache, scope=cache_scope)
            if cached is not None:
                logger.info("cache_hit_semantic question=%r", question[:80])
                _record_stage("cache_lookup")
                _prov_cache_hit = True
                _emit_provenance()
                await self._record_cache_hit_trace(question, cached, cache_scope, semantic=True)
                return Answer(
                    text=cached.text,
                    sources=cached.sources,
                    confidence=cached.confidence,
                    groundedness_score=cached.groundedness_score,
                    stage_times=_stage_times,
                )

        trace = None
        if self.telemetry:
            trace_kwargs: dict[str, Any] = {
                "name": "rag-query-pipeline",
                "input": _scrub_pii(question),
                "as_type": "trace",
                "tags": ["app:data-engineering-copilot"],
                "metadata": {
                    "app_env": settings.langfuse_environment,
                    "git_sha": settings.image_git_sha,
                },
            }
            if user_id:
                trace_kwargs["user_id"] = user_id
            if session_id:
                trace_kwargs["session_id"] = session_id
            if settings.langfuse_environment:
                trace_kwargs["environment"] = settings.langfuse_environment
            trace = self.telemetry.start_observation(**trace_kwargs)

        # Phase 2A: Query rewriting — collect all queries for multi-step retrieval
        rewritten = None
        all_queries: list[str] = [question]  # Always include original

        rewrite_span = None
        if trace:
            rewrite_span = trace.start_observation(name="query-rewriting", as_type="span")

        if self.query_rewriter is not None:
            if on_step:
                on_step("Rewriting query")
            rewritten = await self.query_rewriter.async_rewrite(question)
            if rewritten.decomposed_steps:
                all_queries.extend(rewritten.decomposed_steps)
                effective_query = rewritten.decomposed_steps[0]
            # Multi-query expansion: generate additional variations
            expanded = await self.query_rewriter.expand_queries(
                question, max_variations=self.config.max_expansion_queries
            )
            for q in expanded:
                if q not in all_queries:
                    all_queries.append(q)
            _record_stage("rewrite")
            logger.info(
                "query_rewritten intent=%s steps=%d expanded=%d hyde=%s original=%r",
                rewritten.intent,
                len(rewritten.decomposed_steps),
                len(expanded),
                bool(rewritten.hyde_query),
                question[:80],
            )

            if rewrite_span:
                rewrite_span.update(
                    input=question,
                    output={
                        "intent": rewritten.intent,
                        "decomposed_steps": len(rewritten.decomposed_steps),
                        "expanded_queries": len(expanded),
                        "hyde_query": bool(rewritten.hyde_query),
                    },
                )
                rewrite_span.end()

        _emit_detail(
            "rewrite",
            {
                "original_query": question,
                "rewritten_query": effective_query,
                "intent": rewritten.intent if rewritten is not None else None,
                "hyde_query": rewritten.hyde_query if rewritten is not None else None,
                "hyde_reason": rewritten.hyde_reason if rewritten is not None else None,
                "decomposed_steps": list(rewritten.decomposed_steps) if rewritten is not None else [],
                "expansions": list(all_queries),
            },
        )

        if on_step:
            on_step("Embedding query")
        logger.info("async_rag.answer question=%r queries=%d", question[:100], len(all_queries))

        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")

        # Determine chunk type filter based on intent. When a hard ``modules``
        # filter is present the exact API page is already pinned, so the "api"
        # chunk-type restriction is dropped — rendered reference pages carry
        # ``chunk_type="text"`` and combining both would empty the result set.
        chunk_type_filter = None
        if rewritten is not None and rewritten.intent == "api_lookup":
            metadata_filters = rewritten.filters if rewritten is not None else None
            if metadata_filters is None or not metadata_filters.modules:
                chunk_type_filter = "api"

        # Structured metadata filters extracted during rewriting.
        metadata_filters = rewritten.filters if rewritten is not None else None

        # Retrieve with all query variations and merge results
        all_retrieved: list = []
        seen_ids: set[str] = set()
        any_success = False
        last_error: Exception | None = None
        try:
            # Collect per-query result sets for rank fusion; the original query
            # is always first so it receives the fusion bonus.
            per_query_results: list[list[RetrievedChunk]] = []
            queries_to_run = list(all_queries)
            if rewritten is not None and rewritten.hyde_query:
                queries_to_run.append(rewritten.hyde_query)

            # Human-readable labels for provenance diagnostics (original query,
            # decomposed steps, HyDE variant, or rewrite expansions).
            decomposed_count = len(rewritten.decomposed_steps) if rewritten is not None else 0
            variant_labels: list[str] = []
            for i in range(len(queries_to_run)):
                if i == 0:
                    variant_labels.append("original")
                elif i <= decomposed_count:
                    variant_labels.append("decomposed")
                elif rewritten is not None and rewritten.hyde_query and i == len(queries_to_run) - 1:
                    variant_labels.append("hyde")
                else:
                    variant_labels.append("expanded")

            (
                per_query_results,
                _prov_variants,
                all_retrieved,
                seen_ids,
                any_success,
                last_error,
                query_dim,
                q_emb,
            ) = await self._retrieve_variant_queries(
                queries_to_run,
                variant_labels,
                source_filter=source_filter,
                chunk_type_filter=chunk_type_filter,
                metadata_filters=metadata_filters,
            )

            if not any_success and last_error is not None:
                raise RetrievalError(f"Vector store query failed: {last_error}") from last_error

            # Rank-fusion merge with original-query bonus.
            if per_query_results:
                retrieved_chunks = merge_retrieval_results(per_query_results, question)
            else:
                retrieved_chunks = sorted(all_retrieved, key=lambda c: c.confidence, reverse=True)
            _prov_pool = len(retrieved_chunks)
            _prov_fused = [_chunk_provenance_ref(c, rank) for rank, c in enumerate(retrieved_chunks)]

            # Parent-doc re-assembly: restore sibling segments so cross-mode
            # questions see both the YARN and Kubernetes paragraphs of a split
            # parent instead of a single matched segment.
            retrieved_chunks = await self._rejoin_sibling_chunks(retrieved_chunks)

            _emit_detail(
                "embed",
                {
                    "variants": len(queries_to_run),
                    "dimension": query_dim,
                    "l2_norm": round(_l2_norm(q_emb), 4) if query_dim and q_emb else None,
                },
            )
            _emit_detail(
                "retrieve",
                {
                    "pool_size": len(retrieved_chunks),
                    "candidates": list(_retrieval_details(retrieved_chunks)),
                    "rrf_profiles": sorted({str(v["rrf_profile"]) for v in _prov_variants}),
                },
            )

            if retrieval_span:
                retrieval_span.update(
                    output=[c.chunk.text for c in retrieved_chunks],
                    input=effective_query,
                )
                retrieval_span.end()
            _record_stage("retrieval")
            if on_step:
                on_step("Retrieving results")
            logger.info("Multi-step retrieval: %d queries → %d unique chunks", len(all_queries), len(retrieved_chunks))

            # Track retrieval scores for observability
            if self.retrieval_tracker is not None and retrieved_chunks:
                scores = [c.confidence for c in retrieved_chunks]
                self.retrieval_tracker.record_retrieval(scores, query=question)
        except RetrievalError:
            if trace:
                trace.update(output="RetrievalError")
                trace.end()
            raise
        except Exception as exc:
            logger.exception("Failed during retrieval: %s", exc)
            if retrieval_span:
                retrieval_span.update(output=str(exc), level="ERROR")
                retrieval_span.end()
            if trace:
                trace.update(output=str(exc))
                trace.end()
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        if not retrieved_chunks:
            if trace:
                trace.update(output="No chunks retrieved")
                trace.end()
            _emit_provenance()
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        # Indirect prompt injection guard: drop retrieved chunks that look like
        # embedded instructions before they reach the prompt.
        if self.input_guardrails is not None:
            scan_result = self.input_guardrails.scan_chunks(retrieved_chunks)
            retrieved_chunks = scan_result.kept
            if not retrieved_chunks:
                logger.warning("All retrieved chunks rejected by input guardrails")
                if trace:
                    trace.update(output="All chunks rejected by input guardrails")
                    trace.end()
                _emit_provenance()
                return Answer(
                    text="I cannot answer this question because it is outside my knowledge repository.",
                    sources=tuple(),
                    confidence=0.0,
                )

        generation_span: Any = None
        try:
            if on_step:
                on_step("Reranking results")

            rerank_span = None
            if trace:
                rerank_span = trace.start_observation(name="reranking", as_type="span")

            pre_rerank_count = len(retrieved_chunks)
            reranker = self.reranker
            rerank_used = False
            selective_skip = (
                self.config.reranker_selective_threshold < 1.0
                and retrieved_chunks
                and retrieved_chunks[0].confidence >= self.config.reranker_selective_threshold
            )
            if not selective_skip and self.config.reranker_enabled and reranker is not None and pre_rerank_count > 1:
                # The reranker model is loaded lazily (off the event loop) so a
                # cold model cache degrades to "no reranking" instead of either
                # failing the answer or silently skipping reranking forever.
                await self._ensure_reranker_ready()
                if reranker.is_available():
                    rerank_used = True
                    # Rerank a broad candidate pool against the original question.
                    # Multi-query retrieval and dense+sparse fusion generate recall;
                    # the cross-encoder is the single generic relevance decision.
                    rerank_pool = min(
                        pre_rerank_count,
                        _rerank_pool_size(
                            self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
                        ),
                    )
                    # Combined-corpus safeguard: multi-query fusion can union many
                    # hundreds of chunks across sources (query variants each return
                    # up to ``fused_limit``). Feed the cross-encoder only the top
                    # fused pool — ``merge_retrieval_results`` already sorts by
                    # fused score descending — so inference cost stays bounded and
                    # the pool handed to the reranker is focused, not dominated by
                    # low-rank union tail. The pool is still 8x the final top_k,
                    # preserving the reranker's recall-rescue role.
                    if pre_rerank_count > rerank_pool:
                        retrieved_chunks = retrieved_chunks[:rerank_pool]
                        pre_rerank_count = len(retrieved_chunks)
                    # Rerank against the original question, not the rewritten
                    # query (restores ``spark_ingestion_OPERATIONAL_ROLLOUT.md``
                    # fix 7): the rewrite can drift user-typed API terms (e.g.
                    # ``dense_rank`` → "dense ranking"), and the cross-encoder
                    # scores code/API pairs far higher against the verbatim
                    # question. The rewrite still drives the retrieval variants
                    # and the generation prompt.
                    rerank_query = question
                    retrieved_chunks = await reranker.rerank(rerank_query, retrieved_chunks, top_k=rerank_pool)

            _prov_rerank = {
                "enabled": rerank_used,
                "query": question,
                "pool_size": pre_rerank_count,
                "top_k": (
                    min(
                        pre_rerank_count,
                        _rerank_pool_size(
                            self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
                        ),
                    )
                    if rerank_used
                    else None
                ),
                "final_top_k": self.config.reranker_top_k,
            }

            # The reranker already returns the final relevance ordering. Do not
            # apply a second diversity objective here: lexical MMR can discard
            # complementary evidence needed to answer multi-part questions.
            if len(retrieved_chunks) > self.config.reranker_top_k:
                retrieved_chunks = retrieved_chunks[: self.config.reranker_top_k]

            # Phase 2D: Context compression (dedup + relevance re-ranking) — runs after reranking
            # to avoid wasted work (compressed results were previously discarded by re-fetch)
            if self.context_compressor is not None:
                retrieved_chunks = self.context_compressor.compress(retrieved_chunks, effective_query)
                logger.info("context_compressed chunks=%d", len(retrieved_chunks))
            _record_stage("rerank")

            if rerank_span:
                rerank_span.update(
                    input=f"{len(retrieved_chunks)} chunks before reranking",
                    output=f"{len(retrieved_chunks)} chunks after reranking",
                )
                rerank_span.end()

            rejected = await self._reject_low_confidence(
                retrieved_chunks,
                rerank_used=rerank_used,
                trace=trace,
                question=question,
                emit_provenance=_emit_provenance,
            )
            if rejected is not None:
                return rejected

            assembler = ContextAssembler(
                max_context_chars=self.config.max_context_chars,
                max_chunks_per_source=self.config.max_chunks_per_source,
            )
            context_str, source_names, dropped_records = assembler.assemble(
                retrieved_chunks,
                deduplicate=self.context_compressor is None,
            )
            _prov_dropped = dropped_records
            _emit_detail(
                "rerank",
                {
                    "enabled": rerank_used,
                    "pool_size": pre_rerank_count,
                    "top_k": (
                        min(
                            pre_rerank_count,
                            _rerank_pool_size(
                                self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
                            ),
                        )
                        if rerank_used
                        else None
                    ),
                    "final_top_k": self.config.reranker_top_k,
                    "compressed_dropped": len(dropped_records),
                },
            )
            # The final context reflects only the segments actually placed in
            # the prompt; budget-dropped segments must not be claimed as
            # retrieved into the final context.
            _prov_dropped_ids = {record["chunk_id"] for record in dropped_records}
            _final_chunks = [c for c in retrieved_chunks if c.chunk.chunk_id not in _prov_dropped_ids]
            _prov_final = [_chunk_provenance_ref(c, rank) for rank, c in enumerate(_final_chunks)]

            if retrieval_only:
                # Retrieval-only mode short-circuits before generation: the caller
                # (e.g. ``dec evaluate --spark``) needs the assembled context and
                # final chunk sources to score retrieval recall, without paying for
                # answer generation, groundedness verification or scope checks.
                _record_stage("total")
                _emit_provenance()
                return self._build_retrieval_only_answer(
                    _final_chunks=_final_chunks,
                    retrieved_chunks=retrieved_chunks,
                    effective_query=effective_query,
                    all_queries=all_queries,
                    rewritten=rewritten,
                    trace=trace,
                    context_str=context_str,
                    _stage_times=_stage_times,
                    _prov_rerank=_prov_rerank,
                    dropped_records=dropped_records,
                )

            if on_step:
                on_step("Generating answer")
            intent = rewritten.intent if rewritten else "factual"
            safe_question = PromptBuilder.sanitize_query(question)
            if self._pii_redactor is not None:
                safe_question, _pii_types = self._pii_redactor.redact(safe_question)
                if _pii_types:
                    logger.info("pii_redacted types=%s", _pii_types)
            prompt = self._prompt_builder.build_rag_prompt(context=context_str, question=safe_question, intent=intent)

            llm_client = self._select_llm_client(intent)
            _emit_detail(
                "generate",
                {
                    "context_chunks": len(_final_chunks),
                    "context_chars": len(context_str),
                    "prompt_chars": len(prompt),
                    "intent": intent,
                    "model": _llm_model_name(llm_client),
                },
            )
            generation_span = None
            if trace:
                generation_span = trace.start_observation(
                    name="llm-generation",
                    as_type="generation",
                    model=_llm_model_name(llm_client),
                )
                generation_span.update(input=prompt)
            answer_text = await llm_client.generate(prompt)

            # JSON retry: if the intent expects JSON output but parsing fails,
            # retry once with a stricter instruction.
            if intent not in CODE_INTENTS and isinstance(answer_text, str):
                parsed_attempt = parse_rag_response(answer_text)
                if not parsed_attempt.answer and len(answer_text.strip()) > 20:
                    logger.info("json_parse_retry intent=%s response_len=%d", intent, len(answer_text))
                    retry_prompt = prompt + get_langfuse_prompt("rag-json-retry-suffix").compile()
                    retry_span = None
                    if trace:
                        retry_span = trace.start_observation(
                            name="llm-json-retry",
                            as_type="generation",
                            model=_llm_model_name(llm_client),
                            input=retry_prompt,
                        )
                    answer_text = await llm_client.generate(retry_prompt)
                    if retry_span:
                        retry_span.update(output=answer_text)
                        retry_span.end()

            _record_stage("generation")

            # Post-generation syntax check for code intents
            answer_text = await self._validate_and_fix_code_syntax(answer_text, intent, llm_client, trace)

            # Track token usage from LLM provider
            if self.token_tracker is not None and hasattr(llm_client, "last_usage"):
                usage = getattr(llm_client, "last_usage", None)
                if usage is not None:
                    self.token_tracker.record(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        model=usage.model,
                    )

                    # Also send token usage and cost to Langfuse
                    if self.telemetry and trace:
                        try:
                            trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None)
                            if trace_id:
                                # Calculate cost based on provider pricing
                                cost = self._estimate_cost(
                                    usage.prompt_tokens,
                                    usage.completion_tokens,
                                    usage.model,
                                )

                                # Record token usage + cost on the generation observation
                                if generation_span:
                                    generation_span.update(
                                        usage_details={
                                            "input": usage.prompt_tokens,
                                            "output": usage.completion_tokens,
                                            "total": usage.prompt_tokens + usage.completion_tokens,
                                            "unit": "TOKENS",
                                        },
                                        cost_details={
                                            "input": cost,
                                            "output": cost,
                                            "total": cost,
                                            "currency": "USD",
                                        },
                                    )

                                # Update trace with token usage and cost metadata
                                trace.update(
                                    metadata={
                                        "token_usage": {
                                            "prompt_tokens": usage.prompt_tokens,
                                            "completion_tokens": usage.completion_tokens,
                                            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                                            "model": usage.model,
                                        },
                                        "cost_usd": cost,
                                    }
                                )

                                # Score cost for monitoring
                                self.telemetry.score(
                                    trace_id=trace_id,
                                    name="cost_usd",
                                    value=cost,
                                    data_type="NUMERIC",
                                )

                                logger.debug(
                                    "Langfuse token usage and cost recorded: tokens=%d cost=$%.6f",
                                    usage.prompt_tokens + usage.completion_tokens,
                                    cost,
                                )
                        except Exception as exc:
                            logger.warning("Failed to record Langfuse token usage: %s", exc)

            # Output guardrails: verify structure and quality
            from data_engineering_copilot.services.output_guardrails import OutputGuardrails

            raw_answer_text = answer_text  # preserved for empty-fallback below
            validated = OutputGuardrails.verify(answer_text, len(retrieved_chunks))
            if validated is not None:
                if validated.status == "INSUFFICIENT_CONTEXT" and validated.missing_info:
                    answer_text = f"{validated.answer}\n\nMissing information: {validated.missing_info}"
                else:
                    answer_text = validated.answer
                logger.info(
                    "output_guardrails passed status=%s confidence=%.2f citations=%d",
                    validated.status,
                    validated.confidence,
                    len(validated.citations),
                )
            else:
                logger.info("output_guardrails rejected answer, using raw output")

            # Guardrails may blank the answer (e.g. INSUFFICIENT_CONTEXT with an
            # empty answer and missing_info). Prefer the pre-guardrail raw LLM
            # output over rendering nothing; only fall back to a message when
            # both are empty so the UI never shows an empty "answer".
            if not answer_text or not str(answer_text).strip():
                if raw_answer_text and str(raw_answer_text).strip():
                    logger.warning("output_guardrails blanked answer, falling back to raw output")
                    # Surface the human-readable raw text if available; a raw
                    # structured-JSON blob with an empty answer still resolves
                    # to the default message rather than leaking raw JSON.
                    answer_text = parse_rag_response(str(raw_answer_text)).answer
                if not answer_text or not str(answer_text).strip():
                    logger.warning("LLM returned an empty answer, substituting default message")
                    answer_text = (
                        "No answer could be generated for this question. "
                        "The knowledge base may not contain enough information, or the "
                        "LLM returned an empty response. Try rephrasing the question."
                    )

            # Post-LLM PII redaction: strip PII from answer before returning
            if self._pii_redactor is not None:
                answer_text, pii_types_answer = self._pii_redactor.redact(answer_text)
                if pii_types_answer:
                    logger.info("pii_redacted_in_answer types=%s", pii_types_answer)

            # Citation verification: keep only citations matching retrieved sources
            if not isinstance(answer_text, str):
                answer_text = str(answer_text) if answer_text else ""
            parsed = parse_rag_response(answer_text)
            source_names = [c.chunk.source_name for c in retrieved_chunks]
            verified_citations = verify_citations(parsed.citations, source_names)
            if verified_citations:
                answer_text = parsed.answer
                logger.info("citations_verified=%d total=%d", len(verified_citations), len(parsed.citations))

            if generation_span:
                generation_span.update(output=answer_text)
                generation_span.end()
            if trace:
                trace.update(
                    output=answer_text,
                    metadata={
                        "stage_times_ms": _stage_times,
                        "num_sources": len(retrieved_chunks),
                        "intent": intent,
                    },
                )
                trace.end()

            if self.telemetry:
                try:
                    await self.telemetry.flush_async()
                except Exception:
                    logger.warning("Telemetry flush failed, ignoring", exc_info=True)

            _record_stage("total")
            _emit_provenance()
            trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None) if trace else None
            result = Answer(
                text=answer_text,
                sources=tuple(c.chunk for c in _final_chunks),
                confidence=retrieved_chunks[0].confidence,
                stage_times=_stage_times,
                trace_id=trace_id,
                rewritten_query=effective_query,
                query_variants=tuple(all_queries),
                intent=intent,
                retrieval_details=_retrieval_details(retrieved_chunks),
                rerank_details={**(_prov_rerank or {}), "compressed_dropped": len(dropped_records)},
                context=context_str,
                prompt=prompt,
                token_usage=_usage_to_dict(getattr(llm_client, "last_usage", None)),
            )

            # Phase 2B: Groundedness verification (annotate-only, fail-open)
            groundedness_score = 1.0

            groundedness_span = None
            if trace:
                groundedness_span = trace.start_observation(name="groundedness-verification", as_type="span")

            if self.groundedness_verifier is not None:
                (
                    supported,
                    unsupported_claims,
                    groundedness_score,
                ) = await self.groundedness_verifier.async_verify_with_score(result, retrieved_chunks)
                logger.info(
                    "groundedness_supported=%s unsupported=%d score=%.2f",
                    supported,
                    len(unsupported_claims),
                    groundedness_score,
                )
                if not supported and unsupported_claims:
                    result = replace(
                        result,
                        text=result.text + "\n\n[Note: Some claims may not be fully supported by the documentation.]",
                        groundedness_score=groundedness_score,
                        groundedness_claims=tuple(unsupported_claims),
                    )
                else:
                    result = replace(
                        result,
                        groundedness_score=groundedness_score,
                        groundedness_claims=tuple(unsupported_claims),
                    )

                if groundedness_span:
                    groundedness_span.update(
                        input="Answer and retrieved chunks",
                        output={
                            "supported": supported,
                            "unsupported_claims_count": len(unsupported_claims),
                            "groundedness_score": groundedness_score,
                        },
                    )
                    groundedness_span.end()

            # Phase 2C: Topic-scope gate (fail-open) — refuse the answer when the
            # retrieved context does not cover the question's topic.
            result = await self._apply_scope_gate(result, safe_question, context_str, trace)

            # Phase 2E: Langfuse scoring (confidence, groundedness, quality)
            if self.telemetry and trace:
                try:
                    # Get trace ID from the trace object
                    trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None)
                    if trace_id:
                        # Score confidence
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="confidence",
                            value=result.confidence,
                            data_type="NUMERIC",
                        )
                        # Score groundedness
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="groundedness",
                            value=groundedness_score,
                            data_type="NUMERIC",
                        )

                        # LLM-as-Judge: compute relevance score
                        # Relevance = confidence * groundedness (composite metric)
                        relevance_score = result.confidence * groundedness_score
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="relevance",
                            value=relevance_score,
                            data_type="NUMERIC",
                        )

                        # Completeness heuristic: based on answer length and source count
                        # Longer answers with more sources tend to be more complete
                        source_count = len(result.sources)
                        answer_length = len(result.text.split())
                        completeness = min(1.0, (answer_length / 100) * (1 + source_count * 0.1))
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="completeness",
                            value=completeness,
                            data_type="NUMERIC",
                        )

                        # Phase 7: richer score types — boolean + categorical.
                        # A full pipeline trace is never a cache hit (cache hits
                        # return early via the cache-hit trace), so this is always
                        # False here.
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="cache_hit",
                            value=False,
                            data_type="BOOLEAN",
                        )
                        # Categorical intent score, config-bound so the UI renders
                        # the category labels. Falls back to a bare numeric value
                        # when the score config cannot be resolved.
                        intent_config_id = _get_intent_config_id()
                        if intent_config_id:
                            self.telemetry.score(
                                trace_id=trace_id,
                                name="intent",
                                value=intent,
                                data_type="CATEGORICAL",
                                config_id=intent_config_id,
                            )
                        else:
                            intent_value = {
                                "factual": 0.0,
                                "code_example": 1.0,
                                "api_lookup": 2.0,
                                "comparative": 3.0,
                                "debugging": 4.0,
                                "how_to": 5.0,
                            }.get(intent, 0.0)
                            self.telemetry.score(
                                trace_id=trace_id,
                                name="intent_label",
                                value=intent_value,
                                data_type="CATEGORICAL",
                            )

                        # Log quality metrics for monitoring
                        logger.info(
                            "quality_scores confidence=%.3f groundedness=%.3f relevance=%.3f completeness=%.3f",
                            result.confidence,
                            groundedness_score,
                            relevance_score,
                            completeness,
                        )
                        logger.debug("Langfuse scores recorded for trace %s", trace_id)
                except Exception as exc:
                    logger.warning("Failed to record Langfuse scores: %s", exc)

            # Phase 2C: Cache the result (exact + semantic tiers). Only cache
            # answers that pass the quality gate; the QueryCache also enforces
            # this at the set level (defense-in-depth).
            if cache_active and self.cache is not None:
                envelope = CachedAnswer(
                    text=result.text,
                    sources=result.sources,
                    confidence=result.confidence,
                    groundedness_score=result.groundedness_score,
                    cached_at=time.time(),
                )
                if self.cache.is_cacheable(envelope):
                    await self.cache.aset_exact(question, envelope, scope=cache_scope)
                    if query_emb_for_cache is not None:
                        await self.cache.aset_semantic(question, query_emb_for_cache, envelope, scope=cache_scope)

            # Log cache hit rate for observability
            if self.cache is not None:
                logger.info(
                    "query_cache_stats hits=%d misses=%d hit_rate=%.2f",
                    self.cache._hits,
                    self.cache._misses,
                    self.cache.hit_rate,
                )

            return result
        except LLMGenerationError:
            if trace:
                trace.update(output="LLMGenerationError")
                trace.end()
            if self.telemetry:
                with contextlib.suppress(Exception):
                    await self.telemetry.flush_async()
            raise
        except Exception as exc:
            logger.exception("Failed during answer generation: %s", exc)
            if generation_span:
                generation_span.update(output=str(exc), level="ERROR")
                generation_span.end()
            if trace:
                trace.update(output=str(exc))
                trace.end()
            if self.telemetry:
                with contextlib.suppress(Exception):
                    await self.telemetry.flush_async()
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc

    async def answer_stream(
        self,
        question: str,
        source_filter: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream answer tokens via SSE while performing retrieval.

        Yields JSON-encoded SSE events:
        - ``{"type": "status", "message": "..."}`` for pipeline progress
        - ``{"type": "token", "content": "..."}`` for answer tokens
        - ``{"type": "done", "text": "...", "confidence": 0.0}`` at completion
        """
        _t0 = time.monotonic()
        yield _sse({"type": "status", "message": "Sanitizing query"})

        # Create trace for streaming path
        trace = None
        if self.telemetry:
            trace_kwargs: dict[str, Any] = {
                "name": "rag-query-pipeline-stream",
                "input": _scrub_pii(question),
                "as_type": "trace",
                "tags": ["app:data-engineering-copilot"],
                "metadata": {
                    "app_env": settings.langfuse_environment,
                    "git_sha": settings.image_git_sha,
                },
            }
            if user_id:
                trace_kwargs["user_id"] = user_id
            if session_id:
                trace_kwargs["session_id"] = session_id
            if settings.langfuse_environment:
                trace_kwargs["environment"] = settings.langfuse_environment
            trace = self.telemetry.start_observation(**trace_kwargs)

        # PII redaction
        safe_question = PromptBuilder.sanitize_query(question)
        if self._pii_redactor is not None:
            safe_question, _pii_types = self._pii_redactor.redact(safe_question)
            if _pii_types:
                yield _sse({"type": "status", "message": f"PII redacted: {', '.join(_pii_types)}"})

        # Query rewriting
        rewritten = None
        rewrite_span = None
        if trace:
            rewrite_span = trace.start_observation(name="query-rewriting", as_type="span")
        if self.query_rewriter is not None:
            yield _sse({"type": "status", "message": "Rewriting query"})
            try:
                rewritten = await self.query_rewriter.async_rewrite(safe_question)
            except Exception:
                logger.warning("Query rewrite failed during streaming, using raw query", exc_info=True)
                rewritten = None
            if rewritten is not None:
                yield _sse({"type": "status", "message": f"Intent: {rewritten.intent}"})
        if rewrite_span:
            rewrite_span.update(
                input=question,
                output={"intent": rewritten.intent if rewritten else "factual"},
            )
            rewrite_span.end()

        if rewritten is not None and rewritten.decomposed_steps:
            effective_query = rewritten.decomposed_steps[0]
        else:
            effective_query = safe_question
        intent = rewritten.intent if rewritten else "factual"

        # Retrieval
        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")
        yield _sse({"type": "status", "message": "Retrieving documents"})
        try:
            query_emb = await self.embedder.embed_query(effective_query)
        except Exception as exc:
            if trace:
                trace.update(output=f"EmbeddingError: {exc}")
                trace.end()
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        top_k = self.config.retrieval_top_k
        retrieved_chunks = await self.vector_store.query(
            query_embedding=query_emb,
            top_k=top_k,
            query_text=effective_query,
            source_filter=source_filter,
            fused_limit=_rerank_pool_size(
                self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
            ),
        )
        # Parent-doc re-assembly: restore sibling segments so cross-mode
        # questions see both the YARN and Kubernetes paragraphs.
        retrieved_chunks = await self._rejoin_sibling_chunks(retrieved_chunks)
        if retrieval_span:
            retrieval_span.update(
                input=effective_query,
                output=f"{len(retrieved_chunks)} chunks retrieved",
            )
            retrieval_span.end()
        yield _sse({"type": "status", "message": f"Retrieved {len(retrieved_chunks)} chunks"})

        if not retrieved_chunks:
            if trace:
                trace.update(output="No chunks retrieved")
                trace.end()
            yield _sse({"type": "done", "text": "No relevant documents found.", "confidence": 0.0})
            return

        # Reranking
        rerank_span = None
        if trace:
            rerank_span = trace.start_observation(name="reranking", as_type="span")
        if self.reranker is not None and len(retrieved_chunks) > 1 and await self._ensure_reranker_ready():
            yield _sse({"type": "status", "message": "Reranking"})
            retrieved_chunks = await self.reranker.rerank(
                query=effective_query,
                chunks=retrieved_chunks,
                top_k=self.config.reranker_top_k,
            )
        if rerank_span:
            rerank_span.update(
                input=f"{len(retrieved_chunks)} chunks before reranking",
                output=f"{len(retrieved_chunks)} chunks after reranking",
            )
            rerank_span.end()

        # Context assembly
        sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)
        assembler = ContextAssembler(
            max_context_chars=self.config.max_context_chars,
            max_chunks_per_source=self.config.max_chunks_per_source,
        )
        context_str, _source_names, _dropped_records = assembler.assemble(sorted_chunks)

        # Build prompt
        prompt = self._prompt_builder.build_rag_prompt(context=context_str, question=safe_question, intent=intent)
        yield _sse({"type": "status", "message": "Generating answer"})

        # Stream LLM tokens
        llm_client = self._select_llm_client(intent)
        generation_span = None
        if trace:
            generation_span = trace.start_observation(
                name="generation",
                as_type="generation",
                model=_llm_model_name(llm_client),
            )
        full_text = ""
        try:
            async for token in llm_client.generate_stream(prompt):
                full_text += token
                yield _sse({"type": "token", "content": token})
        except Exception:
            logger.exception("Streaming generation failed")
            if generation_span:
                generation_span.update(output="Generation failed", level="ERROR")
                generation_span.end()
            if trace:
                trace.update(output="Streaming generation failed")
                trace.end()
            yield _sse({"type": "error", "message": "Generation failed"})
            return
        if generation_span:
            generation_span.update(input=prompt, output=full_text)
            generation_span.end()

        # Post-generation validation: PII redaction, output guardrails, caching
        if self._pii_redactor is not None:
            full_text, _pii_types = self._pii_redactor.redact(full_text)
            if _pii_types:
                logger.info("pii_redacted_in_stream types=%s", _pii_types)

        from data_engineering_copilot.services.output_guardrails import OutputGuardrails

        validated = OutputGuardrails.verify(full_text, len(retrieved_chunks))
        if validated is not None:
            full_text = validated.answer

        confidence = retrieved_chunks[0].confidence if retrieved_chunks else 0.0

        # Score and end trace
        if trace:
            trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None)
            if trace_id and self.telemetry:
                with contextlib.suppress(Exception):
                    self.telemetry.score(trace_id=trace_id, name="confidence", value=confidence)
            trace.update(
                output=full_text,
                metadata={
                    "intent": intent,
                    "num_sources": len(retrieved_chunks),
                    "streaming": True,
                    "total_ms": round((time.monotonic() - _t0) * 1000, 1),
                },
            )
            trace.end()

        if self.cache is not None:
            try:
                envelope = CachedAnswer(
                    text=full_text,
                    sources=tuple(c.chunk for c in retrieved_chunks),
                    confidence=confidence,
                    cached_at=time.time(),
                )
                await self.cache.aset_exact(safe_question, envelope)
                query_emb_for_cache = await self.embedder.embed_query(effective_query)
                await self.cache.aset_semantic(safe_question, query_emb_for_cache, envelope)
            except Exception:
                logger.warning("Cache write failed in stream", exc_info=True)

        if self.telemetry:
            with contextlib.suppress(Exception):
                await self.telemetry.flush_async()

        # Done event with metadata
        if self.scope_verifier is not None and full_text.strip():
            scope_covered = await self.scope_verifier.verify(safe_question, context_str)
            if not scope_covered:
                logger.info("scope_gate_refused_stream topic_not_covered question=%r", safe_question[:80])
                full_text = _SCOPE_REFUSAL_TEXT
        groundedness_score, unsupported_claims = await self._verify_stream_groundedness(
            full_text, retrieved_chunks, trace
        )
        yield _sse(
            {
                "type": "done",
                "text": full_text,
                "confidence": confidence,
                "groundedness_score": groundedness_score,
                "groundedness_claims": list(unsupported_claims),
            }
        )

    async def chat_stream(
        self,
        question: str,
        source_filter: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        conversation_history: Sequence[ChatMessage] | None = None,
        max_history_tokens: int = 2048,
        cache_scope: CacheScope | None = None,
        chat_query_rewriter: QueryRewriter | None = None,
        chat_scope_verifier: ScopeVerifier | None = None,
        chat_llm_client: LLMClientProtocol | None = None,
        chat_reranker: RerankerProtocol | None = None,
        chat_blocked_url_substrings: Sequence[str] | None = None,
        chat_domain_sources: Sequence[str] | None = None,
        chat_system_role: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a conversational RAG answer via SSE events.

        Self-contained multi-turn pipeline (``answer``/``answer_stream`` are
        untouched): history-aware contextual rewriting, multi-query retrieval
        with expansion + HyDE + rank fusion, reranking, history-injected
        prompt, and streaming generation.

        Yields data-only SSE events (``{"type": ...}`` convention):
        - ``status`` — pipeline progress
        - ``sources`` — JSON-safe source references (before the first token)
        - ``token`` — answer deltas
        - ``done`` — final answer + confidence
        - ``error`` — terminal failure

        The shared two-tier answer cache is only read/written on **turn 1**
        (empty history — the question is standalone and cache-safe). Follow-up
        turns never touch the cache so context-dependent answers are never
        served stale.
        """
        _t0 = time.monotonic()
        yield _sse({"type": "status", "message": "Sanitizing query"})

        history = list(conversation_history or [])
        history_text = render_conversation_history(history)

        # Optional local-model routing: the ConversationService may inject a
        # local Ollama rewriter/scope-verifier so the cheap short steps never
        # hit the cloud. Defaults to the service's own (cloud) components.
        query_rewriter = chat_query_rewriter or self.query_rewriter
        scope_verifier = chat_scope_verifier or self.scope_verifier

        # P0-A: identity questions are answered directly by the fixed persona —
        # never from retrieved documents or cache (prevents the "I am Claude"
        # hijack).
        if _IDENTITY_QUESTION_RE.search(question):
            yield _sse({"type": "done", "text": _IDENTITY_ANSWER, "confidence": 1.0})
            return

        # Turn-1 cache: standalone question → safe to read/write the shared
        # two-tier cache (mirrors ``answer()``). Follow-ups bypass entirely.
        is_turn_one = not history
        query_emb_for_cache: list[float] | None = None
        if is_turn_one and self.cache is not None and self.config.cache_enabled:
            query_emb_for_cache = None
            cached = await self.cache.aget(question, scope=cache_scope)
            if cached is None and self.embedder is not None:
                with contextlib.suppress(Exception):
                    query_emb_for_cache = await self.embedder.embed_query(question)
                cached = await self.cache.aget(question, query_embedding=query_emb_for_cache, scope=cache_scope)
            if cached is not None:
                logger.info("chat_cache_hit turn_one question=%r", question[:80])
                await self._record_cache_hit_trace(
                    question, cached, cache_scope, semantic=query_emb_for_cache is not None
                )
                yield _sse(
                    {
                        "type": "done",
                        "text": _clean_chat_text(cached.text),
                        "confidence": cached.confidence,
                    }
                )
                # Cache hits surface follow-up suggestions too — either from the
                # cached envelope (instant, no regeneration) or generated from the
                # cached sources for older entries without cached suggestions.
                if self.config.chat_suggestions_enabled:
                    suggestions = list(cached.suggestions)
                    if not suggestions:
                        try:
                            cache_chunks = [
                                RetrievedChunk(chunk=src, distance=0.0, confidence=cached.confidence)
                                for src in cached.sources
                            ]
                            suggestions = await self._generate_suggestions(
                                question,
                                _clean_chat_text(cached.text),
                                cache_chunks,
                                chat_llm_client,
                                intent="factual",
                                history=history,
                            )
                        except Exception:
                            logger.warning("Follow-up suggestion generation failed (cache hit)", exc_info=True)
                            suggestions = []
                    if suggestions:
                        yield _sse({"type": "suggestions", "suggestions": suggestions})
                return

        # Phase F: smart-cache recall tier (follow-up turns only). Reuse similar
        # cached (question→answer) pairs via local synthesis, gated by scope
        # verify; on any failure fall through to the full pipeline.
        if (
            not is_turn_one
            and self.config.chat_cache_recall_enabled
            and self.cache is not None
            and self.embedder is not None
        ):
            recall = await self._smart_cache_recall(
                question=question,
                cache_scope=cache_scope,
                chat_llm_client=chat_llm_client,
                scope_verifier=scope_verifier,
            )
            if recall is not None:
                yield _sse({"type": "sources", "sources": recall["sources"]})
                yield _sse(
                    {"type": "done", "text": _clean_chat_text(recall["text"]), "confidence": recall["confidence"]}
                )
                return

        trace = None
        if self.telemetry:
            trace_kwargs: dict[str, Any] = {
                "name": "rag-chat-pipeline",
                "input": _scrub_pii(question),
                "as_type": "trace",
                "tags": ["app:data-engineering-copilot"],
                "metadata": {
                    "conversation": True,
                    "turn_index": len(history) // 2 + 1,
                    "app_env": settings.langfuse_environment,
                    "git_sha": settings.image_git_sha,
                },
            }
            if user_id:
                trace_kwargs["user_id"] = user_id
            if session_id:
                trace_kwargs["session_id"] = session_id
            if settings.langfuse_environment:
                trace_kwargs["environment"] = settings.langfuse_environment
            trace = self.telemetry.start_observation(**trace_kwargs)

        # PII redaction on the current message.
        safe_question = PromptBuilder.sanitize_query(question)
        if self._pii_redactor is not None:
            safe_question, _pii_types = self._pii_redactor.redact(safe_question)
            if _pii_types:
                yield _sse({"type": "status", "message": f"PII redacted: {', '.join(_pii_types)}"})

        # History-aware query rewriting (turn 1 skips the contextual prompt).
        rewritten = None
        intent = "factual"
        rewrite_span = None
        if trace:
            rewrite_span = trace.start_observation(name="query-rewriting", as_type="span")
        if query_rewriter is not None:
            yield _sse({"type": "status", "message": "Rewriting query"})
            try:
                # P2: anchor terse follow-ups to the session topic (first user turn).
                session_topic = ""
                for m in history:
                    if getattr(m, "role", "") == "user":
                        session_topic = (m.content or "")[:300]
                        break
                rewritten = await query_rewriter.async_rewrite(
                    safe_question, conversation_history=history, session_topic=session_topic or None
                )
            except Exception:
                logger.warning("Chat query rewrite failed, using raw query", exc_info=True)
                rewritten = None
            if rewritten is not None:
                intent = rewritten.intent
                yield _sse({"type": "status", "message": f"Intent: {intent}"})
        if rewrite_span:
            rewrite_span.update(
                input=question,
                output={"intent": intent},
            )
            rewrite_span.end()

        # Build the full query set: original + rewrite steps + expansions + HyDE.
        all_queries: list[str] = [safe_question]
        effective_query = safe_question
        if rewritten is not None:
            if rewritten.decomposed_steps:
                for step in rewritten.decomposed_steps:
                    if step not in all_queries:
                        all_queries.append(step)
                effective_query = rewritten.decomposed_steps[0]
            if query_rewriter is not None:
                expanded = await query_rewriter.expand_queries(
                    safe_question, max_variations=self.config.max_expansion_queries
                )
                for q in expanded:
                    if q not in all_queries:
                        all_queries.append(q)
        if rewritten is not None and rewritten.hyde_query and rewritten.hyde_query not in all_queries:
            all_queries.append(rewritten.hyde_query)

        # Multi-query retrieval + rank fusion.
        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")
        yield _sse({"type": "status", "message": "Retrieving documents"})
        chunk_type_filter = None
        metadata_filters = None
        if rewritten is not None and rewritten.intent == "api_lookup":
            metadata_filters = rewritten.filters
            if metadata_filters is None or not metadata_filters.modules:
                chunk_type_filter = "api"

        all_retrieved: list[RetrievedChunk] = []
        per_query_results: list[list[RetrievedChunk]] = []
        seen_ids: set[str] = set()
        any_success = False
        last_error: Exception | None = None
        try:
            for q in all_queries:
                try:
                    q_emb = await self.embedder.embed_query(q)
                    results = await self.vector_store.query(
                        q_emb,
                        top_k=self.config.retrieval_top_k,
                        query_text=q,
                        source_filter=source_filter,
                        chunk_type_filter=chunk_type_filter,
                        metadata_filters=metadata_filters,
                        fused_limit=_rerank_pool_size(
                            self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
                        ),
                    )
                    if not results and metadata_filters is not None and not metadata_filters.is_empty:
                        results = await self.vector_store.query(
                            q_emb,
                            top_k=self.config.retrieval_top_k,
                            query_text=q,
                            source_filter=source_filter,
                            chunk_type_filter=chunk_type_filter,
                            fused_limit=_rerank_pool_size(
                                self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
                            ),
                        )
                    any_success = True
                    per_query_results.append(results)
                    for r in results:
                        if r.chunk.chunk_id not in seen_ids:
                            seen_ids.add(r.chunk.chunk_id)
                            all_retrieved.append(r)
                except Exception as sub_exc:
                    last_error = sub_exc
                    logger.warning("Failed to retrieve for sub-query %r: %s", q[:50], sub_exc)

            if not any_success and last_error is not None:
                raise RetrievalError(f"Vector store query failed: {last_error}") from last_error

            if per_query_results:
                retrieved_chunks = merge_retrieval_results(per_query_results, safe_question)
            else:
                retrieved_chunks = sorted(all_retrieved, key=lambda c: c.confidence, reverse=True)

            # Parent-doc re-assembly: restore sibling segments so cross-mode
            # chat questions see both the YARN and Kubernetes paragraphs.
            retrieved_chunks = await self._rejoin_sibling_chunks(retrieved_chunks)
        except RetrievalError:
            if trace:
                trace.update(output="RetrievalError")
                trace.end()
            yield _sse({"type": "error", "message": "Retrieval failed"})
            return
        except Exception as exc:
            logger.exception("Failed during chat retrieval: %s", exc)
            if trace:
                trace.update(output=str(exc))
                trace.end()
            yield _sse({"type": "error", "message": "Retrieval failed"})
            return

        if retrieval_span:
            retrieval_span.update(
                input=effective_query,
                output=f"{len(retrieved_chunks)} chunks retrieved",
            )
            retrieval_span.end()
        yield _sse({"type": "status", "message": f"Retrieved {len(retrieved_chunks)} chunks"})

        if not retrieved_chunks:
            if trace:
                trace.update(output="No chunks retrieved")
                trace.end()
            yield _sse({"type": "done", "text": "No relevant documents found.", "confidence": 0.0})
            return

        # P0-A: identity-injection defense — drop chunks whose URL matches a
        # blocked substring (e.g. Claude's self-identifying system-prompts.md)
        # before they reach the prompt. P1: restrict to whitelisted sources.
        blocked = set(chat_blocked_url_substrings or ())
        allowed_sources = set(chat_domain_sources or ())
        if blocked or allowed_sources:
            kept: list[RetrievedChunk] = []
            for r in retrieved_chunks:
                if blocked and any(sub in r.chunk.url for sub in blocked):
                    logger.info(
                        "chat_blocked_chunk url=%r sub=%r",
                        r.chunk.url,
                        next((s for s in blocked if s in r.chunk.url), None),
                    )
                    continue
                if allowed_sources and r.chunk.source_name not in allowed_sources:
                    logger.info("chat_domain_filtered source=%r", r.chunk.source_name)
                    continue
                kept.append(r)
            retrieved_chunks = kept
            if not retrieved_chunks:
                logger.warning("All retrieved chunks filtered out for chat (blocked/domain)")
                if trace:
                    trace.update(output="All chunks filtered out (blocked/domain)")
                    trace.end()
                yield _sse({"type": "done", "text": "No relevant documents found.", "confidence": 0.0})
                return

        # P3: domain-coherence fail-safe. If the surviving context is dominated
        # by a foreign domain (e.g. Claude docs) for a data-engineering query,
        # refuse cleanly instead of generating an off-topic answer/example.
        if _domain_mismatch(retrieved_chunks):
            logger.warning("chat_domain_mismatch refusing topic_not_covered question=%r", safe_question[:80])
            if trace:
                trace.update(output="Domain mismatch: refusing")
                trace.end()
            yield _sse({"type": "done", "text": _clean_chat_text(_SCOPE_REFUSAL_TEXT), "confidence": 0.0})
            return

        # Indirect prompt injection guard on retrieved chunks.
        if self.input_guardrails is not None:
            scan_result = self.input_guardrails.scan_chunks(retrieved_chunks)
            retrieved_chunks = scan_result.kept
            if not retrieved_chunks:
                logger.warning("All retrieved chunks rejected by input guardrails (chat)")
                if trace:
                    trace.update(output="All chunks rejected by input guardrails")
                    trace.end()
                yield _sse({"type": "done", "text": "No relevant documents found.", "confidence": 0.0})
                return

        # Reranking. Chat may inject a local-only reranker (skip the ~5s cloud
        # LLM rerank chain); defaults to the service's own reranker.
        reranker = chat_reranker or self.reranker
        rerank_span = None
        if trace:
            rerank_span = trace.start_observation(name="reranking", as_type="span")
        pre_rerank_count = len(retrieved_chunks)
        if reranker is not None and self.config.reranker_enabled and pre_rerank_count > 1:
            yield _sse({"type": "status", "message": "Reranking"})
            await self._ensure_reranker_ready()
            if reranker.is_available():
                rerank_pool = min(
                    pre_rerank_count,
                    _rerank_pool_size(
                        self.config.retrieval_top_k, self.config.reranker_top_k, self.config.reranker_pool_size
                    ),
                )
                if pre_rerank_count > rerank_pool:
                    retrieved_chunks = retrieved_chunks[:rerank_pool]
                retrieved_chunks = await reranker.rerank(question, retrieved_chunks, top_k=rerank_pool)
            if len(retrieved_chunks) > self.config.reranker_top_k:
                retrieved_chunks = retrieved_chunks[: self.config.reranker_top_k]
        if rerank_span:
            rerank_span.update(
                input=f"{len(retrieved_chunks)} chunks before reranking",
                output=f"{len(retrieved_chunks)} chunks after reranking",
            )
            rerank_span.end()

        # P3: domain-coherence fail-safe. After reranking, the TOP chunks that
        # reach the prompt should not be dominated by a foreign domain (e.g.
        # Claude docs) for a data-engineering query. Evaluated post-rerank so a
        # wide fused candidate pool (which legitimately mixes domains) does not
        # falsely trigger the refusal.
        if _domain_mismatch(retrieved_chunks):
            logger.warning("chat_domain_mismatch refusing topic_not_covered question=%r", safe_question[:80])
            if trace:
                trace.update(output="Domain mismatch: refusing")
                trace.end()
            yield _sse({"type": "done", "text": _clean_chat_text(_SCOPE_REFUSAL_TEXT), "confidence": 0.0})
            return

        # Context assembly.
        sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)
        assembler = ContextAssembler(
            max_context_chars=self.config.max_context_chars,
            max_chunks_per_source=self.config.max_chunks_per_source,
        )
        context_str, _source_names, _dropped = assembler.assemble(sorted_chunks)

        source_refs = [
            {
                "source_name": c.chunk.source_name,
                "title": c.chunk.title,
                "url": c.chunk.url,
                "snippet": c.chunk.text[:240],
                "chunk_id": c.chunk.chunk_id,
            }
            for c in retrieved_chunks
        ]
        yield _sse({"type": "sources", "sources": source_refs})

        # Build prompt with history injected (compile-vars only).
        prompt = self._prompt_builder.build_rag_prompt(
            context=context_str,
            question=safe_question,
            intent=intent,
            history=history_text if history_text.strip() else None,
            max_history_tokens=max_history_tokens,
            system_role=chat_system_role,
        )
        yield _sse({"type": "status", "message": "Generating answer"})

        llm_client = chat_llm_client or self._select_llm_client(intent)
        generation_span = None
        if trace:
            generation_span = trace.start_observation(
                name="generation",
                as_type="generation",
                model=_llm_model_name(llm_client),
            )
        full_text = ""
        try:
            async for token in llm_client.generate_stream(prompt):
                full_text += token
                yield _sse({"type": "token", "content": token})
        except Exception:
            logger.exception("Chat streaming generation failed")
            if generation_span:
                generation_span.update(output="Generation failed", level="ERROR")
                generation_span.end()
            if trace:
                trace.update(output="Streaming generation failed")
                trace.end()
            yield _sse({"type": "error", "message": "Generation failed"})
            return
        if generation_span:
            generation_span.update(input=prompt, output=full_text)
            generation_span.end()

        # Post-generation: PII redaction + output guardrails.
        if self._pii_redactor is not None:
            full_text, _pii_types = self._pii_redactor.redact(full_text)
            if _pii_types:
                logger.info("pii_redacted_in_chat types=%s", _pii_types)

        from data_engineering_copilot.services.output_guardrails import OutputGuardrails

        validated = OutputGuardrails.verify(full_text, len(retrieved_chunks))
        if validated is not None:
            full_text = validated.answer

        confidence = retrieved_chunks[0].confidence if retrieved_chunks else 0.0

        # Scope gate (fail-open).
        if scope_verifier is not None and full_text.strip():
            scope_covered = await scope_verifier.verify(safe_question, context_str)
            if not scope_covered:
                logger.info("scope_gate_refused_chat topic_not_covered question=%r", safe_question[:80])
                full_text = _SCOPE_REFUSAL_TEXT

        # Annotate-only groundedness verification (never blocks or regenerates).
        groundedness_score, unsupported_claims = await self._verify_stream_groundedness(
            full_text, retrieved_chunks, trace
        )

        if trace:
            trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None)
            if trace_id and self.telemetry:
                with contextlib.suppress(Exception):
                    self.telemetry.score(trace_id=trace_id, name="confidence", value=confidence)
            trace.update(
                output=full_text,
                metadata={
                    "intent": intent,
                    "num_sources": len(retrieved_chunks),
                    "streaming": True,
                    "conversation": True,
                    "total_ms": round((time.monotonic() - _t0) * 1000, 1),
                },
            )
            trace.end()

        if self.telemetry:
            with contextlib.suppress(Exception):
                await self.telemetry.flush_async()

        # ChatGPT-style follow-up suggestions (emitted after done so the answer
        # shows first; chips pop in a moment later). Best-effort/fail-open.
        # Intent + history give the logical-expansion context (like ChatGPT/Gemini).
        suggestions: list[str] = []
        if self.config.chat_suggestions_enabled:
            try:
                suggestions = await self._generate_suggestions(
                    safe_question,
                    full_text,
                    retrieved_chunks,
                    chat_llm_client,
                    intent=intent,
                    history=history,
                )
            except Exception:
                logger.warning("Follow-up suggestion generation failed", exc_info=True)
                suggestions = []

        # Turn-1 cache write: standalone question → safe to cache. Include the
        # generated suggestions so a later cache hit returns answer + chips
        # without regeneration.
        if is_turn_one and self.cache is not None and self.config.cache_enabled and full_text.strip():
            try:
                envelope = CachedAnswer(
                    text=full_text,
                    sources=tuple(c.chunk for c in retrieved_chunks),
                    confidence=confidence,
                    cached_at=time.time(),
                    suggestions=tuple(suggestions),
                )
                await self.cache.aset_exact(question, envelope, scope=cache_scope)
                if query_emb_for_cache is not None:
                    await self.cache.aset_semantic(question, query_emb_for_cache, envelope, scope=cache_scope)
            except Exception:
                logger.warning("Chat turn-1 cache write failed", exc_info=True)

        yield _sse(
            {
                "type": "done",
                "text": _clean_chat_text(full_text),
                "confidence": confidence,
                "groundedness_score": groundedness_score,
                "groundedness_claims": list(unsupported_claims),
            }
        )
        if suggestions:
            yield _sse({"type": "suggestions", "suggestions": suggestions})

    async def _generate_suggestions(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        chat_llm_client: LLMClientProtocol | None,
        intent: str = "factual",
        history: Sequence[ChatMessage] | None = None,
    ) -> list[str]:
        """Generate ChatGPT-style follow-up suggestions for the just-answered turn.

        Uses intent + conversation history for logical-expansion steering (the
        "next step / alternative / edge case" taxonomy). Mode from
        ``self.config.chat_suggestions_mode``:
        - ``llm``: prompt an LLM to propose short follow-ups (deduped, filtered).
        - ``rule``: deterministic suggestions from retrieved source titles/topic.
        - ``hybrid``: try LLM, fall back to rule on error/empty/degenerate output.
        """
        if not self.config.chat_suggestions_enabled:
            return []

        rule_suggestions = self._rule_suggestions(question, retrieved_chunks, intent=intent)
        mode = self.config.chat_suggestions_mode
        if mode == "rule":
            return rule_suggestions

        llm_suggestions: list[str] = []
        if mode in ("llm", "hybrid"):
            llm_suggestions = await self._llm_suggestions(
                question, answer, retrieved_chunks, chat_llm_client, intent=intent, history=history
            )

        if mode == "llm" or llm_suggestions:
            return llm_suggestions
        return rule_suggestions

    async def _llm_suggestions(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        chat_llm_client: LLMClientProtocol | None,
        intent: str = "factual",
        history: Sequence[ChatMessage] | None = None,
    ) -> list[str]:
        """Ask an LLM for follow-up suggestions grounded in the retrieved context.

        The prompt includes the retrieved documentation chunks (the same ones
        that grounded the answer), the query intent, and the conversation
        history, and instructs the LLM to only propose follow-ups answerable
        from that context — so clicking a chip never leads to an
        ``INSUFFICIENT_CONTEXT`` turn.
        """
        client = chat_llm_client or self._select_llm_client("factual")
        max_suggestions = self.config.chat_suggestions_count
        context_text = self._suggestion_context(retrieved_chunks)
        history_text = render_conversation_history(list(history or [])[-4:])
        try:
            prompt = _SUGGESTION_PROMPT.format(
                count=max_suggestions,
                history=history_text or "(no prior turns)",
                context=context_text,
                question=question[:500],
                intent=intent or "factual",
                answer=(answer or "")[:1500],
            )
            raw = await client.generate(prompt)
        except Exception:
            logger.warning("Follow-up suggestion LLM call failed", exc_info=True)
            return []

        suggestions: list[str] = []
        seen: set[str] = set()
        for line in raw.splitlines():
            line = line.strip().lstrip("-*0123456789.) ")
            line = line.strip().strip('"').strip()
            if not line or is_degenerate_query(line) or line.lower() in seen:
                continue
            if len(line) > 200:
                continue
            seen.add(line.lower())
            suggestions.append(line)
            if len(suggestions) >= max_suggestions:
                break
        return suggestions

    def _build_retrieval_only_answer(
        self,
        *,
        _final_chunks: list[RetrievedChunk],
        retrieved_chunks: list[RetrievedChunk],
        effective_query: str | None,
        all_queries: Sequence[str],
        rewritten,
        trace,
        context_str: str,
        _stage_times: dict[str, float],
        _prov_rerank: dict[str, object] | None,
        dropped_records: list[dict[str, object]],
    ) -> Answer:
        """Build the Answer for ``retrieval_only`` mode (no generation).

        Returns sources and context for retrieval-stage scoring (used by
        ``dec evaluate --spark``) without paying for answer generation,
        groundedness verification or scope checks.
        """
        trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None) if trace else None
        return Answer(
            text="",
            sources=tuple(c.chunk for c in _final_chunks),
            confidence=retrieved_chunks[0].confidence if retrieved_chunks else 0.0,
            stage_times=_stage_times,
            trace_id=trace_id,
            rewritten_query=effective_query,
            query_variants=tuple(all_queries),
            intent=rewritten.intent if rewritten else "factual",
            retrieval_details=_retrieval_details(retrieved_chunks),
            rerank_details={**(_prov_rerank or {}), "compressed_dropped": len(dropped_records)},
            context=context_str,
        )

    @staticmethod
    def _suggestion_context(retrieved_chunks: list[RetrievedChunk], limit: int = 5) -> str:
        """Build a bounded context block from retrieved chunks for suggestion grounding."""
        parts: list[str] = []
        for chunk in retrieved_chunks[:limit]:
            title = (chunk.chunk.title or chunk.chunk.source_name or "doc").strip()
            snippet = (chunk.chunk.text or "").strip()[:400]
            if snippet:
                parts.append(f"[{title}] {snippet}")
        return "\n\n".join(parts) if parts else "(no retrieved context)"

    @staticmethod
    def _rule_suggestions(
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        intent: str = "factual",
    ) -> list[str]:
        """Deterministic follow-up suggestions derived from retrieved source titles.

        Intent-aware: shapes follow-ups toward the user's goal (examples for
        ``code_example``/``how_to``, alternatives for ``comparative``, related
        functions for ``api_lookup``, causes/fixes for ``debugging``).
        """
        titles: list[str] = []
        seen: set[str] = set()
        for chunk in retrieved_chunks:
            title = (chunk.chunk.title or "").strip()
            if title and title.lower() not in seen:
                seen.add(title.lower())
                titles.append(title)
        if not titles:
            return []
        primary = titles[0]
        second = titles[1] if len(titles) > 1 else primary
        third = titles[2] if len(titles) > 2 else second
        if intent in ("code_example", "api_lookup", "how_to"):
            suggestions = [
                f"Show me a code example for {primary}.",
                f"What are the parameters or options for {primary}?",
                f"How does {second} work?",
            ]
        elif intent == "comparative":
            suggestions = [
                f"What are the differences between {primary} and {second}?",
                f"When should I use {primary} over {second}?",
                f"Give me an example using {third}.",
            ]
        elif intent == "debugging":
            suggestions = [
                f"What are common causes of errors with {primary}?",
                f"How do I fix issues with {primary}?",
                f"What does the documentation say about {second}?",
            ]
        else:  # factual
            suggestions = [
                f"What does the documentation say about {primary}?",
                f"Can you explain {second} in more detail?",
                f"Give me a code example for {third}.",
            ]
        return suggestions[:3]

    async def _smart_cache_recall(
        self,
        question: str,
        cache_scope: CacheScope | None,
        chat_llm_client: LLMClientProtocol | None,
        scope_verifier,
    ) -> dict | None:
        """Phase F smart-cache recall: synthesize an answer from similar cached pairs.

        Returns ``{"text", "confidence", "sources"}`` when a verified answer can
        be produced from the cache, else ``None`` (caller falls through to the
        full pipeline). Stale or below-threshold pairs, generation errors, and a
        scope-verify rejection all produce ``None``.
        """
        try:
            query_emb = await self.embedder.embed_query(question)
        except Exception:
            logger.warning("Smart-cache recall embed failed; falling through", exc_info=True)
            return None

        cache = self.cache
        if cache is None:
            return None
        pairs = await cache.atop_k(
            query_emb,
            scope=cache_scope,
            k=self.config.chat_cache_top_k,
            min_similarity=self.config.chat_cache_recall_threshold,
        )
        now = time.time()
        fresh = [
            (score, envelope)
            for score, envelope in pairs
            if envelope.cached_at and (now - envelope.cached_at) <= self.config.chat_cache_max_age_seconds
        ]
        if not fresh:
            logger.info("smart_cache_recall no_fresh_pairs question=%r", question[:80])
            return None

        blocks: list[str] = []
        for _score, envelope in fresh:
            blocks.append(f"Prior grounded answer: {envelope.text}")
        context = "\n\n".join(blocks)

        if scope_verifier is not None and not await scope_verifier.verify(question, context):
            logger.info("smart_cache_recall scope_rejected question=%r", question[:80])
            return None

        prompt = (
            "You are a documentation assistant. A similar question was answered "
            "from the documentation before. Using ONLY the prior grounded answers "
            "below, answer the current question. If they do not cover it, say so.\n\n"
            + SYSTEM_BLOCK_SEPARATOR
            + f"## PRIOR GROUNDED ANSWERS\n{context}\n\n"
            f"## CURRENT QUESTION\n{question}\n\n"
            "Your answer:"
        )
        generator = chat_llm_client or self._select_llm_client("factual")
        try:
            answer = await generator.generate(prompt)
        except Exception:
            logger.warning("Smart-cache recall generation failed; falling through", exc_info=True)
            return None

        text = answer.strip()
        if not text or any(marker.search(text) for marker in _NON_ANSWER_MARKERS):
            return None

        source_refs: list[dict] = []
        seen_urls: set[str] = set()
        for _score, envelope in fresh:
            for src in envelope.sources:
                if src.url in seen_urls:
                    continue
                seen_urls.add(src.url)
                source_refs.append(
                    {
                        "source_name": src.source_name,
                        "title": src.title,
                        "url": src.url,
                        "snippet": src.text[:240],
                    }
                )
        return {"text": text, "confidence": fresh[0][1].confidence, "sources": source_refs}

    def _select_llm_client(self, intent: str) -> LLMClientProtocol:
        """Route code intents to the code-specific LLM if configured."""
        if self.code_llm_client and intent in CODE_INTENTS:
            return self.code_llm_client
        return self.llm_client

    async def _apply_scope_gate(
        self,
        result: Answer,
        question: str,
        context: str,
        trace: TelemetryTracerProtocol | None,
    ) -> Answer:
        """Topic-scope gate: refuse the answer when the retrieved context does
        not cover the question's topic. Fail-open — errors never block.
        """
        if self.scope_verifier is None:
            return result
        scope_span = None
        if trace:
            scope_span = trace.start_observation(name="scope-verification", as_type="span")
        scope_covered = await self.scope_verifier.verify(question, context)
        if not scope_covered:
            logger.info("scope_gate_refused topic_not_covered question=%r", question[:80])
            result = replace(result, text=_SCOPE_REFUSAL_TEXT)
        if scope_span:
            scope_span.update(
                input={"question": question, "context_chars": len(context)},
                output={"covered": scope_covered},
            )
            scope_span.end()
        return result

    async def _verify_stream_groundedness(
        self,
        text: str,
        retrieved_chunks: list[RetrievedChunk],
        trace: TelemetryTracerProtocol | None = None,
    ) -> tuple[float, tuple[str, ...]]:
        """Run annotate-only groundedness verification on a streamed answer.

        Returns ``(groundedness_score, unsupported_claims)``. Fail-open: on
        any error or when the verifier is absent, returns ``(1.0, ())`` so a
        verifier hiccup never blocks or stalls a streaming turn. Does NOT
        regenerate or refuse — matches the annotate-only contract of
        ``answer()`` (see ``groundedness.py``).
        """
        if self.groundedness_verifier is None or not text.strip():
            return 1.0, ()
        groundedness_span = None
        if trace:
            groundedness_span = trace.start_observation(name="groundedness-verification", as_type="span")
        groundedness_score = 1.0
        unsupported_claims: list[str] = []
        try:
            pseudo = Answer(text=text, sources=(), confidence=0.0)
            (
                _supported,
                unsupported_claims,
                groundedness_score,
            ) = await self.groundedness_verifier.async_verify_with_score(pseudo, retrieved_chunks)
        except Exception:
            logger.warning("Stream groundedness verification failed", exc_info=True)
            return 1.0, ()
        finally:
            if groundedness_span:
                groundedness_span.update(
                    input={"answer_chars": len(text), "context_chunks": len(retrieved_chunks)},
                    output={"groundedness_score": groundedness_score, "unsupported": len(unsupported_claims)},
                )
                groundedness_span.end()
        logger.info(
            "stream_groundedness unsupported=%d score=%.2f",
            len(unsupported_claims),
            groundedness_score,
        )
        return groundedness_score, tuple(unsupported_claims)

    async def _rejoin_sibling_chunks(
        self,
        retrieved_chunks: list[RetrievedChunk],
        max_sibling_blocks: int = 3,
    ) -> list[RetrievedChunk]:
        """Post-retrieval parent-doc re-assembly (sibling rejoin).

        When a retrieved chunk is a segment of a losslessly-split parent, its
        siblings share the same ``parent_content_hash``. Rejoining them restores
        the surrounding context so a cross-mode answer sees BOTH the YARN
        paragraph and its Kubernetes sibling — not just the single matched
        segment (which is the root cause of mode-confusion hallucination).

        Each distinct parent hash is collapsed into a single rejoined block that
        replaces its member segments in-place (bounded to ``max_sibling_blocks``
        parents). Fail-open: any store error or absent parent metadata leaves
        the retrieved set untouched.
        """
        parents: dict[str, list[RetrievedChunk]] = {}
        for r in retrieved_chunks:
            parent_hash = r.chunk.parent_content_hash
            if isinstance(parent_hash, str) and parent_hash:
                parents.setdefault(parent_hash, []).append(r)

        if not parents:
            return retrieved_chunks

        # Select the parents to rejoin (highest-confidence group first) so the
        # scroll cost stays bounded.
        ordered_parents = sorted(
            parents.items(),
            key=lambda item: max(r.confidence for r in item[1]),
            reverse=True,
        )
        selected_hashes = [ph for ph, _ in ordered_parents[:max_sibling_blocks]]
        selected_ids = {r.chunk.chunk_id for ph in selected_hashes for r in parents[ph]}

        rejoined: list[RetrievedChunk] = []
        for chunk in retrieved_chunks:
            parent_hash = chunk.chunk.parent_content_hash
            if isinstance(parent_hash, str) and parent_hash in selected_hashes:
                continue
            rejoined.append(chunk)

        scroll = getattr(self.vector_store, "scroll_chunks_by_parent_hash", None)
        if not callable(scroll):
            return retrieved_chunks

        async def _scroll(parent_hash: str, source_name: str) -> list[DocumentChunk]:
            result = scroll(parent_hash, source_name=source_name)
            if inspect.isawaitable(result):
                awaited = await cast(Awaitable[object], result)
                return list(awaited) if isinstance(awaited, (list, tuple)) else []
            return list(result) if isinstance(result, (list, tuple)) else []

        for parent_hash in selected_hashes:
            group = parents[parent_hash]
            try:
                siblings = await _scroll(parent_hash, source_name=group[0].chunk.source_name)
            except Exception:
                logger.warning("Sibling rejoin scroll failed for parent=%r", parent_hash, exc_info=True)
                rejoined.extend(group)
                continue
            if not siblings or len(siblings) <= 1:
                rejoined.extend(group)
                continue
            block_text = "\n".join(s.text for s in siblings)
            if len(block_text) > self._rejoin_item_limit_chars:
                block_text = _cap_rejoined_block(block_text, self._rejoin_item_limit_chars)
            best = max(group, key=lambda r: r.confidence)
            rejoined_chunk = replace(
                best.chunk,
                text=block_text,
                word_count=len(block_text.split()),
                character_count=len(block_text),
                segment_index=-1,
                segment_total=1,
            )
            rejoined.append(RetrievedChunk(chunk=rejoined_chunk, distance=best.distance, confidence=best.confidence))

        logger.info(
            "sibling_rejoin parents=%d rejoined=%d segments_merged=%d",
            len(selected_hashes),
            len([r for r in rejoined if r.chunk.chunk_id not in selected_ids]),
            len(selected_ids),
        )
        return rejoined

    async def _ensure_reranker_ready(self) -> bool:
        """Lazily load the cross-encoder model so reranking actually runs.

        Returns True when reranking can be used for this request. The model
        load runs off the event loop and is fail-open: on timeout, missing
        optional dependency, or any load error we degrade to "no reranking"
        rather than failing or stalling the answer. Non-async doubles (test
        mocks) are treated as already ready per ``is_available()``.
        """
        reranker = self.reranker
        if reranker is None or not self.config.reranker_enabled or reranker.is_available():
            return reranker is not None and reranker.is_available()
        initialize = reranker.initialize()
        if not hasattr(initialize, "__await__"):
            logger.debug("Reranker initialize() is not async; assuming ready=%s", reranker.is_available())
            return bool(reranker.is_available())
        try:
            await asyncio.wait_for(initialize, timeout=_RERANKER_INIT_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("Reranker initialization failed or timed out; proceeding without reranking")
        return bool(reranker.is_available())

    async def _validate_and_fix_code_syntax(
        self, answer_text: str, intent: str, llm_client: LLMClientProtocol, trace: Any | None = None
    ) -> str:
        """Validate Python code blocks in answer. Retry once if syntax fails.

        Only validates Python code blocks — other languages (Scala, SQL, etc.)
        are returned as-is since we can't syntax-check them with ast.parse.
        """
        if intent not in CODE_INTENTS:
            return answer_text

        code_blocks = re.findall(r"```python\n(.*?)```", answer_text, re.DOTALL)
        if not code_blocks:
            return answer_text

        invalid_blocks = []
        for block in code_blocks:
            try:
                ast.parse(block.strip())
            except SyntaxError:
                invalid_blocks.append(block)

        if not invalid_blocks:
            return answer_text

        fix_prompt = (
            "The following Python code has syntax errors. Fix ONLY the code, "
            "keeping the same structure and imports. Return valid Python only.\n\n"
            f"Broken code:\n```python\n{invalid_blocks[0]}\n```"
        )
        fix_span = None
        if trace:
            fix_span = trace.start_observation(
                name="llm-code-syntax-fix",
                as_type="generation",
                model=_llm_model_name(llm_client),
                input=fix_prompt,
            )
        try:
            fixed = await llm_client.generate(fix_prompt)
            fixed_code = fixed.strip().strip("`").removeprefix("python").strip()
            if fix_span:
                fix_span.update(output=fixed_code)
                fix_span.end()
            return answer_text.replace(invalid_blocks[0], fixed_code)
        except Exception:
            if fix_span:
                fix_span.update(output="syntax fix failed", level="ERROR")
                fix_span.end()
            logger.warning("Code syntax fix retry failed, returning original")
            return answer_text

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost in USD based on token usage and provider pricing.

        Pricing is approximate and based on common provider rates.
        For Ollama (local), cost is 0.
        """
        # Ollama is free (local inference)
        if not model or model.startswith(("llama", "qwen", "mistral", "gemma", "phi")):
            return 0.0

        # Approximate pricing per 1M tokens (as of 2024-2025)
        # Source: provider documentation
        pricing = {
            # OpenRouter models (approximate)
            "openrouter/": {"input": 0.0000003, "output": 0.0000006},
            # Cloudflare Workers AI
            "cf-": {"input": 0.000000011, "output": 0.000000036},
            # Groq
            "groq/": {"input": 0.00000005, "output": 0.00000008},
            # Cerebras
            "cerebras/": {"input": 0.0000001, "output": 0.0000001},
            # Gemini
            "gemini/": {"input": 0.000000125, "output": 0.000000375},
            # NVIDIA
            "nvidia/": {"input": 0.0000001, "output": 0.0000001},
        }

        # Find matching provider
        for prefix, rates in pricing.items():
            if model.lower().startswith(prefix.lower()) or prefix.lower() in model.lower():
                input_cost = prompt_tokens * rates["input"]
                output_cost = completion_tokens * rates["output"]
                return input_cost + output_cost

        # Default: assume free (Ollama or unknown)
        return 0.0

    async def close(self) -> None:
        """Close all underlying clients and connection pools. Idempotent."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        components = [
            self.vector_store,
            self.llm_client,
            self.embedder,
            self.code_llm_client,
            self.evaluation_llm_client,
            self.cache,
            self.reranker,
            self.groundedness_verifier,
            self.query_rewriter,
            self.context_compressor,
        ]
        for component in components:
            if component is None:
                continue
            with contextlib.suppress(TypeError, AttributeError, Exception):
                closer = None
                if hasattr(component, "aclose"):
                    closer = component.aclose
                elif hasattr(component, "close"):
                    closer = component.close
                if closer is not None:
                    await closer()
