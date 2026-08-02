from __future__ import annotations

import ast
import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable

from data_engineering_copilot.domain.exceptions import LLMGenerationError, RetrievalError
from data_engineering_copilot.domain.models import Answer, CachedAnswer, CacheScope, RagConfig
from data_engineering_copilot.domain.protocols import (
    EmbedderProtocol,
    LLMClientProtocol,
    RerankerProtocol,
    TelemetryTracerProtocol,
    VectorStoreProtocol,
)
from data_engineering_copilot.infrastructure.pii_redactor import PiiRedactor
from data_engineering_copilot.services.context_assembler import ContextAssembler
from data_engineering_copilot.services.context_compression import ContextCompressor
from data_engineering_copilot.services.groundedness import GroundednessVerifier
from data_engineering_copilot.services.prompt_builder import CODE_INTENTS, PromptBuilder
from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
from data_engineering_copilot.services.query_rewriting import QueryRewriter
from data_engineering_copilot.services.structured_output import parse_rag_response, verify_citations

logger = logging.getLogger(__name__)


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


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
        context_compressor: ContextCompressor | None = None,
        token_tracker: object | None = None,
        retrieval_tracker: object | None = None,
        code_llm_client: LLMClientProtocol | None = None,
        evaluation_llm_client: LLMClientProtocol | None = None,
        pii_redactor: PiiRedactor | None = None,
        input_guardrails: object | None = None,
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
        self.context_compressor = context_compressor
        self.token_tracker = token_tracker
        self.retrieval_tracker = retrieval_tracker
        self._pii_redactor = pii_redactor
        self.input_guardrails = input_guardrails
        self._prompt_builder = PromptBuilder()

    async def answer(
        self,
        question: str,
        on_step: Callable[[str], None] | None = None,
        source_filter: list[str] | None = None,
        cache_scope: CacheScope | None = None,
    ) -> Answer:
        _t0 = time.monotonic()
        _stage_times: dict[str, float] = {}

        def _record_stage(name: str) -> None:
            _stage_times[name] = round((time.monotonic() - _t0) * 1000, 1)

        query_emb_for_cache: list[float] | None = None
        if self.cache is not None:
            # Exact tier first — no embedding round-trip on an exact hit.
            cached = await self.cache.aget(question, scope=cache_scope)
            if cached is not None:
                logger.info("cache_hit question=%r", question[:80])
                _record_stage("cache_lookup")
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
                return Answer(
                    text=cached.text,
                    sources=cached.sources,
                    confidence=cached.confidence,
                    groundedness_score=cached.groundedness_score,
                    stage_times=_stage_times,
                )

        trace = None
        if self.telemetry:
            trace = self.telemetry.start_observation(
                name="rag-query-pipeline",
                input=question,
                as_type="trace",
            )

        # Phase 2A: Query rewriting — collect all queries for multi-step retrieval
        rewritten = None
        all_queries: list[str] = [question]  # Always include original
        effective_query = question
        if self.query_rewriter is not None:
            rewritten = await self.query_rewriter.async_rewrite(question)
            if rewritten.decomposed_steps:
                all_queries.extend(rewritten.decomposed_steps)
                effective_query = rewritten.decomposed_steps[0]
            # Multi-query expansion: generate additional variations
            expanded = await self.query_rewriter.expand_queries(question, max_variations=self.config.max_expansion_queries)
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

        if on_step:
            on_step("Embedding query")
        logger.info("async_rag.answer question=%r queries=%d", question[:100], len(all_queries))

        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")

        # Determine chunk type filter based on intent
        chunk_type_filter = None
        if rewritten is not None:
            if rewritten.intent == "api_lookup":
                chunk_type_filter = "api"
            elif rewritten.intent == "code_example":
                chunk_type_filter = "code"

        # Embed the HyDE hypothesis once and reuse the vector across all sub-queries,
        # instead of regenerating the hypothetical text via the LLM for every query.
        hyde_emb = None
        if self.query_rewriter is not None and rewritten is not None and rewritten.hyde_query:
            hyde_emb = await self.embedder.embed_query(rewritten.hyde_query)

        # Retrieve with all query variations and merge results
        all_retrieved: list = []
        seen_ids: set[str] = set()
        any_success = False
        last_error: Exception | None = None
        try:
            for q in all_queries:
                try:
                    q_emb = hyde_emb if hyde_emb is not None else await self.embedder.embed_query(q)
                    results = await self.vector_store.query(
                        q_emb,
                        top_k=self.config.retrieval_top_k,
                        query_text=q,
                        source_filter=source_filter,
                        chunk_type_filter=chunk_type_filter,
                    )
                    any_success = True
                    for r in results:
                        cid = r.chunk.chunk_id
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            all_retrieved.append(r)
                except Exception as sub_exc:
                    last_error = sub_exc
                    logger.warning("Failed to retrieve for sub-query %r: %s", q[:50], sub_exc)

            if not any_success and last_error is not None:
                raise RetrievalError(f"Vector store query failed: {last_error}") from last_error

            # Sort merged results by confidence
            retrieved_chunks = sorted(all_retrieved, key=lambda c: c.confidence, reverse=True)

            if retrieval_span:
                retrieval_span.update(
                    output=[c.chunk.text for c in retrieved_chunks],
                    input=effective_query,
                )
                retrieval_span.end()
            _record_stage("retrieval")
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
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        if retrieved_chunks[0].confidence < self.config.confidence_threshold:
            if trace:
                trace.update(
                    output=f"Low confidence: {retrieved_chunks[0].confidence:.4f} < threshold {self.config.confidence_threshold:.4f}"
                )
                trace.end()
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
                return Answer(
                    text="I cannot answer this question because it is outside my knowledge repository.",
                    sources=tuple(),
                    confidence=0.0,
                )

        generation_span = None
        if trace:
            generation_span = trace.start_observation(
                name="ollama-generation",
                as_type="generation",
            )

        try:
            if on_step:
                on_step("Reranking results")
            if (
                self.config.reranker_enabled
                and self.reranker is not None
                and self.reranker.is_available()
                and len(retrieved_chunks) > 1
            ):
                retrieved_chunks = await self.reranker.rerank(
                    effective_query, retrieved_chunks, top_k=self.config.reranker_top_k
                )

            # MMR diversity reranking — ensures diverse context
            if len(retrieved_chunks) > 3 and self.reranker is not None:
                retrieved_chunks = self.reranker.diversify_by_lexical_content(
                    retrieved_chunks, top_k=self.config.reranker_top_k
                )

            # Phase 2D: Context compression (dedup + relevance re-ranking) — runs after reranking
            # to avoid wasted work (compressed results were previously discarded by re-fetch)
            if self.context_compressor is not None:
                retrieved_chunks = self.context_compressor.compress(retrieved_chunks, effective_query)
                logger.info("context_compressed chunks=%d", len(retrieved_chunks))
            _record_stage("rerank")

            assembler = ContextAssembler(max_context_chars=self.config.max_context_chars)
            context_str, source_names = assembler.assemble(
                retrieved_chunks,
                deduplicate=self.context_compressor is None,
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

            if generation_span:
                generation_span.update(input=prompt)

            llm_client = self._select_llm_client(intent)
            answer_text = await llm_client.generate(prompt)

            # JSON retry: if the intent expects JSON output but parsing fails,
            # retry once with a stricter instruction.
            if intent not in CODE_INTENTS and isinstance(answer_text, str):
                parsed_attempt = parse_rag_response(answer_text)
                if not parsed_attempt.answer and len(answer_text.strip()) > 20:
                    logger.info("json_parse_retry intent=%s response_len=%d", intent, len(answer_text))
                    retry_prompt = prompt + (
                        "\n\nIMPORTANT: Your previous response was not valid JSON. "
                        "Return ONLY raw JSON with no markdown, no code fences, no preamble."
                    )
                    answer_text = await llm_client.generate(retry_prompt)

            _record_stage("generation")

            # Post-generation syntax check for code intents
            answer_text = await self._validate_and_fix_code_syntax(answer_text, intent, llm_client)

            # Track token usage from LLM provider
            if self.token_tracker is not None and hasattr(llm_client, "last_usage"):
                usage = getattr(llm_client, "last_usage", None)
                if usage is not None:
                    self.token_tracker.record(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        model=usage.model,
                    )

            # Output guardrails: verify structure and quality
            from data_engineering_copilot.services.output_guardrails import OutputGuardrails

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
            result = Answer(
                text=answer_text,
                sources=tuple(c.chunk for c in retrieved_chunks),
                confidence=retrieved_chunks[0].confidence,
                stage_times=_stage_times,
            )

            # Phase 2B: Groundedness verification (annotate-only, fail-open)
            groundedness_score = 1.0
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
                    result = Answer(
                        text=result.text + "\n\n[Note: Some claims may not be fully supported by the documentation.]",
                        sources=result.sources,
                        confidence=result.confidence,
                        groundedness_score=groundedness_score,
                    )
                else:
                    result = Answer(
                        text=result.text,
                        sources=result.sources,
                        confidence=result.confidence,
                        groundedness_score=groundedness_score,
                    )

            # Phase 2C: Cache the result (exact + semantic tiers)
            if self.cache is not None:
                envelope = CachedAnswer(
                    text=result.text,
                    sources=result.sources,
                    confidence=result.confidence,
                    groundedness_score=result.groundedness_score,
                    cached_at=time.time(),
                )
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
            raise
        except Exception as exc:
            logger.exception("Failed during answer generation: %s", exc)
            if generation_span:
                generation_span.update(output=str(exc), level="ERROR")
                generation_span.end()
            if trace:
                trace.update(output=str(exc))
                trace.end()
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc

    async def answer_stream(
        self,
        question: str,
        source_filter: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream answer tokens via SSE while performing retrieval.

        Yields JSON-encoded SSE events:
        - ``{"type": "status", "message": "..."}`` for pipeline progress
        - ``{"type": "token", "content": "..."}`` for answer tokens
        - ``{"type": "done", "text": "...", "confidence": 0.0}`` at completion
        """
        yield _sse({"type": "status", "message": "Sanitizing query"})

        # PII redaction
        safe_question = PromptBuilder.sanitize_query(question)
        if self._pii_redactor is not None:
            safe_question, _pii_types = self._pii_redactor.redact(safe_question)
            if _pii_types:
                yield _sse({"type": "status", "message": f"PII redacted: {', '.join(_pii_types)}"})

        # Query rewriting
        rewritten = None
        if self.query_rewriter is not None:
            yield _sse({"type": "status", "message": "Rewriting query"})
            try:
                rewritten = await self.query_rewriter.async_rewrite(safe_question)
            except Exception:
                logger.warning("Query rewrite failed during streaming, using raw query", exc_info=True)
                rewritten = None
            if rewritten is not None:
                yield _sse({"type": "status", "message": f"Intent: {rewritten.intent}"})

        if rewritten is not None and rewritten.decomposed_steps:
            effective_query = rewritten.decomposed_steps[0]
        else:
            effective_query = safe_question
        intent = rewritten.intent if rewritten else "factual"

        # Retrieval
        yield _sse({"type": "status", "message": "Retrieving documents"})
        try:
            query_emb = await self.embedder.embed_query(effective_query)
        except Exception as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        top_k = self.config.retrieval_top_k
        retrieved_chunks = await self.vector_store.query(
            query_embedding=query_emb,
            top_k=top_k,
            query_text=effective_query,
            source_filter=source_filter,
        )
        yield _sse({"type": "status", "message": f"Retrieved {len(retrieved_chunks)} chunks"})

        if not retrieved_chunks:
            yield _sse({"type": "done", "text": "No relevant documents found.", "confidence": 0.0})
            return

        # Reranking
        if self.reranker is not None and self.reranker.is_available() and len(retrieved_chunks) > 1:
            yield _sse({"type": "status", "message": "Reranking"})
            retrieved_chunks = await self.reranker.rerank(
                query=effective_query,
                chunks=retrieved_chunks,
                top_k=self.config.reranker_top_k,
            )

        # Context assembly
        sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)
        assembler = ContextAssembler(max_context_chars=self.config.max_context_chars)
        context_str, _source_names = assembler.assemble(sorted_chunks)

        # Build prompt
        prompt = self._prompt_builder.build_rag_prompt(context=context_str, question=safe_question, intent=intent)
        yield _sse({"type": "status", "message": "Generating answer"})

        # Stream LLM tokens
        llm_client = self._select_llm_client(intent)
        full_text = ""
        try:
            async for token in llm_client.generate_stream(prompt):
                full_text += token
                yield _sse({"type": "token", "content": token})
        except Exception:
            logger.exception("Streaming generation failed")
            yield _sse({"type": "error", "message": "Generation failed"})
            return

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

        # Done event with metadata
        yield _sse({"type": "done", "text": full_text, "confidence": confidence})

    def _select_llm_client(self, intent: str) -> LLMClientProtocol:
        """Route code intents to the code-specific LLM if configured."""
        if self.code_llm_client and intent in CODE_INTENTS:
            return self.code_llm_client
        return self.llm_client

    async def _validate_and_fix_code_syntax(self, answer_text: str, intent: str, llm_client: LLMClientProtocol) -> str:
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
        try:
            fixed = await llm_client.generate(fix_prompt)
            fixed_code = fixed.strip().strip("`").removeprefix("python").strip()
            return answer_text.replace(invalid_blocks[0], fixed_code)
        except Exception:
            logger.warning("Code syntax fix retry failed, returning original")
            return answer_text

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
