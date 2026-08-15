"""Query rewriter: intent classification, multi-step decomposition, HyDE."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from data_engineering_copilot.domain.models import RetrievalFilters
from data_engineering_copilot.domain.protocols import EmbedderProtocol, LLMClientProtocol
from data_engineering_copilot.infrastructure.llm_client import SYSTEM_BLOCK_SEPARATOR
from data_engineering_copilot.observability.langfuse_prompts import get_langfuse_prompt, register_fallback

logger = logging.getLogger(__name__)

# Comprehensive pattern matching explicit code intent
_CODE_INTENT_PATTERN = re.compile(
    r"\b("
    r"(give|show|provide|send|write|generate|get)\s+(me\s+)?(a\s+)?\w*\s*(sample|example|snippet|code)"
    r"|(give|show|provide|send|write|generate|get)\s+(me\s+)?(the\s+)?(sample\s+|example\s+)?code"
    r"|how\s+to\s+(write|code|implement|build|script|program)"
    r"|code\s+(to|for|example|snippet|sample)"
    r"|write\s+(a\s+)?(script|function|program|query|pipeline)"
    r")\b",
    re.IGNORECASE,
)

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "api_lookup",
        re.compile(
            r"\b(\w+\.\w+\s*\(|DataFrame\.|SparkSession\.|spark\.read|spark\.sql|\.groupBy|\.select|\.filter|\.join|\.agg|\.write|\.show)\b",
            re.IGNORECASE,
        ),
    ),
    ("code_example", _CODE_INTENT_PATTERN),
    ("comparative", re.compile(r"\b(compare|vs\.?|versus|difference between|pros and cons)\b", re.IGNORECASE)),
    (
        "debugging",
        re.compile(r"\b(why is|error|fail|bug|oom|exception|not working|broken|crash|issue|problem)\b", re.IGNORECASE),
    ),
    (
        "how_to",
        re.compile(
            r"\b(how (to|do|can|should)|step[- ]by[- ]step|guide|tutorial|set up|configure|install)\b", re.IGNORECASE
        ),
    ),
    ("factual", re.compile(r".*", re.DOTALL)),  # fallback
]

# Pattern to detect code-related keywords in queries (for safety net)
_CODE_KEYWORDS = re.compile(
    r"\b(code|script|function|implement|snippet|sample|example|pyspark|scala|python)\b",
    re.IGNORECASE,
)

# --- Degenerate-rewrite detection ---------------------------------------------
# LLM query rewrites occasionally echo prompt boilerplate or emit placeholder
# SQL templates instead of a real search query. Such strings dominate the fused
# retrieval pool with irrelevant chunks, so they are rejected and replaced by
# the original question (general quality guard, not per-query tuning).
_DEGENERATE_REWRITE_PATTERNS = (
    # Placeholder SQL templates ("SELECT column_name FROM table_name ...")
    re.compile(r"\b(column_name|table_name|placeholder)\b", re.IGNORECASE),
    # Prompt-framing echo ("Here are two different search queries ...",
    # "Original question:", "Rewritten query:", "Return ONLY the queries ...").
    re.compile(
        r"^\s*(here (are|is)|the (queries|search queries|variations) (are|is)|"
        r"original question:|rewritten query:|return only|you are )",
        re.IGNORECASE,
    ),
    # Numbered / bulleted list artifacts (the prompt forbids numbering).
    re.compile(r"^\s*(?:[-*] |\d+[.)]\s+|`?\d+\.\s)", re.IGNORECASE),
)

_CLASSIFY_INTENT_PROMPT = (
    'Classify the user query into exactly one category: "code_example" or "factual".\n'
    '- Choose "code_example" if the user wants code snippets, programming examples, or scripts.\n'
    '- Choose "factual" if the user wants conceptual explanations, documentation text, or theory.\n\n'
    + SYSTEM_BLOCK_SEPARATOR
    + 'Query: "{query}"\n'
    'JSON Response: {{"intent": "code_example" | "factual"}}'
)

_REWRITE_PROMPT = (
    "You are a search query rewriter. Given a user question, produce a concise, "
    "search-optimized query that would best retrieve relevant documentation.\n"
    "Rules:\n"
    "- Return ONLY the rewritten query, no preamble.\n"
    "- Preserve the user's intent.\n"
    "- Expand abbreviations and jargon where helpful.\n"
    "- Output a single line, no more than 30 words.\n\n"
    + SYSTEM_BLOCK_SEPARATOR
    + "User question: {question}\n\nRewritten query:"
)

_REWRITE_CONTEXTUAL_PROMPT = (
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
    + "## CONVERSATION TOPIC\n{session_topic}\n\n"
    "## CONVERSATION HISTORY\n{conversation_history}\n\n"
    "Latest user message: {question}\n\nRewritten query:"
)

_EXPAND_PROMPT = (
    "Generate {max_variations} different search queries that would find "
    "the same information as this question. Return ONLY the queries, "
    "one per line, no numbering.\n\n" + SYSTEM_BLOCK_SEPARATOR + "Original question: {query}\n\nVariations:"
)

_HYDE_PROMPT = (
    "Write a short, authoritative paragraph that would perfectly answer "
    "the following question. Do not address the user directly.\n\n" + SYSTEM_BLOCK_SEPARATOR + "Question: {query}"
)

register_fallback("query-intent-classify", _CLASSIFY_INTENT_PROMPT)
register_fallback("query-rewrite", _REWRITE_PROMPT)
register_fallback("query-rewrite-contextual", _REWRITE_CONTEXTUAL_PROMPT)
register_fallback("query-expand", _EXPAND_PROMPT)
register_fallback("query-hyde", _HYDE_PROMPT)


@dataclass(frozen=True)
class RewrittenQuery:
    original_query: str
    intent: str
    decomposed_steps: tuple[str, ...]
    hyde_query: str = ""
    filters: RetrievalFilters = field(default_factory=lambda: RetrievalFilters())


# Deterministic exact API/module identifier extraction for retrieval.
_DOTTED_IDENTIFIER_RE = re.compile(r"\b(pyspark(?:\.\w+)+)\b")
_VERSION_RE = re.compile(r"\b(?:spark\s*)?v?(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE)
# Known Spark function terms mapped to a module preference (not a hard filter).
_MODULE_TERMS: dict[str, tuple[str, ...]] = {
    "filter": ("pyspark.sql.functions", "pyspark.sql"),
    "transform": ("pyspark.sql.functions",),
    "aggregate": ("pyspark.sql.functions",),
    "window": ("pyspark.sql.window", "pyspark.sql"),
    "dense_rank": ("pyspark.sql.functions",),
    "row_number": ("pyspark.sql.functions",),
    "col": ("pyspark.sql.functions",),
}


def is_degenerate_query(text: str) -> bool:
    """Return True when an LLM-generated query is degenerate boilerplate.

    Detects placeholder SQL templates (``column_name``/``table_name``),
    prompt-framing echo, and numbered/bulleted list artifacts. Used to reject
    low-quality LLM rewrites before they pollute the fused retrieval pool.
    """
    stripped = text.strip()
    if len(stripped) < 4:
        return True
    return any(pattern.search(stripped) for pattern in _DEGENERATE_REWRITE_PATTERNS)


def render_conversation_history(messages) -> str:
    """Render a conversation transcript for prompt injection.

    ``messages`` is an iterable of objects with ``role`` (user/assistant/system)
    and ``content`` attributes (e.g. ``ChatMessage``). Rendered as alternating
    ``User:`` / ``Assistant:`` lines, oldest first. Non user/assistant roles
    (e.g. ``system``) are skipped, and empty history renders an empty string so
    callers can skip the contextual rewrite entirely.
    """
    lines: list[str] = []
    for message in messages:
        role = message.role
        if role not in ("user", "assistant"):
            continue
        content = (message.content or "").strip()
        if not content:
            continue
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def extract_retrieval_constraints(query: str) -> RetrievalFilters:
    """Extract deterministic retrieval constraints from a query.

    Exact dotted PySpark identifiers (e.g. ``pyspark.sql.functions.filter``)
    become hard ``modules`` filters. Known function terms (e.g. ``dense_rank``)
    map to ``preferred_modules`` soft preferences — never hard filters — so
    guide/example chunks without a module remain retrievable. A mutable
    ``latest`` version is never inferred as a release.
    """
    languages: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    preferred_modules: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()

    for match in _DOTTED_IDENTIFIER_RE.finditer(query):
        modules = modules + (match.group(1).lower(),)

    lower = query.lower()
    for term, pref in _MODULE_TERMS.items():
        if term in lower:
            preferred_modules = preferred_modules + pref

    for match in _VERSION_RE.finditer(query):
        version = match.group(1)
        if "latest" not in lower:
            versions = versions + (version,)

    # De-duplicate while preserving order.
    modules = tuple(dict.fromkeys(modules))
    preferred_modules = tuple(dict.fromkeys(preferred_modules))
    versions = tuple(dict.fromkeys(versions))

    return RetrievalFilters(
        languages=languages,
        modules=modules,
        preferred_modules=preferred_modules,
        versions=versions,
    )


class QueryRewriter:
    """Lightweight rule-based query rewriter with optional LLM fallback.

    - Intent classification via regex (fast path) + optional LLM fallback
    - Multi-step decomposition via rule-based heuristics
    - Optional HyDE (hypothetical document embedding) via LLM client
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol | None,
        enabled: bool = True,
        hyde_enabled: bool = True,
        intent_llm_enabled: bool = False,
        intent_llm_client: LLMClientProtocol | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._intent_llm_client = intent_llm_client or llm_client
        self._enabled = enabled
        self._hyde_enabled = hyde_enabled
        self._intent_llm_enabled = intent_llm_enabled

    def classify_intent(self, query: str) -> str:
        """Classify query intent into factual / comparative / how_to / debugging / code_example.

        Uses regex fast-path first. If no match and intent_llm_enabled is True,
        falls back to LLM classifier for ambiguous queries.
        """
        if not self._enabled:
            return "factual"

        # Fast path: regex classification
        for intent, pattern in _INTENT_PATTERNS:
            if intent == "factual":
                continue  # checked last as fallback
            if pattern.search(query):
                return intent

        # Fallback: LLM classifier for ambiguous queries (if enabled)
        if self._intent_llm_enabled and self._intent_llm_client is not None:
            llm_intent = self._classify_intent_with_llm(query)
            if llm_intent is not None:
                return llm_intent

        return "factual"

    def _classify_intent_with_llm(self, query: str) -> str | None:
        """Use LLM to classify intent for ambiguous queries.

        Returns intent string or None on failure.
        """
        client = self._intent_llm_client
        if client is None:
            return None
        try:
            import asyncio

            prompt = get_langfuse_prompt("query-intent-classify").compile(query=query)

            # Try async first if event loop is running
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # Cannot use asyncio.run() in async context; skip LLM fallback
                    logger.debug("LLM intent classification skipped: event loop already running")
                    return None
            except RuntimeError:
                pass

            result = asyncio.run(client.generate(prompt))
            if not result:
                return None

            # Parse JSON response
            parsed = json.loads(result.strip())
            intent = parsed.get("intent", "")
            if intent in ("code_example", "factual"):
                logger.debug("LLM intent classification: %s (query=%r)", intent, query[:80])
                return intent
            return None
        except Exception as exc:
            logger.warning("LLM intent classification failed, falling back to regex: %s", exc)
            return None

    def decompose(self, query: str, *, intent: str = "factual") -> tuple[str, ...]:
        """Break a query into sub-steps based on intent."""
        if not self._enabled:
            return (query,)

        if intent == "comparative":
            return self._decompose_comparative(query)
        if intent == "how_to":
            return self._decompose_how_to(query)
        if intent == "debugging":
            return self._decompose_debugging(query)
        if intent == "api_lookup":
            return self._decompose_api_lookup(query)
        if intent == "code_example":
            return self._decompose_code_example(query)
        # factual: single step
        return (query,)

    def rewrite(self, query: str) -> RewrittenQuery:
        """Full rewrite pipeline: classify → decompose → optional HyDE."""
        if not self._enabled:
            return RewrittenQuery(
                original_query=query,
                intent="factual",
                decomposed_steps=(query,),
                hyde_query="",
                filters=extract_retrieval_constraints(query),
            )

        intent = self.classify_intent(query)
        steps = self.decompose(query, intent=intent)
        hyde = self._generate_hyde(query) if self._hyde_enabled else ""

        return RewrittenQuery(
            original_query=query,
            intent=intent,
            decomposed_steps=steps,
            hyde_query=hyde,
            filters=extract_retrieval_constraints(query),
        )

    async def async_rewrite(
        self, query: str, conversation_history=None, session_topic: str | None = None
    ) -> RewrittenQuery:
        """LLM-based rewrite: classify intent, produce cleaned query via LLM.

        When ``conversation_history`` (an iterable of ``ChatMessage``-like
        objects) is non-empty, the rewrite is context-aware: pronouns and
        references are resolved into a standalone query using the
        ``query-rewrite-contextual`` prompt. Turn 1 (empty history) uses the
        plain ``query-rewrite`` prompt — identical to previous behavior, so
        single-turn latency/quality are unchanged.

        ``session_topic`` (the session's anchor topic, typically the first user
        message) is appended to the contextual prompt so terse follow-ups
        (e.g. "give an example in python") stay anchored to the conversation's
        subject instead of drifting to unrelated retrieved content.

        Falls back to rule-based rewrite if LLM is unavailable or errors.
        """
        if not self._enabled or self._llm_client is None:
            return self.rewrite(query)

        try:
            history_text = render_conversation_history(conversation_history) if conversation_history else ""
            if history_text.strip():
                prompt = get_langfuse_prompt("query-rewrite-contextual").compile(
                    conversation_history=history_text,
                    question=query,
                    session_topic=session_topic or "",
                )
            else:
                prompt = get_langfuse_prompt("query-rewrite").compile(question=query)
            llm_result = await self._llm_client.generate(prompt)
            rewritten = llm_result.strip()

            if not rewritten or len(rewritten) < 3:
                logger.warning("LLM rewrite returned empty result, falling back to rule-based")
                return self.rewrite(query)

            if is_degenerate_query(rewritten):
                logger.warning("LLM rewrite produced degenerate query %r, falling back to rule-based", rewritten[:80])
                return self.rewrite(query)

            intent = self.classify_intent(query)
            hyde = await self._generate_hyde_async(query) if self._hyde_enabled else ""

            return RewrittenQuery(
                original_query=query,
                intent=intent,
                decomposed_steps=(rewritten,),
                hyde_query=hyde,
                filters=extract_retrieval_constraints(query),
            )
        except Exception as exc:
            logger.warning("LLM rewrite failed, falling back to rule-based: %s", exc)
            return self.rewrite(query)

    async def async_classify_intent(self, query: str) -> str:
        """Async LLM-based intent classification with regex fast-path.

        Uses LLM fallback if enabled and regex doesn't match.
        """
        if not self._enabled:
            return "factual"

        # Fast path: regex classification
        for intent, pattern in _INTENT_PATTERNS:
            if intent == "factual":
                continue
            if pattern.search(query):
                return intent

        # Fallback: async LLM classifier (if enabled)
        if self._intent_llm_enabled and self._intent_llm_client is not None:
            try:
                prompt = get_langfuse_prompt("query-intent-classify").compile(query=query)
                result = await self._intent_llm_client.generate(prompt)
                if result:
                    parsed = json.loads(result.strip())
                    intent = parsed.get("intent", "")
                    if intent in ("code_example", "factual"):
                        logger.debug("Async LLM intent classification: %s (query=%r)", intent, query[:80])
                        return intent
            except Exception as exc:
                logger.warning("Async LLM intent classification failed: %s", exc)

        return "factual"

    async def hyde(
        self,
        question: str,
        embedder: EmbedderProtocol,
    ) -> list[float]:
        """Generate a hypothetical document and return its embedding.

        Plan spec: ``hyde(question, embedder) → LLM generates hypothetical
        answer → embed that``.
        """
        hyde_text = await self._generate_hyde_async(question)
        if not hyde_text:
            return await embedder.embed_query(question)
        return await embedder.embed_query(hyde_text)

    async def expand_queries(self, query: str, max_variations: int = 3) -> list[str]:
        """Generate multiple query variations for improved recall.

        Uses the LLM to generate paraphrases and related queries, then appends
        deterministic Spark SQL retrieval variants when the query contains
        window/rank or array/struct indicators so the exact indexed doc terms
        (``PARTITION BY``, ``RANGE BETWEEN``, ``dense_rank``, ``filter``,
        ``transform``, ``aggregate``) are always represented. Falls back to the
        original query on failure.
        """
        base = [query]
        if not self._enabled or self._llm_client is None:
            return base

        prompt = get_langfuse_prompt("query-expand").compile(max_variations=max_variations, query=query)

        try:
            result = await self._llm_client.generate(prompt)
            variations = [q.strip() for q in result.strip().split("\n") if q.strip()]
            variations = [q for q in variations if not is_degenerate_query(q)]
            base = [query] + variations[:max_variations]
        except Exception as exc:
            logger.warning("Query expansion failed, using original: %s", exc)
            base = [query]

        # Deterministic Spark-specific variants using terms present in the
        # indexed Spark source docs.
        spark_variants = self._spark_retrieval_variants(query)
        merged = list(base)
        for variant in spark_variants:
            if variant not in merged:
                merged.append(variant)
        return merged

    @staticmethod
    def _spark_retrieval_variants(query: str) -> list[str]:
        """Return Spark SQL retrieval queries derived from window/array indicators.

        Detects window/ranking intent (rolling, window, dense_rank, rank,
        partition, order) and array/struct intent (array, struct, nested,
        filter, transform, aggregate) and returns queries phrased with the
        exact terms used in the indexed Spark docs.
        """
        lower = query.lower()
        variants: list[str] = []

        if any(k in lower for k in ("rolling", "window", "dense_rank", "rank", "partition", "over (")):
            variants.extend(
                [
                    "Spark SQL window functions syntax PARTITION BY ORDER BY RANGE BETWEEN",
                    "PySpark Window partitionBy orderBy dense_rank rangeBetween sum",
                    "window functions examples dense_rank RANK ROW_NUMBER OVER PARTITION BY",
                ]
            )

        if any(k in lower for k in ("array", "struct", "nested", "explode", "flatten")):
            variants.extend(
                [
                    "Spark SQL array functions filter transform aggregate",
                    "PySpark filter transform aggregate ArrayType StructType nested",
                ]
            )
            if "filter" in lower or "discount" in lower:
                variants.append("filter elements of array of structs without explode")
            if "aggregate" in lower or "sum" in lower or "net" in lower:
                variants.append("aggregate array of structs sum price discount net_total")

        return list(dict.fromkeys(variants))

    # --- private helpers ---

    async def _generate_hyde_async(self, query: str) -> str:
        """Generate a hypothetical document answer for HyDE (async)."""
        if self._llm_client is None:
            return ""
        try:
            prompt = get_langfuse_prompt("query-hyde").compile(query=query)
            result = await self._llm_client.generate(prompt)
            text = str(result).strip() if result else ""
            if not text or is_degenerate_query(text):
                logger.warning("HyDE produced degenerate text %r, dropping", text[:80])
                return ""
            return text
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return ""

    def _generate_hyde(self, query: str) -> str:
        """Generate a hypothetical document answer for HyDE.

        Returns empty string if no LLM client is available.
        """
        if self._llm_client is None:
            return ""
        try:
            prompt = get_langfuse_prompt("query-hyde").compile(query=query)
            # Sync wrapper — caller should use async if available
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                logger.warning("HyDE generation skipped: event loop already running")
                return ""

            result = asyncio.run(self._llm_client.generate(prompt))
            text = str(result).strip() if result else ""
            if not text or is_degenerate_query(text):
                logger.warning("HyDE produced degenerate text %r, dropping", text[:80])
                return ""
            return text
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return ""

    def _decompose_comparative(self, query: str) -> tuple[str, ...]:
        """Split 'Compare X vs Y' into entity-specific sub-queries."""
        m = re.search(
            r"(?:compare|difference between)\s+(.+?)\s+(?:vs\.?|versus|and)\s+(.+?)(?:\?|$)",
            query,
            re.IGNORECASE,
        )
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            return (
                f"What is {left}?",
                f"What is {right}?",
                f"What are the differences between {left} and {right}?",
            )
        # Fallback: try splitting on 'vs'
        parts = re.split(r"\s+vs\.?\s+", query, flags=re.IGNORECASE)
        if len(parts) >= 2:
            left, right = parts[0].strip(), parts[1].strip().rstrip("?")
            return (
                f"What is {left}?",
                f"What is {right}?",
                f"{query}",
            )
        return (query,)

    def _decompose_how_to(self, query: str) -> tuple[str, ...]:
        """Break 'How to X' into prerequisite + steps sub-queries."""
        m = re.search(r"how (?:to|do|can)\s+(.+?)(?:\?|$)", query, re.IGNORECASE)
        if m:
            topic = m.group(1).strip()
            return (
                f"What are the prerequisites for {topic}?",
                f"What are the steps to {topic}?",
            )
        return (query,)

    def _decompose_debugging(self, query: str) -> tuple[str, ...]:
        """Break debugging query into cause + solution sub-queries."""
        m = re.search(r"(?:why|what)\s+.+?(?:failing|error|broken|oom|crash)", query, re.IGNORECASE)
        if m:
            context = m.group(0).strip()
            return (
                f"What causes {context}?",
                f"How to fix {context}?",
            )
        return (query,)

    def _decompose_api_lookup(self, query: str) -> tuple[str, ...]:
        """Expand API lookup into signature + parameters + examples."""
        # Extract the method name from patterns like spark.read.parquet() or DataFrame.groupBy()
        m = re.search(r"(\w+(?:\.\w+)*)\s*\(", query)
        if m:
            method = m.group(1)
            return (
                f"What is the signature of {method}?",
                f"What are the parameters of {method}?",
                f"{query}",
            )
        return (query,)

    def _decompose_code_example(self, query: str) -> tuple[str, ...]:
        """Expand code example query into implementation + usage sub-queries."""
        m = re.search(r"(?:write|code|implement|show)\s+(.+?)(?:\?|$)", query, re.IGNORECASE)
        if m:
            topic = m.group(1).strip()
            return (
                f"Show me a code example for {topic}",
                f"What is the recommended way to implement {topic}?",
            )
        return (query,)
