"""Query rewriter: intent classification, multi-step decomposition, HyDE."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from data_engineering_copilot.domain.protocols import EmbedderProtocol, LLMClientProtocol

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

_CLASSIFY_INTENT_PROMPT = (
    'Classify the user query into exactly one category: "code_example" or "factual".\n'
    '- Choose "code_example" if the user wants code snippets, programming examples, or scripts.\n'
    '- Choose "factual" if the user wants conceptual explanations, documentation text, or theory.\n\n'
    'Query: "{query}"\n'
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
    "User question: {question}\n\nRewritten query:"
)


@dataclass(frozen=True)
class RewrittenQuery:
    original_query: str
    intent: str
    decomposed_steps: tuple[str, ...]
    hyde_query: str = ""


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

            prompt = _CLASSIFY_INTENT_PROMPT.format(query=query)

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
            )

        intent = self.classify_intent(query)
        steps = self.decompose(query, intent=intent)
        hyde = self._generate_hyde(query) if self._hyde_enabled else ""

        return RewrittenQuery(
            original_query=query,
            intent=intent,
            decomposed_steps=steps,
            hyde_query=hyde,
        )

    async def async_rewrite(self, query: str) -> RewrittenQuery:
        """LLM-based rewrite: classify intent, produce cleaned query via LLM.

        Falls back to rule-based rewrite if LLM is unavailable or errors.
        """
        if not self._enabled or self._llm_client is None:
            return self.rewrite(query)

        try:
            prompt = _REWRITE_PROMPT.format(question=query)
            llm_result = await self._llm_client.generate(prompt)
            rewritten = llm_result.strip()

            if not rewritten or len(rewritten) < 3:
                logger.warning("LLM rewrite returned empty result, falling back to rule-based")
                return self.rewrite(query)

            intent = self.classify_intent(query)
            hyde = await self._generate_hyde_async(query) if self._hyde_enabled else ""

            return RewrittenQuery(
                original_query=query,
                intent=intent,
                decomposed_steps=(rewritten,),
                hyde_query=hyde,
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
                prompt = _CLASSIFY_INTENT_PROMPT.format(query=query)
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

        Uses the LLM to generate paraphrases and related queries.
        Falls back to original query on failure.
        """
        if not self._enabled or self._llm_client is None:
            return [query]

        prompt = (
            f"Generate {max_variations} different search queries that would find "
            f"the same information as this question. Return ONLY the queries, "
            f"one per line, no numbering.\n\n"
            f"Original question: {query}\n\nVariations:"
        )

        try:
            result = await self._llm_client.generate(prompt)
            variations = [q.strip() for q in result.strip().split("\n") if q.strip()]
            return [query] + variations[:max_variations]
        except Exception as exc:
            logger.warning("Query expansion failed, using original: %s", exc)
            return [query]

    # --- private helpers ---

    async def _generate_hyde_async(self, query: str) -> str:
        """Generate a hypothetical document answer for HyDE (async)."""
        if self._llm_client is None:
            return ""
        try:
            prompt = (
                "Write a short, authoritative paragraph that would perfectly answer "
                f"the following question. Do not address the user directly.\n\nQuestion: {query}"
            )
            result = await self._llm_client.generate(prompt)
            return str(result).strip() if result else ""
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
            prompt = (
                "Write a short, authoritative paragraph that would perfectly answer "
                f"the following question. Do not address the user directly.\n\nQuestion: {query}"
            )
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
            return str(result).strip() if result else ""
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
