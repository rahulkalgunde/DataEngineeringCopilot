"""Query rewriter: intent classification, multi-step decomposition, HyDE."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("comparative", re.compile(r"\b(compare|vs\.?|versus|difference between|pros and cons)\b", re.IGNORECASE)),
    ("debugging", re.compile(r"\b(why is|error|fail|bug|oom|exception|not working|broken|crash|issue|problem)\b", re.IGNORECASE)),
    ("how_to", re.compile(r"\b(how (to|do|can|should)|step[- ]by[- ]step|guide|tutorial|set up|configure|install)\b", re.IGNORECASE)),
    ("factual", re.compile(r".*", re.DOTALL)),  # fallback
]


@dataclass(frozen=True)
class RewrittenQuery:
    original_query: str
    intent: str
    decomposed_steps: tuple[str, ...]
    hyde_query: str = ""


class QueryRewriter:
    """Lightweight rule-based query rewriter.

    - Intent classification via regex (no LLM needed)
    - Multi-step decomposition via rule-based heuristics
    - Optional HyDE (hypothetical document embedding) via LLM client
    """

    def __init__(
        self,
        llm_client: object | None,
        enabled: bool = True,
        hyde_enabled: bool = False,
    ) -> None:
        self._llm_client = llm_client
        self._enabled = enabled
        self._hyde_enabled = hyde_enabled

    def classify_intent(self, query: str) -> str:
        """Classify query intent into factual / comparative / how_to / debugging."""
        if not self._enabled:
            return "factual"

        for intent, pattern in _INTENT_PATTERNS:
            if intent == "factual":
                continue  # checked last as fallback
            if pattern.search(query):
                return intent
        return "factual"

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

    # --- private helpers ---

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
        m = re.search(r"(?:why|what).+?(?:failing|error|broken|oom|crash)\s+(.+?)(?:\?|$)", query, re.IGNORECASE)
        if m:
            context = m.group(0).strip()
            return (
                f"What causes {context}?",
                f"How to fix {context}?",
            )
        return (query,)
