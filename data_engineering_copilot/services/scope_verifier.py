"""Topic-scope verifier: fail-open check whether the context covers the question's topic.

The scope gate refuses answers when the retrieved documentation does not cover the
topic the user actually asked about (e.g. the retrieval pulled tangential material
from a different product). It complements ``GroundednessVerifier`` (claims in the
answer vs. context) by checking the *question's* topic vs. context.

Fail-open by design: only a confident ``does_not_cover`` verdict refuses; every
other outcome — errors, ambiguity, disabled state, ``covers`` or ``partially`` —
returns ``True`` (covered) so a genuine answer is never wrongly refused.
"""

from __future__ import annotations

import json
import logging
import re

from data_engineering_copilot.domain.protocols import LLMClientProtocol
from data_engineering_copilot.infrastructure.llm_client import SYSTEM_BLOCK_SEPARATOR
from data_engineering_copilot.observability.langfuse_prompts import get_langfuse_prompt, register_fallback

logger = logging.getLogger(__name__)

_SCOPE_PROMPT = (
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
    '{{"verdict": "covers" | "partially" | "does_not_cover", "reason": "..."}}\n\n'
    + SYSTEM_BLOCK_SEPARATOR
    + "QUESTION:\n{question}\n\n"
    "CONTEXT (excerpted from documentation):\n{context}\n\n"
    "JSON:"
)

register_fallback("scope-check", _SCOPE_PROMPT)


def _parse_verdict(raw: str) -> bool | None:
    """Return True (covered) / False (refuse) / None (unparseable → fail-open).

    Recognizes the 3-way ``verdict`` field first; falls back to a legacy
    ``covered`` boolean. Only an explicit ``does_not_cover`` (or ``covered:
    false``) refuses.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            verdict = data.get("verdict")
            if isinstance(verdict, str):
                verdict = verdict.strip().lower()
                if verdict == "does_not_cover":
                    return False
                if verdict in {"covers", "partially"}:
                    return True
            if isinstance(data.get("covered"), bool):
                return data["covered"]
    except (ValueError, TypeError):
        pass
    match = re.search(r'"verdict"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
    if match:
        verdict = match.group(1).strip().lower()
        return verdict != "does_not_cover"
    match = re.search(r'"covered"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return None


class ScopeVerifier:
    """Determines whether the retrieved context covers the question's topic (fail-open)."""

    def __init__(
        self,
        llm_client: LLMClientProtocol | None,
        enabled: bool = True,
    ) -> None:
        self._llm_client = llm_client
        self._enabled = enabled

    async def verify(self, question: str, context: str) -> bool:
        """Return True when the context covers the question's topic (fail-open).

        Only an explicit ``does_not_cover`` verdict (or legacy ``covered:
        false``) produces a refusal; errors, disabled state, ``covers`` /
        ``partially``, and unparseable output all return ``True``.
        """
        if not self._enabled or self._llm_client is None or not question.strip():
            return True
        try:
            prompt = get_langfuse_prompt("scope-check").compile(
                question=question[:500],
                context=context[:6000],
            )
            raw = await self._llm_client.generate(prompt)
            covered = _parse_verdict(raw)
            if covered is None:
                logger.warning("scope_check_unparseable response=%r; fail-open", (raw or "")[:200])
                return True
            logger.info("scope_check_covered=%s", covered)
            return covered
        except Exception as exc:
            logger.warning("scope_check_failed fail_open=%s", exc)
            return True
