from __future__ import annotations

import json
import re

import structlog

from data_engineering_copilot.domain.models import RetrievedChunk
from data_engineering_copilot.domain.protocols import LLMClientProtocol

logger = structlog.get_logger(__name__)


class RelevanceGrader:
    def __init__(self, llm_client: LLMClientProtocol) -> None:
        self.llm_client = llm_client

    async def grade_chunks(self, query: str, chunks: list[RetrievedChunk]) -> float:
        """Returns a relevance score in range [0.0, 1.0] by checking the chunks."""
        if not chunks:
            return 0.0

        context = "\n\n".join([c.chunk.text[:500] for c in chunks[:3]])

        prompt = f"""You are a relevance grader.
Analyze the retrieved context below and decide if it is relevant to the query and contains enough information to construct a helpful response.

Query: "{query}"

Retrieved Context:
{context}

Output ONLY a JSON object of this format:
{{
  "relevance_score": 0.0 to 1.0
}}
Do NOT output markdown fences, preamble, or explain your reasoning. Just the JSON.
"""
        clean_text = ""
        try:
            response = await self.llm_client.generate(prompt)
            clean_text = response.strip()
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", clean_text, re.DOTALL)
            if m:
                clean_text = m.group(1).strip()
            data = json.loads(clean_text)
            return float(data.get("relevance_score", 0.0))
        except Exception as e:
            logger.warning("failed_to_grade_relevance", error=str(e), response=clean_text)
            return 1.0
