from __future__ import annotations

import json
import re

import structlog

from data_engineering_copilot.domain.protocols import LLMClientProtocol
from data_engineering_copilot.infrastructure.graph_store import GraphStore

logger = structlog.get_logger(__name__)


class GraphExtractor:
    def __init__(self, llm_client: LLMClientProtocol, graph_store: GraphStore) -> None:
        self.llm_client = llm_client
        self.graph_store = graph_store

    async def extract_and_store(self, text: str) -> None:
        """Analyze text, extract triplets, and save to GraphStore."""
        prompt = f"""Extract key entity relationships (Subject, Relationship, Object) from the technical documentation snippet below.
Focus on architectural components, libraries, frameworks, configuration settings, or programming concepts.

Examples:
- Subject: Spark streaming, Relationship: handled_by, Object: Custom migration framework
- Subject: Celery, Relationship: uses_backend, Object: Redis

Text Snippet:
"{text}"

Output ONLY a JSON array of objects of this exact format:
[
  {{"source": "Subject", "target": "Object", "relation": "relationship_type"}}
]
Do NOT write markdown fences, preamble, or code blocks. Just valid JSON.
"""
        clean_text = ""
        try:
            response = await self.llm_client.generate(prompt)
            clean_text = response.strip()
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", clean_text, re.DOTALL)
            if m:
                clean_text = m.group(1).strip()

            triplets = json.loads(clean_text)
            if isinstance(triplets, list):
                for trip in triplets:
                    src = trip.get("source")
                    tgt = trip.get("target")
                    rel = trip.get("relation")
                    if src and tgt and rel:
                        self.graph_store.add_edge(src, tgt, rel)
        except Exception as e:
            logger.warning("failed_to_extract_relationships", error=str(e), response=clean_text)
