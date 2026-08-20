from __future__ import annotations

import json
import re

import structlog

from data_engineering_copilot.domain.protocols import LLMClientProtocol
from data_engineering_copilot.infrastructure.graph_store import GraphStore

logger = structlog.get_logger(__name__)


class GraphTraversalService:
    def __init__(self, llm_client: LLMClientProtocol, graph_store: GraphStore) -> None:
        self.llm_client = llm_client
        self.graph_store = graph_store

    async def get_topological_context(self, query: str) -> str:
        """Extract entities from query, traverse GraphStore, and return topological context."""
        prompt = f"""Extract the main technical terms, tools, or concepts from the query below as a JSON array of strings.
Query: "{query}"

Output ONLY a JSON array of strings, e.g.: ["spark streaming", "migration framework"]
Do NOT output markdown fences or code blocks. Just JSON.
"""
        clean_text = ""
        try:
            response = await self.llm_client.generate(prompt)
            clean_text = response.strip()
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", clean_text, re.DOTALL)
            if m:
                clean_text = m.group(1).strip()
            entities = json.loads(clean_text)

            triplets: list[tuple[str, str, str]] = []
            if isinstance(entities, list):
                for ent in entities:
                    triplets.extend(self.graph_store.get_neighbors(str(ent)))

            if not triplets:
                return ""

            lines = ["Topological & Entity Relationships found in Knowledge Graph:"]
            seen = set()
            for src, rel, tgt in triplets:
                line = f"- ({src}) --[{rel}]--> ({tgt})"
                if line not in seen:
                    seen.add(line)
                    lines.append(line)
            return "\n".join(lines)
        except Exception as e:
            logger.warning("failed_to_traverse_graph", error=str(e), response=clean_text)
            return ""
