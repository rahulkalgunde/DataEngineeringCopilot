from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import structlog

from data_engineering_copilot.domain.protocols import LLMClientProtocol

logger = structlog.get_logger(__name__)


@dataclass
class QueryStep:
    step_id: int
    query: str
    depends_on: list[int]
    status: str = "pending"  # pending, completed, failed
    result_summary: str = ""


@dataclass
class QueryPlan:
    steps: list[QueryStep]
    is_multi_hop: bool


class MultiHopDecomposer:
    def __init__(self, llm_client: LLMClientProtocol) -> None:
        self.llm_client = llm_client

    async def plan_query(self, query: str, history: str = "") -> QueryPlan:
        """Analyze query and return a sequence of sub-queries if multi-hop is needed."""
        prompt = f"""You are an expert query decomposition planner.
Analyze the following user query and decide if it is a complex, multi-part query that needs to be broken down into multiple sequential or parallel retrieval steps (e.g. comparing Spark streaming in version 3.2 vs custom migration framework).

User Query: "{query}"
History: "{history}"

Output ONLY a JSON object of this format:
{{
  "is_multi_hop": true,
  "steps": [
    {{
      "step_id": 1,
      "query": "search query for step 1",
      "depends_on": []
    }},
    {{
      "step_id": 2,
      "query": "search query for step 2 comparing/using results from step 1",
      "depends_on": [1]
    }}
  ]
}}
If it is a simple query that can be resolved in a single search, return:
{{
  "is_multi_hop": false,
  "steps": []
}}
Do NOT output any markdown fences, preamble, or comments. Just the raw JSON.
"""
        clean_text = ""
        try:
            response = await self.llm_client.generate(prompt)
            clean_text = response.strip()
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", clean_text, re.DOTALL)
            if m:
                clean_text = m.group(1).strip()
            data = json.loads(clean_text)
            is_multi_hop = bool(data.get("is_multi_hop", False))
            steps_data = data.get("steps", [])
            steps = []
            for s in steps_data:
                steps.append(
                    QueryStep(
                        step_id=int(s["step_id"]),
                        query=str(s["query"]),
                        depends_on=[int(x) for x in s.get("depends_on", [])],
                    )
                )
            return QueryPlan(steps=steps, is_multi_hop=is_multi_hop)
        except Exception as e:
            logger.warning("failed_to_decompose_query", error=str(e), response=clean_text)
            return QueryPlan(steps=[], is_multi_hop=False)

    async def execute_step(self, step: QueryStep, previous_results: dict[int, str], rag_service: Any) -> str:
        """Execute retrieval and summarize evidence for a step."""
        query = step.query
        if step.depends_on:
            context_str = "\n".join(
                [f"Step {dep_id} summary: {previous_results.get(dep_id, '')}" for dep_id in step.depends_on]
            )
            refine_prompt = f"""Given the previous steps context:
{context_str}

Refine the query for this step: "{step.query}"
Return ONLY the refined query text, no other text or explanation.
"""
            try:
                query = await self.llm_client.generate(refine_prompt)
                query = query.strip()
            except Exception as e:
                logger.warning("failed_to_refine_step_query", error=str(e))

        try:
            q_emb = await rag_service.embedder.embed_query(query)
            profile = rag_service._rrf_profile_for(query)
            results = await rag_service.vector_store.query(
                q_emb,
                top_k=rag_service.config.retrieval_top_k,
                query_text=query,
                rrf_profile=profile,
            )

            context = "\n\n".join([r.chunk.text for r in results[:3]])
            summary_prompt = f"""Given the context retrieved for the query: "{query}":
{context}

Provide a concise summary answering this sub-query based ONLY on the context.
If no context is found, say "No relevant info found."
"""
            summary = await self.llm_client.generate(summary_prompt)
            return summary.strip()
        except Exception as e:
            logger.warning("failed_to_execute_step", error=str(e))
            return "Execution failed."
