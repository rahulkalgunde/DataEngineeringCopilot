"""Isolated reranker evaluation harness.

Runs reranker against a golden dataset with frozen candidate pools to produce
nDCG@K, MRR, Precision@K metrics. No pipeline coupling — standalone eval.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.evaluation.rerank_metrics import evaluate_reranker

if TYPE_CHECKING:
    from data_engineering_copilot.services.async_rag import AsyncRagService

logger = logging.getLogger(__name__)


@runtime_checkable
class RerankEvalServiceProtocol(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class RerankEvalServiceAdapter:
    """Adapts AsyncRagService to RerankEvalServiceProtocol for eval harness."""

    def __init__(self, rag_service: AsyncRagService) -> None:
        self._rag = rag_service

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        q_emb = await self._rag.embedder.embed_query(query)
        return await self._rag.vector_store.query(
            q_emb,
            top_k=top_k,
            query_text=query,
        )

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if self._rag.reranker is None:
            return chunks[:top_k]
        await self._rag._ensure_reranker_ready()
        if not self._rag.reranker.is_available():
            return chunks[:top_k]
        return await self._rag.reranker.rerank(query, chunks, top_k=top_k)


@dataclass
class RerankEvalRow:
    query: str
    source_urls: list[str]
    relevance_labels: list[int]


@dataclass
class RerankEvalResult:
    query: str
    pre_rerank_relevance: list[int]
    post_rerank_relevance: list[int]
    metrics: dict[str, float]


@dataclass
class RerankEvalReport:
    results: list[RerankEvalResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Queries evaluated: {len(self.results)}"]
        for k, v in self.aggregate.items():
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)


def load_rerank_eval_dataset(path: pathlib.Path) -> list[RerankEvalRow]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(
                RerankEvalRow(
                    query=data["query"],
                    source_urls=data["source_urls"],
                    relevance_labels=data.get("relevance_labels", []),
                )
            )
    return rows


def _urls_to_relevance(urls: list[str], labels: list[int], ranked_urls: list[str]) -> list[int]:
    """Map relevance labels from source_urls order to ranked_urls order."""
    url_to_label = dict(zip(urls, labels, strict=False))
    return [url_to_label.get(url, 0) for url in ranked_urls]


def save_candidate_pool(path: pathlib.Path, pool: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pool, f, indent=2)


def load_candidate_pool(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


async def run_rerank_eval(
    dataset: list[RerankEvalRow],
    service: RerankEvalServiceProtocol,
    k: int = 10,
    candidate_pool_path: pathlib.Path | None = None,
) -> RerankEvalReport:
    """Run evaluation: for each query, retrieve, rerank, compute metrics.

    If candidate_pool_path is provided and exists, load frozen pools.
    Otherwise, run retrieval and save pools for reproducibility.
    """
    report = RerankEvalReport()
    frozen_pools: dict[str, list[dict]] = {}

    if candidate_pool_path and candidate_pool_path.exists():
        frozen_pools = load_candidate_pool(candidate_pool_path)
        logger.info("Loaded frozen candidate pool from %s (%d queries)", candidate_pool_path, len(frozen_pools))

    new_pools: dict[str, list[dict]] = {}

    for row in dataset:
        if row.query in frozen_pools:
            retrieved_data = frozen_pools[row.query]
            retrieved = []
            for r in retrieved_data:
                chunk_data = dict(r["chunk"])
                if "heading_path" in chunk_data and isinstance(chunk_data["heading_path"], list):
                    chunk_data["heading_path"] = tuple(chunk_data["heading_path"])
                retrieved.append(
                    RetrievedChunk(
                        chunk=DocumentChunk(**chunk_data),
                        distance=r["distance"],
                        confidence=r["confidence"],
                    )
                )
        else:
            retrieved = await service.retrieve(row.query, top_k=k * 4)
            new_pools[row.query] = [
                {
                    "chunk": asdict(r.chunk),
                    "distance": r.distance,
                    "confidence": r.confidence,
                }
                for r in retrieved
            ]

        reranked = await service.rerank(row.query, retrieved, top_k=k)

        pre_urls = [r.chunk.url for r in retrieved[:k]]
        post_urls = [r.chunk.url for r in reranked]

        pre_relevance = _urls_to_relevance(row.source_urls, row.relevance_labels, pre_urls)
        post_relevance = _urls_to_relevance(row.source_urls, row.relevance_labels, post_urls)

        metrics = evaluate_reranker(post_relevance, pre_relevance, k)

        report.results.append(
            RerankEvalResult(
                query=row.query,
                pre_rerank_relevance=pre_relevance,
                post_rerank_relevance=post_relevance,
                metrics=metrics,
            )
        )

    if candidate_pool_path and new_pools:
        merged = {**frozen_pools, **new_pools}
        save_candidate_pool(candidate_pool_path, merged)
        logger.info("Saved %d new queries to candidate pool at %s", len(new_pools), candidate_pool_path)

    if report.results:
        metric_keys = report.results[0].metrics.keys()
        report.aggregate = {
            key: sum(r.metrics[key] for r in report.results) / len(report.results) for key in metric_keys
        }

    return report
