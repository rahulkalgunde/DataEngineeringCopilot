"""Isolated context assembly evaluation harness.

Runs the assembly layer against frozen candidate pools to produce
duplicate-rate, source-coverage, compression-ratio, and needle-loss metrics.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass

from data_engineering_copilot.evaluation.assembly_metrics import (
    AssemblyEvalReport,
    context_compression_ratio,
    duplicate_candidate_rate,
    needle_loss_rate,
    source_coverage_rate,
)

logger = logging.getLogger(__name__)


@dataclass
class AssemblyEvalRow:
    query: str
    source_urls: list[str]
    gold_facts: list[str]


def load_assembly_eval_dataset(path: pathlib.Path) -> list[AssemblyEvalRow]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(
                AssemblyEvalRow(
                    query=data["query"],
                    source_urls=data["source_urls"],
                    gold_facts=data.get("gold_facts", []),
                )
            )
    return rows


def run_assembly_eval(
    dataset: list[AssemblyEvalRow],
    rag_service: object,
    k: int = 20,
) -> list[AssemblyEvalReport]:
    """Run evaluation: for each query, retrieve, assemble, compute metrics."""
    from data_engineering_copilot.services.context_assembler import ContextAssembler

    reports = []
    for row in dataset:
        retrieved = rag_service.retrieve(row.query, top_k=k)  # type: ignore[union-attr]
        assembler = ContextAssembler(max_context_chars=16000)
        context_str, source_names, _ = assembler.assemble(retrieved)

        total_source_urls = len(set(r.chunk.url for r in retrieved))
        initial_chars = sum(len(r.chunk.text) for r in retrieved)

        report = AssemblyEvalReport(
            duplicate_rate=duplicate_candidate_rate(context_str),
            source_coverage=source_coverage_rate(source_names, total_source_urls),
            compression_ratio=context_compression_ratio(len(context_str), initial_chars),
            needle_loss=needle_loss_rate(context_str, row.gold_facts),
        )
        reports.append(report)

    return reports
