"""Prompt augmentation evaluation harness.

Runs frozen-input evaluation: swaps prompt templates on identical
query/context pairs and computes format, citation, injection, and
zero-context metrics without a live LLM. Supports optional LLM generation
mode for actual output quality evaluation.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field

from data_engineering_copilot.evaluation.prompt_aug_metrics import (
    PromptAugMetrics,
    compute_citation_precision,
    compute_citation_recall,
    compute_format_compliance,
    compute_injection_defense_rate,
    compute_zero_context_fallback_accuracy,
)

logger = logging.getLogger(__name__)


@dataclass
class PromptAugEvalRow:
    query: str
    context: str
    intent: str
    expected_citations: list[str]
    expected_format: str
    has_sufficient_context: bool
    injection_payload: str | None


@dataclass
class PromptAugEvalReport:
    metrics: PromptAugMetrics
    total_samples: int
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return f"Prompt Aug Eval — {self.total_samples} samples\n{self.metrics.summary()}"


def load_dataset(path: pathlib.Path) -> list[PromptAugEvalRow]:
    rows: list[PromptAugEvalRow] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(
                PromptAugEvalRow(
                    query=data["query"],
                    context=data["context"],
                    intent=data["intent"],
                    expected_citations=data.get("expected_citations", []),
                    expected_format=data.get("expected_format", "json"),
                    has_sufficient_context=data.get("has_sufficient_context", True),
                    injection_payload=data.get("injection_payload"),
                )
            )
    return rows


def run_prompt_aug_eval(dataset_path: pathlib.Path) -> PromptAugEvalReport:
    """Run isolated eval: build prompts for each frozen row, compute metrics.

    This is hermetic — it validates prompt construction and metric computation
    without calling a live LLM. LLM generation is a future extension.
    """
    from data_engineering_copilot.services.prompt_builder import PromptBuilder

    dataset = load_dataset(dataset_path)
    builder = PromptBuilder()

    outputs: list[str] = []
    details: list[dict] = []

    for row in dataset:
        prompt = builder.build_rag_prompt(
            context=row.context or "No relevant documents found.",
            question=row.query,
            intent=row.intent,
        )
        outputs.append(prompt)
        details.append(
            {
                "query": row.query,
                "intent": row.intent,
                "prompt_length": len(prompt),
                "expected_citations": row.expected_citations,
                "expected_format": row.expected_format,
                "has_sufficient_context": row.has_sufficient_context,
                "injection_payload": row.injection_payload,
            }
        )

    metrics = PromptAugMetrics(
        format_compliance_rate=compute_format_compliance(outputs, [r.expected_format for r in dataset]),
        citation_precision=compute_citation_precision(outputs, [r.expected_citations for r in dataset]),
        citation_recall=compute_citation_recall(outputs, [r.expected_citations for r in dataset]),
        injection_defense_rate=compute_injection_defense_rate(outputs, [r.injection_payload for r in dataset]),
        zero_context_fallback_accuracy=compute_zero_context_fallback_accuracy(
            outputs, [r.has_sufficient_context for r in dataset]
        ),
    )

    return PromptAugEvalReport(
        metrics=metrics,
        total_samples=len(dataset),
        details=details,
    )


async def run_prompt_aug_eval_llm(
    dataset_path: pathlib.Path,
    provider: str = "ollama",
) -> PromptAugEvalReport:
    """Run eval with LLM generation: build prompts, call LLM, compute metrics on actual outputs.

    This calls a live LLM (via provider fallback chain) to generate actual responses
    and then computes quality metrics on the LLM outputs.
    """
    from data_engineering_copilot.config.settings import settings
    from data_engineering_copilot.factory import build_llm_fallback_chain
    from data_engineering_copilot.services.prompt_builder import PromptBuilder

    # Use global settings; provider argument can override fallback order
    settings.llm_fallback_order = [provider]
    llm_client = build_llm_fallback_chain("answer", settings)

    dataset = load_dataset(dataset_path)
    builder = PromptBuilder()

    outputs: list[str] = []
    details: list[dict] = []

    for row in dataset:
        prompt = builder.build_rag_prompt(
            context=row.context or "No relevant documents found.",
            question=row.query,
            intent=row.intent,
        )
        # Call LLM with the prompt
        try:
            llm_output = await llm_client.generate(prompt)
        except Exception as e:
            logger.warning("LLM generation failed for query '%s': %s", row.query, e)
            llm_output = f"ERROR: {e}"

        outputs.append(llm_output)
        details.append(
            {
                "query": row.query,
                "intent": row.intent,
                "prompt_length": len(prompt),
                "output_length": len(llm_output),
                "expected_citations": row.expected_citations,
                "expected_format": row.expected_format,
                "has_sufficient_context": row.has_sufficient_context,
                "injection_payload": row.injection_payload,
            }
        )

    metrics = PromptAugMetrics(
        format_compliance_rate=compute_format_compliance(outputs, [r.expected_format for r in dataset]),
        citation_precision=compute_citation_precision(outputs, [r.expected_citations for r in dataset]),
        citation_recall=compute_citation_recall(outputs, [r.expected_citations for r in dataset]),
        injection_defense_rate=compute_injection_defense_rate(outputs, [r.injection_payload for r in dataset]),
        zero_context_fallback_accuracy=compute_zero_context_fallback_accuracy(
            outputs, [r.has_sufficient_context for r in dataset]
        ),
    )

    return PromptAugEvalReport(
        metrics=metrics,
        total_samples=len(dataset),
        details=details,
    )
