"""RAGAS evaluation integration — CI-gated RAG quality metrics.

Lazily imports ``ragas`` and ``datasets`` so the system works without them.
Metrics: context_recall, context_precision, faithfulness, answer_relevancy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RagasEvalResult:
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevancy: float
    overall: float


class RagasEvaluator:
    """Wraps the RAGAS framework for production evaluation.

    Requires ``ragas`` and ``datasets`` packages. Lazily imported so the
    system works without them.
    """

    def __init__(self) -> None:
        self._evaluate = None
        self._metrics = None
        self._dataset_class = None

    def _lazy_init(self) -> bool:
        if self._evaluate is not None:
            return True
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            self._evaluate = evaluate
            self._metrics = [
                context_recall,
                context_precision,
                faithfulness,
                answer_relevancy,
            ]
            return True
        except ImportError:
            logger.debug("ragas package not installed — evaluation unavailable")
            return False

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truth: list[str] | None = None,
    ) -> RagasEvalResult | None:
        """Run RAGAS evaluation on a batch of Q&A pairs.

        Returns ``RagasEvalResult`` or ``None`` if ragas unavailable.
        """
        if not self._lazy_init():
            return None

        from datasets import Dataset

        data: dict[str, list] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truth:
            data["ground_truth"] = ground_truth

        dataset = Dataset.from_dict(data)
        result = self._evaluate(dataset=dataset, metrics=self._metrics)

        # RAGAS returns a dict-like; extract scores with safe defaults
        recall = float(result.get("context_recall", 0))
        precision = float(result.get("context_precision", 0))
        faithful = float(result.get("faithfulness", 0))
        relevancy = float(result.get("answer_relevancy", 0))

        return RagasEvalResult(
            context_recall=recall,
            context_precision=precision,
            faithfulness=faithful,
            answer_relevancy=relevancy,
            overall=recall * 0.3 + faithful * 0.4 + relevancy * 0.3,
        )
