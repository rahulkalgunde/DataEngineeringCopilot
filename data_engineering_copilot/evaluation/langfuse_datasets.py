"""Langfuse dataset and experiment management.

Provides functions to upload evaluation datasets to Langfuse, run RAG
experiments via the v4 ``dataset.run_experiment`` API, score experiment runs
with offline RAGAS metrics, and collect low-confidence production answers into
a review dataset. Targets the Langfuse v4 SDK surface (see
``docs/langfuse-v4-sdk-surface.md``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_langfuse_client():
    """Get Langfuse client instance."""
    try:
        from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

        return get_langfuse_instance()
    except Exception as exc:
        logger.warning("Failed to get Langfuse client: %s", exc)
        return None


def upload_evaluation_dataset(dataset_path: str, dataset_name: str) -> bool:
    """Upload evaluation dataset to Langfuse.

    Args:
        dataset_path: Path to JSONL dataset file
        dataset_name: Name for the dataset in Langfuse

    Returns:
        True if successful, False otherwise
    """
    client = get_langfuse_client()
    if client is None:
        logger.warning("Langfuse client not available, cannot upload dataset")
        return False

    try:
        # Load JSONL dataset
        examples = []
        with open(dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if not examples:
            logger.warning("No examples found in dataset %s", dataset_path)
            return False

        return upload_evaluation_dataset_rows(
            dataset_name=dataset_name,
            items=[
                {
                    "input": ex.get("input", {}),
                    "expected_output": ex.get("expected_output", {}),
                    "metadata": ex.get("metadata", {}),
                }
                for ex in examples
            ],
            description=f"Evaluation dataset uploaded from {dataset_path}",
        )

    except Exception as exc:
        logger.error("Failed to upload dataset to Langfuse: %s", exc)
        return False


def upload_evaluation_dataset_rows(
    dataset_name: str,
    items: list[dict[str, Any]],
    description: str | None = None,
) -> bool:
    """Create (or reuse) a Langfuse dataset and add items.

    Uses the v4 top-level ``create_dataset`` / ``create_dataset_item`` API.

    Args:
        dataset_name: Name for the dataset in Langfuse
        items: List of dicts with ``input``, ``expected_output`` (optional),
            ``metadata`` (optional)
        description: Optional dataset description

    Returns:
        True if successful, False otherwise
    """
    client = get_langfuse_client()
    if client is None:
        logger.warning("Langfuse client not available, cannot upload dataset")
        return False

    try:
        inner = client._client
        if not hasattr(inner, "create_dataset"):
            logger.warning("Langfuse client does not support create_dataset")
            return False

        # Reuse an existing dataset if present (create_dataset is not idempotent).
        try:
            inner.create_dataset(
                name=dataset_name,
                description=description or f"Evaluation dataset {dataset_name}",
            )
        except Exception as exc:
            logger.info("Dataset '%s' may already exist (%s); adding items to it", dataset_name, exc)

        if not hasattr(inner, "create_dataset_item"):
            logger.warning("Langfuse client does not support create_dataset_item")
            return False

        for i, item in enumerate(items):
            inner.create_dataset_item(
                dataset_name=dataset_name,
                input=item.get("input", {}),
                expected_output=item.get("expected_output"),
                metadata=item.get("metadata", {"index": i}),
            )

        logger.info("Uploaded %d examples to Langfuse dataset '%s'", len(items), dataset_name)
        return True

    except Exception as exc:
        logger.error("Failed to upload dataset to Langfuse: %s", exc)
        return False


def create_review_item(trace_id: str, question: str, answer: str) -> bool:
    """Add a low-confidence production answer to the ``low-confidence-review`` dataset.

    The dataset item links back to the source trace so reviewers can inspect the
    full retrieval context in the Langfuse UI.

    Args:
        trace_id: Langfuse trace id of the low-confidence answer
        question: The user's question
        answer: The produced (low-confidence) answer

    Returns:
        True if the item was created, False otherwise
    """
    client = get_langfuse_client()
    if client is None:
        logger.debug("Langfuse client not available, cannot create review item")
        return False

    try:
        inner = client._client
        if not hasattr(inner, "create_dataset"):
            logger.warning("Langfuse client does not support create_dataset")
            return False

        try:
            inner.create_dataset(
                name="low-confidence-review",
                description="Production answers below the confidence threshold, queued for manual review.",
            )
        except Exception as exc:
            logger.debug("Dataset 'low-confidence-review' may already exist (%s); adding item to it", exc)

        if not hasattr(inner, "create_dataset_item"):
            logger.warning("Langfuse client does not support create_dataset_item")
            return False

        inner.create_dataset_item(
            dataset_name="low-confidence-review",
            input={"query": question},
            expected_output={"answer": answer},
            metadata={"source_trace_id": trace_id},
            source_trace_id=trace_id,
        )
        logger.info("Added low-confidence answer to 'low-confidence-review' dataset (trace %s)", trace_id)
        return True

    except Exception as exc:
        logger.warning("Failed to create low-confidence review item for trace %s: %s", trace_id, exc)
        return False


def list_review_items(limit: int = 100) -> list[dict[str, object]]:
    """Return queued items from the OSS-compatible review dataset.

    The Langfuse annotation-queue API is organization-scoped and unavailable
    on this OSS deployment. Dataset items are the supported review queue.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse is unavailable; cannot list review items")

    dataset = client.get_dataset("low-confidence-review")
    items: list[dict[str, object]] = []
    for item in dataset.items[:limit]:
        expected_output = item.expected_output if isinstance(item.expected_output, dict) else {}
        items.append(
            {
                "item_id": item.id,
                "question": (item.input or {}).get("query"),
                "answer": expected_output.get("answer"),
                "source_trace_id": item.source_trace_id or (item.metadata or {}).get("source_trace_id"),
                "status": getattr(item.status, "value", item.status),
                "created_at": item.created_at.isoformat() if item.created_at else None,
            }
        )
    return items


def run_rag_experiment(
    dataset_name: str,
    experiment_name: str,
    source_filter: list[str] | None = None,
    description: str | None = None,
    max_concurrency: int = 2,
):
    """Run the RAG pipeline over every item in a Langfuse dataset as an experiment.

    Uses the v4 ``dataset.run_experiment`` API (SYNC — do not wrap in
    ``asyncio.run``; the task function itself is async and handled internally).
    Each dataset item's ``input["query"]`` is answered by the production RAG
    service; a term-overlap ``faithfulness`` evaluator scores the output against
    ``expected_output["answer"]``. After the run, offline RAGAS metrics are
    scored onto each item trace (``ragas_*``).

    Args:
        dataset_name: Existing Langfuse dataset to run against
        experiment_name: Name for this experiment run
        source_filter: Optional list of source names to restrict retrieval
        description: Optional experiment description
        max_concurrency: Max concurrent RAG task executions

    Returns:
        ``ExperimentResult`` from the v4 SDK, or ``None`` if Langfuse is unavailable
    """
    client = get_langfuse_client()
    if client is None:
        logger.warning("Langfuse client not available, cannot run experiment")
        return None

    try:
        dataset = client.get_dataset(dataset_name)
    except Exception as exc:
        logger.warning("Failed to load dataset '%s': %s", dataset_name, exc)
        return None

    service_holder: dict[str, Any] = {}

    async def _task(*, item, **kwargs):
        svc = service_holder.get("service")
        if svc is None:
            from data_engineering_copilot.factory import build_rag_service

            svc = build_rag_service()
            service_holder["service"] = svc
        query = item.input.get("query", item.input) if isinstance(item.input, dict) else item.input
        ans = await svc.answer(query, source_filter=source_filter)
        return ans.text

    def _faithfulness_eval(*, input, output, expected_output=None, **kwargs):
        from langfuse import Evaluation

        gt = None
        if isinstance(expected_output, dict):
            gt = expected_output.get("answer") or expected_output.get("expected_output")
        elif expected_output is not None:
            gt = expected_output
        score = 1.0 if gt and str(gt).lower()[:20] in str(output).lower() else 0.0
        return Evaluation(name="faithfulness", value=score, comment="term-overlap")

    try:
        result = dataset.run_experiment(
            name=experiment_name,
            description=description or f"RAG experiment on {dataset_name}",
            task=_task,
            evaluators=[_faithfulness_eval],
            max_concurrency=max_concurrency,
        )
    except Exception as exc:
        logger.warning("Failed to run experiment '%s' on dataset '%s': %s", experiment_name, dataset_name, exc)
        return None

    ragas_report = score_experiment_with_ragas(result.item_results)
    if ragas_report is not None:
        logger.info(
            "Recorded offline RAGAS scores for experiment '%s' "
            "(context_recall=%.3f context_precision=%.3f faithfulness=%.3f answer_relevancy=%.3f)",
            experiment_name,
            ragas_report.context_recall,
            ragas_report.context_precision,
            ragas_report.faithfulness,
            ragas_report.answer_relevancy,
        )
    return result


def score_experiment_with_ragas(item_results: list[Any]):
    """Score an experiment run's item traces with offline RAGAS metrics.

    Collects questions/answers/contexts/ground-truth from the experiment item
    results, runs ``RagasEvaluator``, then pushes the four aggregate metrics
    (``ragas_context_recall`` etc.) onto each item's Langfuse trace.

    Args:
        item_results: ``ExperimentItemResult`` list from ``run_experiment``

    Returns:
        ``RagasEvalResult`` or ``None`` if RAGAS is unavailable
    """
    if not item_results:
        logger.debug("No item results — skipping offline RAGAS scoring")
        return None

    from data_engineering_copilot.services.ragas_evaluation import RagasEvaluator

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truth: list[str] = []

    for ir in item_results:
        item = getattr(ir, "item", None)
        input_ = getattr(item, "input", None)
        questions.append(input_.get("query", input_) if isinstance(input_, dict) else (input_ or ""))
        answers.append(str(getattr(ir, "output", "") or ""))
        contexts.append((getattr(item, "metadata", None) or {}).get("contexts", []) or [])
        gt = getattr(item, "expected_output", None)
        ground_truth.append(gt.get("answer", "") if isinstance(gt, dict) else (gt or ""))

    report = RagasEvaluator().evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truth=ground_truth if any(ground_truth) else None,
    )
    if report is None:
        logger.debug("RAGAS unavailable — skipping offline experiment scores")
        return None

    client = get_langfuse_client()
    if client is not None:
        metrics = [
            ("context_recall", report.context_recall),
            ("context_precision", report.context_precision),
            ("faithfulness", report.faithfulness),
            ("answer_relevancy", report.answer_relevancy),
        ]
        for ir in item_results:
            trace_id = getattr(ir, "trace_id", None)
            if not trace_id:
                continue
            for metric, value in metrics:
                try:
                    client.score(trace_id=trace_id, name=f"ragas_{metric}", value=value, comment="offline-RAGAS")
                except Exception as exc:
                    logger.warning("Failed to score ragas_%s on trace %s: %s", metric, trace_id, exc)
    return report


def get_experiment_results(experiment_id: str) -> dict[str, Any] | None:
    """Get results from a Langfuse experiment.

    Args:
        experiment_id: ID of the experiment

    Returns:
        Experiment results or None if failed
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        # This would need to be implemented based on Langfuse's API
        # For now, return a placeholder
        logger.info("Getting results for experiment %s", experiment_id)
        return {"experiment_id": experiment_id, "status": "placeholder"}

    except Exception as exc:
        logger.error("Failed to get experiment results: %s", exc)
        return None
