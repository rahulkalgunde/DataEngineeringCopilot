"""Langfuse dataset and experiment management.

Provides functions to upload evaluation datasets to Langfuse and run experiments.
Targets the Langfuse v4 SDK surface (see ``docs/langfuse-v4-sdk-surface.md``).
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


def run_experiment(
    experiment_name: str,
    dataset_name: str,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
) -> dict[str, Any] | None:
    """Run A/B experiment comparing two configurations.

    The full experiment runner is implemented in Phase 6 (``dataset.run_experiment``
    from the Langfuse v4 SDK). Until then this raises a clear error so callers do
    not silently do nothing.

    Args:
        experiment_name: Name for the experiment
        dataset_name: Name of the dataset to run against
        config_a: First configuration to test
        config_b: Second configuration to test

    Returns:
        Experiment results or None if failed
    """
    client = get_langfuse_client()
    if client is None:
        logger.warning("Langfuse client not available, cannot run experiment")
        return None

    raise NotImplementedError(
        "Experiments require Phase 6 (langfuse dataset.run_experiment). Use the Langfuse UI for now."
    )


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
