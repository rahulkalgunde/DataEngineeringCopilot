"""Langfuse dataset and experiment management.

Provides functions to upload evaluation datasets to Langfuse and run experiments.
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

        # Create dataset using the underlying Langfuse client
        if hasattr(client._client, "create_dataset"):
            client._client.create_dataset(
                name=dataset_name,
                description=f"Evaluation dataset uploaded from {dataset_path}",
            )

            # Add examples to dataset
            for i, example in enumerate(examples):
                client._client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=example.get("input", {}),
                    expected_output=example.get("expected_output", {}),
                    metadata=example.get("metadata", {"index": i}),
                )

            logger.info("Uploaded %d examples to Langfuse dataset '%s'", len(examples), dataset_name)
            return True
        else:
            logger.warning("Langfuse client does not support create_dataset")
            return False

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

    try:
        # Create experiment
        if hasattr(client._client, "create_experiment"):
            experiment = client._client.create_experiment(  # type: ignore[attr-defined]  # guarded by hasattr above; Langfuse stub lacks it
                name=experiment_name,
                dataset_name=dataset_name,
            )

            # Note: Actual experiment execution would require running the RAG pipeline
            # with both configurations and comparing results. This is a placeholder.
            logger.info("Created experiment '%s' on dataset '%s'", experiment_name, dataset_name)

            return {
                "experiment_id": getattr(experiment, "id", None),
                "name": experiment_name,
                "dataset": dataset_name,
                "status": "created",
            }
        else:
            logger.warning("Langfuse client does not support create_experiment")
            return None

    except Exception as exc:
        logger.error("Failed to create experiment in Langfuse: %s", exc)
        return None


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
