"""Tests for evaluation/langfuse_datasets.py."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.evaluation.langfuse_datasets import (
    create_review_item,
    get_experiment_results,
    get_langfuse_client,
    list_review_items,
    upload_evaluation_dataset,
    upload_evaluation_dataset_rows,
)


class TestGetLangfuseClient:
    def test_returns_none_on_failure(self) -> None:
        with patch(
            "data_engineering_copilot.observability.langfuse_client.get_langfuse_instance",
            side_effect=Exception("not configured"),
        ):
            assert get_langfuse_client() is None


class TestUploadEvaluationDataset:
    def test_returns_false_when_no_client(self) -> None:
        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=None,
        ):
            assert upload_evaluation_dataset("path.jsonl", "name") is False

    def test_returns_false_on_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name

        mock_client = MagicMock()
        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=mock_client,
        ):
            assert upload_evaluation_dataset(path, "name") is False

    def test_uploads_valid_file(self) -> None:
        data = [{"input": {"query": "q"}, "expected_output": {"answer": "a"}}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
            f.flush()
            path = f.name

        mock_client = MagicMock()
        mock_inner = MagicMock()
        mock_client._client = mock_inner

        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=mock_client,
        ):
            assert upload_evaluation_dataset(path, "name") is True
            mock_inner.create_dataset.assert_called_once()
            mock_inner.create_dataset_item.assert_called_once()


class TestUploadEvaluationDatasetRows:
    def test_returns_false_when_no_client(self) -> None:
        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=None,
        ):
            assert upload_evaluation_dataset_rows("name", []) is False

    def test_returns_false_when_no_create_dataset(self) -> None:
        mock_client = MagicMock()
        mock_inner = MagicMock(spec=[])
        mock_client._client = mock_inner

        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=mock_client,
        ):
            assert upload_evaluation_dataset_rows("name", [{"input": {}}]) is False

    def test_uploads_items(self) -> None:
        mock_client = MagicMock()
        mock_inner = MagicMock()
        mock_client._client = mock_inner

        items = [
            {"input": {"query": "q1"}, "expected_output": {"answer": "a1"}},
            {"input": {"query": "q2"}},
        ]

        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=mock_client,
        ):
            assert upload_evaluation_dataset_rows("name", items) is True
            assert mock_inner.create_dataset_item.call_count == 2


class TestCreateReviewItem:
    def test_returns_false_when_no_client(self) -> None:
        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=None,
        ):
            assert create_review_item("trace123", "question", "answer") is False

    def test_creates_review_item(self) -> None:
        mock_client = MagicMock()
        mock_inner = MagicMock()
        mock_client._client = mock_inner

        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=mock_client,
        ):
            assert create_review_item("trace123", "question", "answer") is True
            mock_inner.create_dataset_item.assert_called_once()


class TestListReviewItems:
    def test_raises_on_invalid_limit(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            list_review_items(limit=0)

    def test_raises_when_no_client(self) -> None:
        with (
            patch(
                "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="unavailable"),
        ):
            list_review_items()

    def test_lists_items(self) -> None:
        mock_item = MagicMock()
        mock_item.id = "item1"
        mock_item.input = {"query": "q"}
        mock_item.expected_output = {"answer": "a"}
        mock_item.source_trace_id = "trace1"
        mock_item.status = MagicMock(value="pending")
        mock_item.created_at = None
        mock_item.metadata = None

        mock_dataset = MagicMock()
        mock_dataset.items = [mock_item]

        mock_client = MagicMock()
        mock_client.get_dataset.return_value = mock_dataset

        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=mock_client,
        ):
            items = list_review_items(limit=10)
            assert len(items) == 1
            assert items[0]["item_id"] == "item1"
            assert items[0]["question"] == "q"
            assert items[0]["answer"] == "a"


class TestGetExperimentResults:
    def test_returns_none_when_no_client(self) -> None:
        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=None,
        ):
            assert get_experiment_results("exp1") is None

    def test_returns_placeholder(self) -> None:
        mock_client = MagicMock()
        with patch(
            "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
            return_value=mock_client,
        ):
            result = get_experiment_results("exp1")
            assert result is not None
            assert result["experiment_id"] == "exp1"
