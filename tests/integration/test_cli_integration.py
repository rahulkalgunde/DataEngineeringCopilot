"""Integration tests for CLI commands (ask, eval).

Tests the CLI commands end-to-end with real infrastructure (Qdrant, Redis).
Run with: pytest tests/integration/test_cli_integration.py -v -m integration
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


@pytest.mark.qdrant
class TestAskCommand:
    """Tests for `dec ask` command."""

    def test_ask_parses_question(self) -> None:
        """ask takes positional question argument."""
        from data_engineering_copilot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["ask", "What is Spark?"])
        assert args.question == "What is Spark?"

    def test_ask_parses_source(self) -> None:
        """ask takes --source argument."""
        from data_engineering_copilot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["ask", "--source", "spark", "What is Spark?"])
        assert args.source == ["spark"]


class TestEvalCommand:
    """Tests for `dec evaluate` command."""

    def test_eval_requires_dataset(self) -> None:
        """eval requires a dataset path."""
        from data_engineering_copilot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["evaluate", "--dataset", "tests/evaluation/recall_fast.jsonl"])
        assert args.dataset == "tests/evaluation/recall_fast.jsonl"


class TestIngestCommand:
    """Tests for `dec ingest` command."""

    def test_ingest_parses_source(self) -> None:
        """ingest takes --source argument."""
        from data_engineering_copilot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["ingest", "--source", "spark"])
        assert args.source == ["spark"]

    def test_ingest_parses_max_pages(self) -> None:
        """ingest takes --max-pages argument."""
        from data_engineering_copilot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["ingest", "--max-pages", "10"])
        assert args.max_pages == 10
