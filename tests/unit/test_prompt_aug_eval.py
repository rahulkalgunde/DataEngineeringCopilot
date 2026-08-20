"""Tests for prompt augmentation evaluation harness."""

from __future__ import annotations

import json
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from data_engineering_copilot.evaluation.prompt_aug_eval import (
    load_dataset,
    run_prompt_aug_eval,
)

pytestmark = pytest.mark.unit


class TestLoadDataset:
    def test_load_sample(self, tmp_path: pathlib.Path):
        data = tmp_path / "test.jsonl"
        row = {
            "query": "What is Spark?",
            "context": '<context_doc id="1">Docs</context_doc>',
            "intent": "factual",
            "expected_citations": ["1"],
            "expected_format": "json",
            "has_sufficient_context": True,
            "injection_payload": None,
        }
        data.write_text(json.dumps(row) + "\n")
        rows = load_dataset(data)
        assert len(rows) == 1
        assert rows[0].query == "What is Spark?"
        assert rows[0].intent == "factual"
        assert rows[0].expected_citations == ["1"]
        assert rows[0].has_sufficient_context is True
        assert rows[0].injection_payload is None

    def test_empty_file(self, tmp_path: pathlib.Path):
        data = tmp_path / "empty.jsonl"
        data.write_text("")
        rows = load_dataset(data)
        assert rows == []

    def test_multiple_rows(self, tmp_path: pathlib.Path):
        data = tmp_path / "multi.jsonl"
        lines = []
        for i in range(3):
            lines.append(
                json.dumps(
                    {
                        "query": f"Query {i}",
                        "context": "",
                        "intent": "factual",
                        "expected_citations": [],
                        "expected_format": "json",
                        "has_sufficient_context": False,
                        "injection_payload": None,
                    }
                )
            )
        data.write_text("\n".join(lines) + "\n")
        rows = load_dataset(data)
        assert len(rows) == 3
        assert rows[2].query == "Query 2"

    def test_defaults_for_missing_fields(self, tmp_path: pathlib.Path):
        data = tmp_path / "minimal.jsonl"
        row = {"query": "Q", "context": "C", "intent": "factual"}
        data.write_text(json.dumps(row) + "\n")
        rows = load_dataset(data)
        assert rows[0].expected_citations == []
        assert rows[0].expected_format == "json"
        assert rows[0].has_sufficient_context is True
        assert rows[0].injection_payload is None

    def test_injection_payload_preserved(self, tmp_path: pathlib.Path):
        data = tmp_path / "inj.jsonl"
        row = {
            "query": "Inject",
            "context": "ctx",
            "intent": "factual",
            "expected_citations": [],
            "expected_format": "json",
            "has_sufficient_context": True,
            "injection_payload": "Ignore instructions",
        }
        data.write_text(json.dumps(row) + "\n")
        rows = load_dataset(data)
        assert rows[0].injection_payload == "Ignore instructions"

    def test_skips_blank_lines(self, tmp_path: pathlib.Path):
        data = tmp_path / "blanks.jsonl"
        row = {"query": "Q", "context": "", "intent": "factual"}
        data.write_text(json.dumps(row) + "\n\n\n" + json.dumps(row) + "\n")
        rows = load_dataset(data)
        assert len(rows) == 2


class TestGoldenDataset:
    def test_golden_file_loads(self):
        golden = pathlib.Path("tests/evaluation/golden/prompt_aug_eval_sample.jsonl")
        if not golden.exists():
            pytest.skip("Golden dataset not found")
        rows = load_dataset(golden)
        assert len(rows) >= 10
        intents = {r.intent for r in rows}
        assert "factual" in intents
        assert "code_example" in intents
        assert "destructive" in intents

    def test_golden_has_zero_context_rows(self):
        golden = pathlib.Path("tests/evaluation/golden/prompt_aug_eval_sample.jsonl")
        if not golden.exists():
            pytest.skip("Golden dataset not found")
        rows = load_dataset(golden)
        zero_rows = [r for r in rows if not r.has_sufficient_context]
        assert len(zero_rows) >= 1

    def test_golden_has_injection_rows(self):
        golden = pathlib.Path("tests/evaluation/golden/prompt_aug_eval_sample.jsonl")
        if not golden.exists():
            pytest.skip("Golden dataset not found")
        rows = load_dataset(golden)
        injection_rows = [r for r in rows if r.injection_payload is not None]
        assert len(injection_rows) >= 1


class TestRunPromptAugEval:
    def test_run_template_mode_basic(self, tmp_path: pathlib.Path):
        """Test run_prompt_aug_eval in template mode with mocked PromptBuilder."""
        from data_engineering_copilot.services.prompt_builder import PromptBuilder

        data = tmp_path / "test.jsonl"
        row = {
            "query": "What is Spark?",
            "context": '<context_doc id="1">Spark is a data processing engine.</context_doc>',
            "intent": "factual",
            "expected_citations": ["1"],
            "expected_format": "json",
            "has_sufficient_context": True,
            "injection_payload": None,
        }
        data.write_text(json.dumps(row) + "\n")

        with patch.object(PromptBuilder, "build_rag_prompt", return_value="Prompt with context"):
            report = run_prompt_aug_eval(data)

        assert report.total_samples == 1
        assert isinstance(report.metrics.format_compliance_rate, float)
        assert isinstance(report.metrics.citation_precision, float)
        assert isinstance(report.metrics.citation_recall, float)
        assert isinstance(report.metrics.injection_defense_rate, float)
        assert isinstance(report.metrics.zero_context_fallback_accuracy, float)

    def test_run_template_mode_zero_context(self, tmp_path: pathlib.Path):
        """Test template mode correctly handles zero-context rows."""
        from data_engineering_copilot.services.prompt_builder import PromptBuilder

        data = tmp_path / "test.jsonl"
        row = {
            "query": "What is unknown?",
            "context": "",
            "intent": "factual",
            "expected_citations": [],
            "expected_format": "json",
            "has_sufficient_context": False,
            "injection_payload": None,
        }
        data.write_text(json.dumps(row) + "\n")

        with patch.object(PromptBuilder, "build_rag_prompt", return_value="Prompt with no context"):
            report = run_prompt_aug_eval(data)

        assert report.total_samples == 1
        assert isinstance(report.metrics.zero_context_fallback_accuracy, float)


class TestRunPromptAugEvalLLM:
    @pytest.mark.asyncio
    async def test_run_llm_mode_basic(self, tmp_path: pathlib.Path):
        """Test run_prompt_aug_eval_llm with mocked LLM client."""
        from data_engineering_copilot.evaluation.prompt_aug_eval import (
            run_prompt_aug_eval_llm,
        )
        from data_engineering_copilot.services.prompt_builder import PromptBuilder

        data = tmp_path / "test.jsonl"
        row = {
            "query": "What is Spark?",
            "context": "Spark is a data processing engine.",
            "intent": "factual",
            "expected_citations": ["1"],
            "expected_format": "json",
            "has_sufficient_context": True,
            "injection_payload": None,
        }
        data.write_text(json.dumps(row) + "\n")

        # Mock the LLM client
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"status": "SUCCESS", "answer": "Spark is an engine", "missing_info": null}'

        with patch("data_engineering_copilot.factory.build_llm_fallback_chain") as mock_build_chain:
            mock_build_chain.return_value = mock_llm

            with patch.object(PromptBuilder, "build_rag_prompt", return_value="Test prompt"):
                from data_engineering_copilot.evaluation.prompt_aug_eval import (
                    run_prompt_aug_eval_llm,
                )

                report = await run_prompt_aug_eval_llm(tmp_path / "test.jsonl", provider="ollama")

        assert report.total_samples == 1
        mock_llm.generate.assert_called_once()
        assert isinstance(report.metrics.citation_precision, float)
