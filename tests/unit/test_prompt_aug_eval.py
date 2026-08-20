"""Tests for prompt augmentation evaluation harness."""

from __future__ import annotations

import json
import pathlib

import pytest

from data_engineering_copilot.evaluation.prompt_aug_eval import (
    load_dataset,
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
