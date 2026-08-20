"""Tests for isolated assembly evaluation harness."""

from __future__ import annotations

import json
import pathlib

import pytest

from data_engineering_copilot.evaluation.assembly_eval import (
    load_assembly_eval_dataset,
)

pytestmark = pytest.mark.unit


class TestLoadDataset:
    def test_load_sample(self, tmp_path: pathlib.Path):
        data = tmp_path / "test.jsonl"
        data.write_text(json.dumps({"query": "test", "source_urls": ["http://a"], "gold_facts": ["fact1"]}) + "\n")
        rows = load_assembly_eval_dataset(data)
        assert len(rows) == 1
        assert rows[0].query == "test"
        assert rows[0].gold_facts == ["fact1"]

    def test_empty_file(self, tmp_path: pathlib.Path):
        data = tmp_path / "empty.jsonl"
        data.write_text("")
        rows = load_assembly_eval_dataset(data)
        assert rows == []
