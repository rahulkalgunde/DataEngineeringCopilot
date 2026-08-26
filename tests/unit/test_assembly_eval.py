"""Tests for evaluation/assembly_eval.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from data_engineering_copilot.evaluation.assembly_eval import (
    AssemblyEvalRow,
    load_assembly_eval_dataset,
)


def _make_row(query: str = "q", urls: list[str] | None = None, facts: list[str] | None = None) -> AssemblyEvalRow:
    return AssemblyEvalRow(query=query, source_urls=urls or ["http://a.com"], gold_facts=facts or ["fact1"])


class TestAssemblyEvalRow:
    def test_creation(self) -> None:
        row = _make_row()
        assert row.query == "q"
        assert row.source_urls == ["http://a.com"]
        assert row.gold_facts == ["fact1"]


class TestLoadAssemblyEvalDataset:
    def test_loads_valid_file(self) -> None:
        data = [
            {"query": "q1", "source_urls": ["http://a.com"], "gold_facts": ["f1"]},
            {"query": "q2", "source_urls": ["http://b.com"]},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
            f.flush()
            path = Path(f.name)

        rows = load_assembly_eval_dataset(path)
        assert len(rows) == 2
        assert rows[0].query == "q1"
        assert rows[1].query == "q2"
        assert rows[1].gold_facts == []

    def test_skips_empty_lines(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"query": "q", "source_urls": []}\n\n')
            f.flush()
            path = Path(f.name)

        rows = load_assembly_eval_dataset(path)
        assert len(rows) == 1
