"""Hermetic semantic gate for the golden retrieval recall datasets.

The existing ``test_eval_datasets_schema.py`` globs ``tests/evaluation/*.jsonl``
(root only) and so never covered ``tests/evaluation/golden/*`` — which is where
the gate dataset lives. That coverage gap is why the ``claude_platform`` /
``claude_code`` template corruption (163 rows, wrong ``expected_urls``) went
undetected for so long (see ``scripts/repair_golden_recall_corruption.py``).

This gate closes the gap and encodes the fix's invariants:
- no snake_case source-name artifact may leak into any question,
- every golden recall row is schema-valid,
- the repaired Claude corpora's in-scope rows must carry ``expected_terms``.

Runs with no corpus / no infra, so CI enforces it on every commit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from data_engineering_copilot.evaluation.eval_schema import validate_eval_row

_GOLDEN = Path(__file__).resolve().parents[1] / "evaluation" / "golden"

# A leaked source-name template artifact: the bare snake_case corpus identifier.
# These are not words — they are generator internals that leaked into queries.
_ARTIFACT_RE = re.compile(r"\b(?:claude_platform|claude_code)\b")

# Files that carry the corruption-repaired Claude corpora (strict term check).
_REPAIRED = {"recall_claude_platform.jsonl", "recall_claude_code.jsonl"}


def _recall_rows() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for p in sorted(_GOLDEN.glob("recall_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append((p.name, json.loads(line)))
    return rows


def test_golden_recall_has_no_template_artifacts():
    hits: list[str] = []
    for filename, row in _recall_rows():
        q = row.get("question") or ""
        if _ARTIFACT_RE.search(q):
            hits.append(f"{filename}:{row.get('id')}: {q!r}")
    assert not hits, "template artifacts leaked into questions:\n  " + "\n  ".join(hits)


def test_golden_recall_rows_schema_valid():
    errors: list[str] = []
    for filename, row in _recall_rows():
        for e in validate_eval_row(row):
            errors.append(f"{filename}: {e}")
    assert not errors, "schema violations:\n  " + "\n  ".join(errors[:20])


def test_repaired_claude_inscope_rows_carry_expected_terms():
    missing: list[str] = []
    for filename, row in _recall_rows():
        if filename not in _REPAIRED:
            continue
        if row.get("out_of_scope"):
            continue
        if not row.get("expected_terms"):
            missing.append(f"{filename}:{row.get('id')}")
    assert not missing, "in-scope Claude rows missing expected_terms:\n  " + "\n  ".join(missing)
