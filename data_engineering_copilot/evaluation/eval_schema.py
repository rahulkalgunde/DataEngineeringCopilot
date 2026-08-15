"""Unified evaluation dataset schema and validation.

Two record kinds share one schema:

- ``recall`` rows (used by ``dec evaluate --spark`` and ``dec eval-coverage``):
  ``id``, ``question``, ``expected_terms``, ``expected_urls`` (and optional
  ``expected_doc_types`` / ``expected_modules`` / ``must_not_require`` /
  ``forbidden_terms`` / ``out_of_scope``).
- ``qa`` rows (used by ``dec evaluate``): ``id``, ``question``,
  ``ground_truth``, ``contexts``, ``source_name``.

Optional metadata on either kind: ``source_name``, ``doc_type``, ``intent``,
``complexity``, ``abstraction``.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

RECALL_FIELDS = (
    "expected_terms",
    "expected_urls",
    "expected_doc_types",
    "expected_modules",
    "must_not_require",
    "forbidden_terms",
    "out_of_scope",
)
QA_FIELDS = ("ground_truth", "contexts")
METADATA_FIELDS = ("source_name", "doc_type", "intent", "complexity", "abstraction")

_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class EvalKind(StrEnum):
    """Discriminator between the two record layouts."""

    RECALL = "recall"
    QA = "qa"


class Complexity(StrEnum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class Abstraction(StrEnum):
    SPECIFIC = "specific"
    ABSTRACT = "abstract"


class Intent(StrEnum):
    FACTUAL = "factual"
    HOW_TO = "how_to"
    CODE = "code"
    API_REFERENCE = "api_reference"
    CONFIGURATION = "configuration"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    SYNTHESIS = "synthesis"
    OUT_OF_SCOPE = "out_of_scope"


def kind_of(row: dict) -> EvalKind:
    """Detect the record kind from its fields."""
    if row.get("expected_terms") or row.get("expected_urls"):
        return EvalKind.RECALL
    return EvalKind.QA


def parse_eval_rows(path: str | Path) -> list[dict]:
    """Read a JSONL evaluation file into a list of row dicts."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_eval_row(row: dict) -> list[str]:
    """Validate one evaluation row.

    Returns a (possibly empty) list of error strings. An empty list means the
    row is schema-valid.
    """
    errors: list[str] = []
    label = str(row.get("id") or (row.get("question") or "")[:40])

    def err(msg: str) -> None:
        errors.append(f"{label}: {msg}")

    if not str(row.get("question") or "").strip():
        err("question is required")

    row_id = row.get("id")
    if row_id is None:
        err("id is required (stable lowercase-hyphen slug)")
    elif not _SLUG_RE.match(str(row_id)):
        err(f"id {row_id!r} must match {_SLUG_RE.pattern!r}")

    oos = bool(row.get("out_of_scope", False))
    kind = kind_of(row)

    if kind is EvalKind.RECALL:
        if not row.get("expected_terms"):
            err("recall row must specify non-empty expected_terms")
        if oos and row.get("expected_urls"):
            err("out-of-scope row must not carry expected_urls")
        if not oos and not row.get("expected_urls"):
            err("in-scope recall row must specify expected_urls")
    else:
        if not str(row.get("ground_truth") or "").strip():
            err("qa row must specify ground_truth")

    if oos and not row.get("expected_terms"):
        err("out-of-scope row must carry expected_terms (the refusal trigger)")

    return errors


def write_eval_rows(path: str | Path, rows: list[dict]) -> None:
    """Serialize rows back to a JSONL file, preserving insertion order."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
