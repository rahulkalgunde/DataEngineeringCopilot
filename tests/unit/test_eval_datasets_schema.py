"""Gate: every evaluation dataset in tests/evaluation is schema-valid.

Runs without the corpus (no infra), so CI can enforce dataset quality on every
commit. Corpus-coverage validation is separate (``dec eval-coverage`` / the
coverage validator tests).
"""

from __future__ import annotations

from pathlib import Path

from data_engineering_copilot.evaluation.eval_schema import parse_eval_rows, validate_eval_row

_EVALS_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def _all_dataset_rows() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for p in sorted(_EVALS_DIR.glob("*.jsonl")):
        for row in parse_eval_rows(p):
            rows.append((p.name, row))
    return rows


def test_all_eval_rows_schema_valid():
    rows = _all_dataset_rows()
    assert rows, "no eval datasets found"
    errors: list[str] = []
    for _filename, row in rows:
        errors.extend(validate_eval_row(row))
    assert not errors, "schema violations:\n  " + "\n  ".join(errors[:20])


def test_every_dataset_has_unique_ids():
    seen: set[str] = set()
    dupes: list[str] = []
    for filename, row in _all_dataset_rows():
        rid = row.get("id")
        if not rid:
            continue
        if rid in seen:
            dupes.append(f"{filename}:{rid}")
        seen.add(rid)
    assert not dupes, f"duplicate eval row ids: {dupes[:10]}"


def test_oos_rows_carry_refusal_terms_and_no_urls():
    for filename, row in _all_dataset_rows():
        if row.get("out_of_scope"):
            assert row.get("expected_terms"), f"{filename} {row['id']} OOS row needs expected_terms"
            assert not row.get("expected_urls"), f"{filename} {row['id']} OOS row must not carry expected_urls"


def test_in_scope_recall_rows_carry_evidence():
    for filename, row in _all_dataset_rows():
        if row.get("expected_terms") and not row.get("out_of_scope"):
            assert row.get("expected_urls"), f"{filename} {row['id']} in-scope recall row needs expected_urls"
