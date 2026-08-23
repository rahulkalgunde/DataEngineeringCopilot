"""CI gate: every golden recall/qa row must satisfy the eval schema."""

import json
import pathlib

from data_engineering_copilot.evaluation.eval_schema import validate_eval_row

GOLDEN = pathlib.Path("tests/evaluation/golden")
PATTERNS = ("recall_*.jsonl", "qa_*.jsonl")


def _rows(path: pathlib.Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def test_all_golden_schema_rows_validate():
    files = sorted(p for pat in PATTERNS for p in GOLDEN.glob(pat))
    assert files, "golden schema datasets missing"
    violations: list[str] = []
    for f in files:
        for i, row in enumerate(_rows(f)):
            errors = validate_eval_row(row)
            if errors:
                violations.append(f"{f.name}:{i} {errors}")
    assert not violations, "\n".join(violations[:20])
