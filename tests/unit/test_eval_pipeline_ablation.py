"""Pipeline ablation flag contract — TDD for Task 2 (ADR-011)."""

from __future__ import annotations

import subprocess
import sys


def test_pipeline_ablation_help() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "data_engineering_copilot.cli", "eval-retrieval", "--help"],
        capture_output=True,
        text=True,
    )
    combined = (r.stdout or "") + (r.stderr or "")
    assert "--pipeline-ablation" in combined


def test_pipeline_ablation_choices() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "data_engineering_copilot.cli", "eval-retrieval", "--help"],
        capture_output=True,
        text=True,
    )
    combined = (r.stdout or "") + (r.stderr or "")
    # choices per plan: guardrails, sibling, dedup, all
    for choice in ("guardrails", "sibling", "dedup", "all"):
        assert choice in combined


def test_pipeline_ablation_stages_constant() -> None:
    from data_engineering_copilot.evaluation.retrieval import PIPELINE_ABLATION_STAGES

    assert set(PIPELINE_ABLATION_STAGES) == {"guardrails", "sibling", "dedup"}
