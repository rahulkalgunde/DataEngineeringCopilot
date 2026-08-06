"""Tests for the FLASH executor driver (``plan_executor.py``).

Hermetic: no network, no git, no Qdrant. The command runner is injected so the
orchestration, checkpointing, dry-run, and failure-schema behavior are exercised
without touching the live index.
"""

from __future__ import annotations

import json
from typing import Any, cast

from data_engineering_copilot import plan_executor
from data_engineering_copilot.plan_executor import (
    CHECKPOINT_FILENAME,
    EXIT_BLOCKED,
    EXIT_GATE,
    EXIT_HALTED,
    EXIT_OK,
    EXIT_USAGE,
    FAILURE_FILENAME,
    PLAN_PHASES,
    RESULT_FILENAME,
    SCHEMA_VERSION,
    PlanOptions,
    StepResult,
    _analyze_manifest,
    default_run_id,
    run_plan,
    write_failure,
)

_GEN = "spark-test-generation-1"


def _patch_env(monkeypatch, tmp_path):
    """Point the run root at tmp and pin active-generation discovery."""
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(plan_executor, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(plan_executor, "_discover_active_generation", lambda: _GEN)
    return runs_root


def _ok_runner(argv, env=None):
    return StepResult(command=list(argv), exit_code=0, output="ok")


def _boom_runner(argv, env=None):
    raise AssertionError("runner must not be called in dry-run mode")


def _options(**kwargs):
    kwargs.setdefault("runner", _ok_runner)
    return PlanOptions(**kwargs)


def test_plan_phases_registry() -> None:
    assert [p.id for p in PLAN_PHASES] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert [p.name for p in PLAN_PHASES] == [
        "preflight",
        "baseline-eval",
        "chunk-audit",
        "contextual-index",
        "multi-query-fusion",
        "rerank-context",
        "tuning",
        "rollout",
    ]
    assert [p.destructive for p in PLAN_PHASES].count(True) == 1
    assert PLAN_PHASES[7].destructive is True
    assert [p.id for p in PLAN_PHASES if p.requires_code_change] == [4, 5, 6]


def test_default_run_id_format() -> None:
    rid = default_run_id()
    assert len(rid) == 16
    assert rid[8] == "T"
    assert rid[-1] == "Z"
    assert int(rid[:8]) > 20260101


def test_analyze_manifest_distributions_and_duplicates() -> None:
    manifest = {
        "manifest_hash": "abc123",
        "files": [
            {"stream": "sql", "relative_path": "same.md", "doc_type": "doc", "language": "markdown"},
            {"stream": "py", "relative_path": "same.md", "doc_type": "api", "language": "python"},
            {"stream": "sql", "relative_path": "unique.md", "doc_type": "doc", "language": "markdown"},
        ],
    }
    analysis = cast(dict[str, Any], _analyze_manifest(manifest))
    assert analysis["file_count"] == 3
    assert analysis["distributions"]["doc_type"] == {"doc": 2, "api": 1}
    assert analysis["duplicate_relative_paths_count"] == 1
    assert analysis["duplicate_relative_paths_examples"]["same.md"] == ["py", "sql"]
    assert analysis["content_based"] is False
    assert analysis["samples_count"] == 3  # up to 20 per doc_type, all files here


def test_analyze_manifest_sample_cap() -> None:
    files = [
        {"stream": "sql", "relative_path": f"f{i}.md", "doc_type": "doc", "language": "markdown"} for i in range(50)
    ]
    analysis = cast(dict[str, Any], _analyze_manifest({"files": files}))
    assert analysis["samples_count"] <= 20


def test_dry_run_does_not_execute_or_persist(monkeypatch, tmp_path) -> None:
    runs_root = _patch_env(monkeypatch, tmp_path)
    exit_code = run_plan(_options(dry_run=True, run_id="dry1", phase=3, runner=_boom_runner))
    assert exit_code == EXIT_OK
    run_dir = runs_root / "dry1"
    assert not (run_dir / CHECKPOINT_FILENAME).exists()
    result = json.loads((run_dir / RESULT_FILENAME).read_text())
    assert result["mode"] == "dry-run"
    assert result["phases"]["3"]["status"] == "planned"
    planned = result["phases"]["3"]["planned"]
    assert any("spark-build" in cmd["command"] for cmd in planned)


def test_phase1_completes_and_checkpoints(monkeypatch, tmp_path) -> None:
    runs_root = _patch_env(monkeypatch, tmp_path)
    exit_code = run_plan(_options(run_id="run1", phase=1))
    assert exit_code == EXIT_OK
    run_dir = runs_root / "run1"
    checkpoint = json.loads((run_dir / CHECKPOINT_FILENAME).read_text())
    assert checkpoint["active_generation"] == _GEN
    assert "1" in checkpoint["completed"]
    assert (run_dir / "artifacts" / "baseline.json").exists()
    assert (run_dir / "artifacts" / "eval_spark.txt").exists()


def test_resume_skips_completed_phase(monkeypatch, tmp_path) -> None:
    _patch_env(monkeypatch, tmp_path)
    calls: list[list[str]] = []

    def counting_runner(argv, env=None):
        calls.append(list(argv))
        return StepResult(command=list(argv), exit_code=0, output="ok")

    first = run_plan(_options(run_id="resume1", phase=1, runner=counting_runner))
    assert first == EXIT_OK
    calls_before = len(calls)
    second = run_plan(_options(run_id="resume1", phase=1, runner=counting_runner))
    assert second == EXIT_OK
    assert len(calls) == calls_before  # already completed, not re-run
    result = json.loads((tmp_path / "runs" / "resume1" / RESULT_FILENAME).read_text())
    assert result["phases"]["1"]["status"] == "completed"


def test_force_re_runs_completed_phase(monkeypatch, tmp_path) -> None:
    _patch_env(monkeypatch, tmp_path)
    calls: list[list[str]] = []

    def counting_runner(argv, env=None):
        calls.append(list(argv))
        return StepResult(command=list(argv), exit_code=0, output="ok")

    run_plan(_options(run_id="forcerun", phase=1, runner=counting_runner))
    run_plan(_options(run_id="forcerun", phase=1, force=True, runner=counting_runner))
    assert len(calls) == 4  # two runs x two evaluate commands


def test_phase3_skipped_without_candidate(monkeypatch, tmp_path) -> None:
    _patch_env(monkeypatch, tmp_path)
    exit_code = run_plan(_options(run_id="skip3", phase=3))
    assert exit_code == EXIT_OK
    result = json.loads((tmp_path / "runs" / "skip3" / RESULT_FILENAME).read_text())
    assert result["phases"]["3"]["status"] == "skipped"


def test_phase3_blocked_without_force(monkeypatch, tmp_path) -> None:
    _patch_env(monkeypatch, tmp_path)
    exit_code = run_plan(_options(run_id="block3", phase=3, candidate_generation="candidate-gen", runner=_boom_runner))
    assert exit_code == EXIT_BLOCKED
    assert (tmp_path / "runs" / "block3" / FAILURE_FILENAME).exists()


def test_phase4_halted_requires_code_change(monkeypatch, tmp_path) -> None:
    _patch_env(monkeypatch, tmp_path)
    exit_code = run_plan(_options(run_id="halt4", phase=4, runner=_boom_runner))
    assert exit_code == EXIT_HALTED
    result = json.loads((tmp_path / "runs" / "halt4" / RESULT_FILENAME).read_text())
    assert result["phases"]["4"]["status"] == "halted"


def test_phase_failure_writes_failure_schema(monkeypatch, tmp_path) -> None:
    _patch_env(monkeypatch, tmp_path)

    def failing_runner(argv, env=None):
        return StepResult(command=list(argv), exit_code=1, output="boom")

    exit_code = run_plan(_options(run_id="fail1", phase=1, runner=failing_runner))
    assert exit_code == EXIT_GATE
    failure_path = tmp_path / "runs" / "fail1" / FAILURE_FILENAME
    failure = json.loads(failure_path.read_text())
    assert failure["schema_version"] == SCHEMA_VERSION
    assert failure["phase"] == 1
    assert failure["phase_name"] == "baseline-eval"
    assert failure["exit_code"] == 1
    assert failure["output_excerpt"] == "boom"
    assert "resume_command" in failure


def test_unknown_phase_returns_usage(monkeypatch, tmp_path) -> None:
    _patch_env(monkeypatch, tmp_path)
    assert run_plan(_options(run_id="usage1", phase=9, runner=_boom_runner)) == EXIT_USAGE


def test_write_failure_schema_direct(tmp_path) -> None:
    from data_engineering_copilot.plan_executor import Phase, RunContext

    ctx = RunContext(
        options=PlanOptions(),
        run_dir=tmp_path / "run-test",
        runner=_ok_runner,
    )
    failure = write_failure(ctx, Phase(2, "chunk-audit"), "phase-chunk-audit", "manifest unreadable")
    assert failure["schema_version"] == SCHEMA_VERSION
    assert failure["phase_name"] == "chunk-audit"
    assert failure["step"] == "phase-chunk-audit"
    assert failure["error"] == "manifest unreadable"
    assert failure["exit_code"] is None
    assert (tmp_path / "run-test" / FAILURE_FILENAME).exists()
