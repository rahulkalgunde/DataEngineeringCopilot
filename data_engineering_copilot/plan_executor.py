"""FLASH-executor driver for the general RAG improvement plan.

Implements the executor contract in ``plans/rag_general_improvement_execution_plan.md``:

- automatic active-generation discovery
- machine-readable JSON artifacts per run
- artifact creation under ``.rag_eval/runs/<run_id>/``
- explicit dry-run mode for build/activate/rollback
- checkpoint/resume handling
- standard failure JSON schema

The driver shells out to the ``dec`` console script so every phase reuses the
existing CLI behavior and exit codes. Phases that require a code change
(4/5/6) halt with a structured status instead of pretending to run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEC = PROJECT_ROOT / "dec_venv" / "bin" / "dec"
RUNS_ROOT = PROJECT_ROOT / ".rag_eval" / "runs"

CHECKPOINT_FILENAME = "checkpoint.json"
RESULT_FILENAME = "result.json"
FAILURE_FILENAME = "failure.json"
SCHEMA_VERSION = "1"

#: Exit codes are stable and documented so executors can branch on them.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_GATE = 4
EXIT_COMMAND = 5
EXIT_HALTED = 10
EXIT_BLOCKED = 11


@dataclass(frozen=True)
class Phase:
    id: int
    name: str
    destructive: bool = False
    requires_code_change: bool = False


PLAN_PHASES: tuple[Phase, ...] = (
    Phase(0, "preflight"),
    Phase(1, "baseline-eval"),
    Phase(2, "chunk-audit"),
    Phase(3, "contextual-index"),
    Phase(4, "multi-query-fusion", requires_code_change=True),
    Phase(5, "rerank-context", requires_code_change=True),
    Phase(6, "tuning", requires_code_change=True),
    Phase(7, "rollout", destructive=True),
)

PHASES_BY_ID: dict[int, Phase] = {p.id: p for p in PLAN_PHASES}


@dataclass(frozen=True)
class StepResult:
    command: list[str]
    exit_code: int
    output: str
    error: str | None = None


def default_run_id() -> str:
    """UTC timestamp run id, e.g. ``20260805T123000Z``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_command(
    argv: Sequence[str],
    env: dict[str, str] | None = None,
    *,
    cwd: Path = PROJECT_ROOT,
    timeout: int = 3600,
) -> StepResult:
    """Run a command, capturing combined output. Never raises on command failure."""
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return StepResult(
            command=list(argv),
            exit_code=-1,
            output=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            error=f"timed out after {timeout}s",
        )
    except OSError as exc:
        return StepResult(command=list(argv), exit_code=-1, output="", error=str(exc))
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return StepResult(command=list(argv), exit_code=proc.returncode, output=combined)


def default_runner(argv: Sequence[str], env: dict[str, str] | None = None) -> StepResult:
    """Default runner for ``dec`` commands (``<repo>/dec_venv/bin/dec``)."""
    return run_command([str(DEC), *argv], env=env)


@dataclass
class PlanOptions:
    run_id: str | None = None
    phase: int | None = None
    dry_run: bool = False
    force: bool = False
    candidate_generation: str | None = None
    json_output: bool = False
    runner: Callable[[Sequence[str], dict[str, str] | None], StepResult] = default_runner


@dataclass
class RunContext:
    options: PlanOptions
    run_dir: Path
    runner: Callable[[Sequence[str], dict[str, str] | None], StepResult]
    active_generation: str = ""
    completed: dict[str, object] = field(default_factory=dict)


def _excerpt(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _artifact_path(ctx: RunContext, name: str) -> Path:
    artifacts = ctx.run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts / name


def write_artifact(ctx: RunContext, name: str, content: str) -> Path:
    path = _artifact_path(ctx, name)
    path.write_text(content, encoding="utf-8")
    return path


def write_json_artifact(ctx: RunContext, name: str, payload: object) -> Path:
    return write_artifact(ctx, name, json.dumps(payload, indent=2, default=str))


def _log(ctx: RunContext, name: str, result: StepResult) -> None:
    logs = ctx.run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": result.command,
        "exit_code": result.exit_code,
        "error": result.error,
        "output": result.output,
    }
    (logs / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(run_dir: Path) -> dict[str, object]:
    path = run_dir / CHECKPOINT_FILENAME
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def save_checkpoint(ctx: RunContext) -> None:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": ctx.run_dir.name,
        "active_generation": ctx.active_generation,
        "completed": ctx.completed,
    }
    (ctx.run_dir / CHECKPOINT_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_failure(
    ctx: RunContext,
    phase: Phase,
    step: str,
    error: str,
    result: StepResult | None = None,
) -> dict[str, object]:
    """Write the standard failure JSON schema and return it."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    failure: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": ctx.run_dir.name,
        "timestamp": datetime.now(UTC).isoformat(),
        "phase": phase.id,
        "phase_name": phase.name,
        "step": step,
        "error": error,
        "command": list(result.command) if result else None,
        "exit_code": result.exit_code if result else None,
        "output_excerpt": _excerpt(result.output) if result else None,
        "artifacts_dir": str(ctx.run_dir / "artifacts"),
        "checkpoint": str(ctx.run_dir / CHECKPOINT_FILENAME),
        "resume_command": f"dec rag-plan --run-id {ctx.run_dir.name}",
    }
    (ctx.run_dir / FAILURE_FILENAME).write_text(json.dumps(failure, indent=2), encoding="utf-8")
    return failure


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def _git_status() -> StepResult:
    return run_command(["git", "status", "--short"], timeout=30)


def _qdrant_reachable(timeout: int = 5) -> tuple[bool, str]:
    from data_engineering_copilot.config.settings import settings

    try:
        req = urllib.request.Request(f"{settings.qdrant_url}/", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        return False, str(exc)


def _discover_active_generation() -> str:
    from data_engineering_copilot.config.settings import resolve_active_generation

    return (resolve_active_generation() or "").strip()


def run_phase_preflight(ctx: RunContext, phase: Phase) -> dict[str, object]:
    """Phase 0: reproducibility gate."""
    git = _git_status()
    status_step = ctx.runner(["status"], None)
    _log(ctx, "phase0_status", status_step)
    config_step = ctx.runner(["spark-config-check"], None)
    _log(ctx, "phase0_config_check", config_step)

    qdrant_ok, qdrant_note = _qdrant_reachable()

    active = _discover_active_generation()
    ctx.active_generation = active
    validation: dict[str, object] | None = None
    if active:
        val_step = ctx.runner(["spark-validate", "--generation", active], None)
        _log(ctx, f"phase0_validate_{active}", val_step)
        validation = {
            "command": val_step.command,
            "exit_code": val_step.exit_code,
            "output_excerpt": _excerpt(val_step.output),
        }
        if val_step.exit_code != 0:
            write_json_artifact(ctx, "preflight.json", {"validation_exit_code": val_step.exit_code})
            return {"ok": False, "reason": f"validation failed for generation {active}"}

    gate_passed = qdrant_ok and bool(active) and bool(validation) and validation["exit_code"] == 0
    issues: list[str] = []
    preflight: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "active_generation": active,
        "qdrant_reachable": qdrant_ok,
        "qdrant_note": qdrant_note,
        "git_status": [line for line in git.output.splitlines()],
        "validation": validation,
        "gate_passed": gate_passed,
        "issues": issues,
    }
    if not active:
        issues.append("no active generation found")
    if not qdrant_ok:
        issues.append(f"qdrant unreachable: {qdrant_note}")
    if validation and validation["exit_code"] != 0:
        issues.append("active generation validation failed")
    write_json_artifact(ctx, "preflight.json", preflight)
    write_artifact(ctx, "git_status.txt", git.output)
    return {"ok": gate_passed, "issues": issues, "preflight": preflight}


def run_phase_baseline(ctx: RunContext, phase: Phase) -> dict[str, object]:
    """Phase 1: baseline retrieval evaluation."""
    eval_dir = ctx.run_dir / "artifacts" / "eval"
    spark_step = ctx.runner(["evaluate", "--spark", "--output-dir", str(eval_dir)], None)
    _log(ctx, "phase1_evaluate_spark", spark_step)
    write_artifact(ctx, "eval_spark.txt", spark_step.output)

    general_step = ctx.runner(["evaluate", "--dataset", "tests/evaluation/eval_dataset.jsonl"], None)
    _log(ctx, "phase1_evaluate_general", general_step)
    write_artifact(ctx, "eval_general.txt", general_step.output)

    baseline: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "spark": {
            "exit_code": spark_step.exit_code,
            "output_excerpt": _excerpt(spark_step.output),
        },
        "general": {
            "exit_code": general_step.exit_code,
            "output_excerpt": _excerpt(general_step.output),
        },
    }
    write_json_artifact(ctx, "baseline.json", baseline)
    if spark_step.exit_code != 0:
        return {"ok": False, "reason": "spark evaluation failed", "step_result": spark_step}
    if general_step.exit_code != 0:
        return {"ok": False, "reason": "general evaluation failed", "step_result": general_step}
    return {"ok": True, "spark_exit_code": spark_step.exit_code, "general_exit_code": general_step.exit_code}


def _analyze_manifest(manifest: dict[str, object]) -> dict[str, object]:
    files = manifest.get("files", [])
    if not isinstance(files, list):
        files = []
    by_doc_type: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_stream: Counter[str] = Counter()
    path_streams: defaultdict[str, list[str]] = defaultdict(list)
    for entry in files:
        if not isinstance(entry, dict):
            continue
        by_doc_type[str(entry.get("doc_type", "?") or "?")] += 1
        by_language[str(entry.get("language", "?") or "?")] += 1
        rel = str(entry.get("relative_path", "") or "")
        stream = str(entry.get("stream", "") or "")
        by_stream[stream] += 1
        path_streams[rel].append(stream)

    duplicate_paths: dict[str, list[str]] = {}
    for path, streams in path_streams.items():
        unique = sorted(set(streams))
        if len(unique) > 1:
            duplicate_paths[path] = unique

    samples: list[dict[str, object]] = []
    for doc_type in sorted(by_doc_type):
        picked = 0
        for entry in files:
            if picked >= 20:
                break
            if not isinstance(entry, dict) or entry.get("doc_type") != doc_type:
                continue
            samples.append(
                {
                    "stream": entry.get("stream"),
                    "relative_path": entry.get("relative_path"),
                    "doc_type": entry.get("doc_type"),
                    "language": entry.get("language"),
                    "source_url": entry.get("source_url"),
                }
            )
            picked += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": manifest.get("manifest_hash"),
        "file_count": len(files),
        "distributions": {
            "doc_type": dict(sorted(by_doc_type.items(), key=lambda kv: kv[1], reverse=True)),
            "language": dict(sorted(by_language.items(), key=lambda kv: kv[1], reverse=True)),
            "stream": dict(sorted(by_stream.items(), key=lambda kv: kv[1], reverse=True)),
        },
        "duplicate_relative_paths_count": len(duplicate_paths),
        "duplicate_relative_paths_examples": dict(sorted(duplicate_paths.items())[:50]),
        "content_based": False,
        "limitations": ["manifest carries no sizes; short-stub/forwarding detection requires reading file content"],
        "samples_count": len(samples),
        "samples": samples,
    }


def run_phase_chunk_audit(ctx: RunContext, phase: Phase) -> dict[str, object]:
    """Phase 2: chunk quality audit over the materialized source manifest."""
    manifest_path = _artifact_path(ctx, "manifest.json")
    manifest_step = ctx.runner(["spark-manifest", "--output", str(manifest_path)], None)
    _log(ctx, "phase2_manifest", manifest_step)
    if manifest_step.exit_code != 0:
        return {"ok": False, "reason": "spark-manifest failed", "exit_code": manifest_step.exit_code}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"manifest unreadable: {exc}"}

    analysis = _analyze_manifest(manifest)
    samples = analysis.pop("samples", [])
    _write_samples(ctx, samples)
    write_json_artifact(ctx, "chunk_quality.json", analysis)
    return {"ok": True, "file_count": analysis["file_count"], "chunk_quality": analysis}


def _write_samples(ctx: RunContext, samples: object) -> None:
    path = _artifact_path(ctx, "chunk_samples.jsonl")
    lines = "\n".join(json.dumps(s) for s in samples if isinstance(s, dict)) if isinstance(samples, list) else ""
    path.write_text(lines + ("\n" if lines else ""), encoding="utf-8")


def run_phase_contextual_index(ctx: RunContext, phase: Phase) -> dict[str, object]:
    """Phase 3: build a candidate generation without activating it."""
    gen = (ctx.options.candidate_generation or "").strip()
    if not gen:
        return {"status": "skipped", "reason": "requires --candidate-generation"}
    if ctx.options.dry_run:
        return {
            "status": "dry-run",
            "planned": [
                ["spark-build", "--generation", gen],
                ["spark-validate", "--generation", gen],
            ],
        }
    if not ctx.options.force:
        return {"status": "blocked", "reason": "requires --force to build a candidate generation"}

    build_step = ctx.runner(["spark-build", "--generation", gen], None)
    _log(ctx, f"phase3_build_{gen}", build_step)
    write_artifact(ctx, f"build_{gen}.txt", build_step.output)
    if build_step.exit_code != 0:
        return {"ok": False, "reason": "spark-build failed", "exit_code": build_step.exit_code}

    validate_step = ctx.runner(["spark-validate", "--generation", gen], None)
    _log(ctx, f"phase3_validate_{gen}", validate_step)
    write_artifact(ctx, f"validate_{gen}.txt", validate_step.output)
    result = {
        "ok": validate_step.exit_code == 0,
        "generation": gen,
        "validate_exit_code": validate_step.exit_code,
        "activated": False,
    }
    write_json_artifact(ctx, "contextual_index.json", result)
    if not result["ok"]:
        result["reason"] = "candidate generation validation failed"
    return result


def run_phase_code_change(ctx: RunContext, phase: Phase) -> dict[str, object]:
    """Phases 4/5/6: halt with a structured status until the code change lands."""
    return {
        "status": "halted",
        "reason": (
            f"phase {phase.id} ({phase.name}) requires a code change that is not "
            "implemented yet; see the plan and the latest retrieval diagnostics "
            "before deciding whether to implement it."
        ),
    }


def run_phase_rollout(ctx: RunContext, phase: Phase) -> dict[str, object]:
    """Phase 7: activate a candidate generation, evaluate, and keep rollback state."""
    gen = (ctx.options.candidate_generation or "").strip()
    if not gen:
        return {"status": "skipped", "reason": "requires --candidate-generation"}
    prior = ctx.active_generation
    if ctx.options.dry_run:
        return {
            "status": "dry-run",
            "planned": [
                {"cmd": ["spark-activate", "--generation", gen], "env": {"FORCE": "1"}},
                {"cmd": ["evaluate", "--spark"], "env": None},
                {"cmd": ["spark-rollback", "--generation", gen], "env": {"FORCE": "1"}},
            ],
            "note": "rollback command listed for reference; only run on gate failure",
        }
    if not ctx.options.force:
        return {"status": "blocked", "reason": "requires --force to activate a generation"}

    activate_step = ctx.runner(["spark-activate", "--generation", gen], {"FORCE": "1"})
    _log(ctx, f"phase7_activate_{gen}", activate_step)
    write_artifact(ctx, f"activate_{gen}.txt", activate_step.output)
    if activate_step.exit_code != 0:
        return {"ok": False, "reason": "spark-activate failed", "exit_code": activate_step.exit_code}

    eval_step = ctx.runner(["evaluate", "--spark"], None)
    _log(ctx, "phase7_post_activate_eval", eval_step)
    write_artifact(ctx, "post_activate_eval.txt", eval_step.output)

    rollout: dict[str, object] = {
        "ok": eval_step.exit_code == 0,
        "generation": gen,
        "prior_generation": prior,
        "activated": True,
        "post_activate_eval_exit_code": eval_step.exit_code,
        "rollback_command": ["dec", "rag-plan", "--run-id", ctx.run_dir.name, "--force"],
    }
    if not rollout["ok"]:
        rollout["reason"] = "post-activation evaluation failed; rollback recommended"
    write_json_artifact(ctx, "rollout.json", rollout)
    return rollout


PHASE_RUNNERS: dict[int, Callable[[RunContext, Phase], dict[str, object]]] = {
    0: run_phase_preflight,
    1: run_phase_baseline,
    2: run_phase_chunk_audit,
    3: run_phase_contextual_index,
    4: run_phase_code_change,
    5: run_phase_code_change,
    6: run_phase_code_change,
    7: run_phase_rollout,
}


def _dec_command(*argv: str) -> list[str]:
    return [str(DEC), *argv]


def _planned_commands(ctx: RunContext, phase: Phase) -> list[dict[str, object]]:
    """Declarative command plan for a phase, used by ``--dry-run``."""
    if phase.requires_code_change:
        return [
            {
                "command": ["(code change required)"],
                "env": None,
                "note": (
                    "structured candidate provenance and retrieval-stage metrics are not "
                    "instrumented yet; implement before running this phase"
                ),
            }
        ]
    if phase.id == 0:
        commands: list[dict[str, object]] = [
            {"command": ["git", "status", "--short"], "env": None},
            {"command": _dec_command("status"), "env": None},
            {"command": _dec_command("spark-config-check"), "env": None},
        ]
        if ctx.active_generation:
            commands.append(
                {"command": _dec_command("spark-validate", "--generation", ctx.active_generation), "env": None}
            )
        return commands
    if phase.id == 1:
        return [
            {
                "command": _dec_command("evaluate", "--spark", "--output-dir", str(_artifact_path(ctx, "eval"))),
                "env": None,
            },
            {"command": _dec_command("evaluate", "--dataset", "tests/evaluation/eval_dataset.jsonl"), "env": None},
        ]
    if phase.id == 2:
        return [
            {
                "command": _dec_command("spark-manifest", "--output", str(_artifact_path(ctx, "manifest.json"))),
                "env": None,
            }
        ]
    if phase.id == 3:
        gen = (ctx.options.candidate_generation or "<candidate-generation>").strip()
        return [
            {"command": _dec_command("spark-build", "--generation", gen), "env": None},
            {"command": _dec_command("spark-validate", "--generation", gen), "env": None},
        ]
    if phase.id == 7:
        gen = (ctx.options.candidate_generation or "<candidate-generation>").strip()
        return [
            {"command": _dec_command("spark-activate", "--generation", gen), "env": {"FORCE": "1"}},
            {"command": _dec_command("evaluate", "--spark"), "env": None},
            {"command": _dec_command("spark-rollback", "--generation", gen), "env": {"FORCE": "1"}},
        ]
    return [{"command": ["(no commands)"], "env": None}]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_plan(options: PlanOptions) -> int:
    """Execute the plan phases and return a documented exit code."""
    run_id = (options.run_id or default_run_id()).strip()
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("artifacts", "logs", "trials"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(run_dir)
    completed = checkpoint.get("completed", {})
    if not isinstance(completed, dict):
        completed = {}
    ctx = RunContext(
        options=options,
        run_dir=run_dir,
        runner=options.runner,
        active_generation=str(checkpoint.get("active_generation", "") or ""),
        completed=completed,
    )
    if not ctx.active_generation:
        ctx.active_generation = _discover_active_generation()

    phases = PHASES_BY_ID.values()
    if options.phase is not None:
        if options.phase not in PHASES_BY_ID:
            print(f"Unknown phase {options.phase}; choose from {sorted(PHASES_BY_ID)}", file=sys.stderr)
            return EXIT_USAGE
        phases = [PHASES_BY_ID[options.phase]]

    phases_summary: dict[str, object] = {}
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "dry-run" if options.dry_run else "run",
        "active_generation": ctx.active_generation,
        "phases": phases_summary,
        "artifacts_dir": str(run_dir / "artifacts"),
        "failure": None,
    }

    exit_code = EXIT_OK
    for phase in phases:
        key = str(phase.id)
        summary_phase: dict[str, object] = {"id": phase.id, "name": phase.name}
        if key in completed and not options.force:
            summary_phase["status"] = "completed"
            phases_summary[key] = summary_phase
            continue

        if options.dry_run:
            summary_phase["status"] = "planned"
            summary_phase["planned"] = _planned_commands(ctx, phase)
            phases_summary[key] = summary_phase
            continue

        try:
            result = PHASE_RUNNERS[phase.id](ctx, phase)
        except Exception as exc:  # pragma: no cover - defensive
            write_failure(ctx, phase, f"phase-{phase.name}", f"unhandled exception: {exc}")
            summary_phase["status"] = "error"
            exit_code = EXIT_COMMAND
            phases_summary[key] = summary_phase
            summary["failure"] = f"unhandled exception: {exc}"
            break

        status = str(result.get("status", "done"))
        if status == "dry-run":
            summary_phase["status"] = "dry-run"
            summary_phase["planned"] = result.get("planned")
            phases_summary[key] = summary_phase
            continue
        if status == "skipped":
            summary_phase["status"] = "skipped"
            summary_phase["reason"] = result.get("reason")
            phases_summary[key] = summary_phase
            continue
        if status == "halted":
            summary_phase["status"] = "halted"
            summary_phase["reason"] = result.get("reason")
            phases_summary[key] = summary_phase
            summary["failure"] = result.get("reason")
            write_failure(ctx, phase, f"phase-{phase.name}", str(result.get("reason", "code change required")))
            exit_code = EXIT_HALTED
            break
        if status == "blocked":
            summary_phase["status"] = "blocked"
            summary_phase["reason"] = result.get("reason")
            phases_summary[key] = summary_phase
            summary["failure"] = result.get("reason")
            write_failure(ctx, phase, f"phase-{phase.name}", str(result.get("reason", "blocked")))
            exit_code = EXIT_BLOCKED
            break
        if result.get("ok") is not True:
            reason = str(result.get("reason", "phase gate failed"))
            step_result = result.get("step_result")
            if not isinstance(step_result, StepResult):
                step_result = None
            write_failure(ctx, phase, f"phase-{phase.name}", reason, step_result)
            summary_phase["status"] = "failed"
            summary_phase["reason"] = reason
            phases_summary[key] = summary_phase
            summary["failure"] = reason
            exit_code = EXIT_GATE
            break

        summary_phase["status"] = "completed"
        summary_phase["result"] = result
        phases_summary[key] = summary_phase
        ctx.completed[key] = result
        save_checkpoint(ctx)

    (run_dir / RESULT_FILENAME).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if not options.dry_run:
        save_checkpoint(ctx)

    if options.json_output:
        print(json.dumps(summary, indent=2, default=str))
    else:
        _print_summary(summary, phases_summary, run_dir)
    return exit_code


def _print_summary(summary: dict[str, object], phases_summary: dict[str, object], run_dir: Path) -> None:
    print(f"\nPlan run {summary['run_id']} [{summary['mode']}]")
    print(f"  active generation : {summary['active_generation'] or '(none)'}")
    print(f"  artifacts         : {summary['artifacts_dir']}")
    for key, phase in sorted(phases_summary.items(), key=lambda kv: int(kv[0])):
        phase_dict = phase if isinstance(phase, dict) else {}
        name = str(phase_dict.get("name", key))
        status = str(phase_dict.get("status", "?"))
        print(f"  phase {key} {name:<20} {status}")
    failure = summary.get("failure")
    if failure:
        print(f"  failure           : {failure}")
        print(f"  resume            : dec rag-plan --run-id {run_dir.name}")

    # echo the failure file path when a standard failure was written
    failure_path = run_dir / FAILURE_FILENAME
    if failure_path.exists():
        print(f"  failure schema    : {failure_path}")
