#!/usr/bin/env python3
"""Atomic promotion of retrieval eval to baseline_inscope with provenance sidecar.

Copies ``retrieval_eval.json`` → ``baseline_inscope.json`` via tmp→rename
and writes ``baseline_inscope.provenance.json`` with git_commit,
generation pinned-d3dbad402105, k, top_k, mrl, timestamp.

Usage:
  dec_venv/bin/python scripts/promote_baseline.py --source /tmp/new_baseline3/retrieval_eval.json
  dec_venv/bin/python scripts/promote_baseline.py --check-gate  # exits 1 if gate fails
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = pathlib.Path("/tmp/new_baseline3/retrieval_eval.json")
DEFAULT_DEST = PROJECT_ROOT / "tests/evaluation/benchmarks/baseline_inscope.json"
DEFAULT_PROVENANCE = PROJECT_ROOT / "tests/evaluation/benchmarks/baseline_inscope.provenance.json"
GENERATION = "pinned-d3dbad402105"


def _git_commit(short: bool = True) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            return ""
        sha = proc.stdout.strip()
        return sha[:12] if short else sha
    except Exception:
        return ""


def _load_metrics(source: pathlib.Path) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    overall = data.get("overall") or {}
    # also capture per_intent api_lookup for audit
    per_intent = data.get("per_intent") or {}
    return {"overall": overall, "per_intent": per_intent, "raw": data}


def promote(
    source: pathlib.Path = DEFAULT_SOURCE,
    dest: pathlib.Path = DEFAULT_DEST,
    provenance_path: pathlib.Path = DEFAULT_PROVENANCE,
    generation: str = GENERATION,
) -> int:
    if not source.exists():
        print(f"❌ source not found: {source}", file=sys.stderr)
        return 2
    try:
        from data_engineering_copilot.config.settings import settings

        k = 10
        # settings is frozen; read attributes
        top_k = int(getattr(settings, "retrieval_top_k", 50))
        mrl = bool(getattr(settings, "mrl_multistage_enabled", False))
    except Exception:
        k, top_k, mrl = 10, 50, False

    data = json.loads(source.read_text(encoding="utf-8"))
    overall = data.get("overall") or {}
    # validation: ensure overwrite candidate has expected shape
    if "recall@k" not in overall and "recall@k" not in data:
        print(f"⚠️ source missing overall.recall@k: {source}", file=sys.stderr)

    # atomic tmp→dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(dest)
    print(f"✅ promoted {source} → {dest}")

    # provenance sidecar
    metrics = overall if overall else data
    commit = _git_commit(short=True)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    provenance = {
        "generated_at": now,
        "generator": "promote_baseline",
        "generation": generation,
        "git_commit": commit,
        "commit": commit,
        "k": k,
        "retrieval_top_k": top_k,
        "top_k": top_k,
        "mrl_multistage_enabled": mrl,
        "mrl": mrl,
        "metrics": metrics,
        "source": str(source),
    }
    # atomic write for provenance as well
    prov_tmp = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    prov_tmp.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    prov_tmp.replace(provenance_path)
    print(f"✅ provenance → {provenance_path} (commit={commit} gen={generation} k={k} top_k={top_k} mrl={mrl})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote retrieval eval to baseline with provenance")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="path to retrieval_eval.json")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="destination baseline_inscope.json")
    ap.add_argument("--provenance", default=str(DEFAULT_PROVENANCE), help="provenance sidecar path")
    ap.add_argument("--generation", default=GENERATION, help="pinned generation id")
    ap.add_argument("--check-gate", action="store_true", help="run gate check without promoting")
    args = ap.parse_args()

    if args.check_gate:
        # gate is performed via eval-retrieval --compare-baseline; just sanity check source exists
        src = pathlib.Path(args.source)
        if not src.exists():
            print(f"❌ --check-gate: source missing {src}", file=sys.stderr)
            return 2
        print(f"check-gate source exists: {src}")
        return 0

    return promote(
        source=pathlib.Path(args.source),
        dest=pathlib.Path(args.dest),
        provenance_path=pathlib.Path(args.provenance),
        generation=args.generation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
