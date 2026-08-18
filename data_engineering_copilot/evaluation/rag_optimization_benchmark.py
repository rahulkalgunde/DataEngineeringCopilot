"""Retrieval benchmark for RAG-pipeline optimization decisions.

Measures retrieval quality end-to-end over the real retrieval path (rewrite,
expansion, HyDE, fusion, rerank, compression) using ``retrieval_only`` mode so
no answer-generation LLM is called. Each row is scored on source recall, MRR,
latency, provider-call count, duplicate rate, source coverage and final context
size. ``compare_benchmarks`` applies the fixed numeric gates from
``plans/2026-08-18_07-55_rag_pipeline_optimization_plan.md``.

The module is hermetic-testable: ``run_retrieval_benchmark`` accepts any object
exposing an async ``answer()`` method (a real ``AsyncRagService`` or a fake).
Provider-call counting installs a thin counter over the LLM clients reachable
from the service (``llm_client``, ``code_llm_client``, ``evaluation_llm_client``
and the query rewriter's clients) and is skipped when none exist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Intent groups used by the fixed gates.
_IDENTIFIER_INTENTS = frozenset({"api_lookup", "code_example", "debugging"})
_GENERIC_INTENTS = frozenset({"factual", "how_to"})
_VALID_INTENTS = frozenset(_IDENTIFIER_INTENTS | _GENERIC_INTENTS)

# Fixed gates (absolute deltas / relative reductions).
RECALL_REGRESSION_LIMIT = 0.01
MRR_REGRESSION_LIMIT = 0.02
IDENTIFIER_RECALL_IMPROVEMENT = 0.05
PROVIDER_CALL_REDUCTION_PCT = 20.0
DUPLICATE_RATE_REDUCTION_PCT = 10.0

# Provider names excluded from live LLM fallback chains built by this module
# (credits budget). Kept out of the primary order entirely.
_EXCLUDED_LLM_PROVIDERS = ("opencodego", "opencodezen")


def _norm_url(url: str) -> str:
    return str(url).strip().rstrip("/")


def load_technical_queries(path: str | Path) -> list[dict[str, object]]:
    """Read a technical-query JSONL file and validate its schema.

    Every row must contain ``id``, ``question``, ``expected_urls`` (a list of
    strings) and ``intent`` (one of ``api_lookup``, ``code_example``,
    ``debugging``, ``factual``, ``how_to``). IDs must be unique. Raises
    ``ValueError`` describing every schema violation.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Technical query dataset not found: {path}")

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    errors: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{lineno}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path.name}:{lineno}: row is not a JSON object")
                continue

            row_id = row.get("id")
            question = row.get("question")
            expected_urls = row.get("expected_urls")
            intent = row.get("intent")

            if not isinstance(row_id, str) or not row_id.strip():
                errors.append(f"{path.name}:{lineno}: missing or non-string 'id'")
            elif row_id in seen_ids:
                errors.append(f"{path.name}:{lineno}: duplicate id {row_id!r}")
            else:
                seen_ids.add(row_id)

            if not isinstance(question, str) or not question.strip():
                errors.append(f"{path.name}:{lineno}: missing or empty 'question'")
            if not isinstance(expected_urls, list) or not expected_urls:
                errors.append(f"{path.name}:{lineno}: 'expected_urls' must be a non-empty list")
            elif not all(isinstance(u, str) and u.strip() for u in expected_urls):
                errors.append(f"{path.name}:{lineno}: 'expected_urls' must contain only non-empty strings")
            if not isinstance(intent, str) or intent not in _VALID_INTENTS:
                errors.append(f"{path.name}:{lineno}: unknown 'intent' {intent!r}")

            rows.append(row)

    if errors:
        raise ValueError("; ".join(errors[:20]))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _install_call_counter(client: Any) -> dict[str, int] | None:
    """Wrap ``client.generate`` with a counter; return the counter dict or None."""
    if client is None or not hasattr(client, "generate"):
        return None
    counter = {"calls": 0}
    original = client.generate

    async def counted(prompt: str, *args: Any, **kwargs: Any) -> str:
        counter["calls"] += 1
        return await original(prompt, *args, **kwargs)

    client.generate = counted  # type: ignore[attr-defined]
    return counter


def _iter_llm_clients(service: Any):
    """Yield the LLM clients reachable from a RAG service (deduplicated)."""
    candidates: list[Any] = []
    for attr in ("llm_client", "code_llm_client", "evaluation_llm_client"):
        candidates.append(getattr(service, attr, None))
    rewriter = getattr(service, "query_rewriter", None)
    if rewriter is not None:
        candidates.append(getattr(rewriter, "_llm_client", None))
        candidates.append(getattr(rewriter, "_intent_llm_client", None))
    seen: set[int] = set()
    for client in candidates:
        if client is not None and id(client) not in seen:
            seen.add(id(client))
            yield client


def _row_metrics(
    question: str, expected_urls: list[str], answer: Any, elapsed_ms: float, calls: int
) -> dict[str, object]:
    """Compute per-row metrics from an ``Answer``-like object."""
    sources = list(getattr(answer, "sources", None) or [])
    urls = [_norm_url(getattr(c, "url", "")) for c in sources]
    expected = {_norm_url(u) for u in expected_urls}

    hit = sum(1 for u in expected if u in urls)
    source_recall = hit / len(expected) if expected else 1.0
    mrr = 0.0
    for rank, u in enumerate(urls, 1):
        if u in expected:
            mrr = 1.0 / rank
            break

    content_hashes = [getattr(c, "content_hash", "") or "" for c in sources]
    unique_hashes = {h for h in content_hashes if h}
    duplicate_rate = (len(content_hashes) - len(unique_hashes)) / len(content_hashes) if content_hashes else 0.0
    source_coverage = len({getattr(c, "source_name", "") or "" for c in sources})

    context = getattr(answer, "context", None) or ""
    final_context_size = len(context)

    return {
        "source_recall": round(source_recall, 4),
        "mrr": round(mrr, 4),
        "latency_ms": round(elapsed_ms, 1),
        "provider_calls": calls,
        "candidate_count": len(sources),
        "duplicate_rate": round(duplicate_rate, 4),
        "source_coverage": source_coverage,
        "final_context_size": final_context_size,
    }


async def run_retrieval_benchmark(service: Any, rows: list[dict[str, object]]) -> dict[str, object]:
    """Run the retrieval benchmark over ``rows`` against ``service``.

    ``service`` must expose an async ``answer(question, provenance=...,
    bypass_cache=..., expected_urls=..., retrieval_only=...)`` method that
    returns an ``Answer``-like object. Returns a JSON-serializable report with
    per-row results and aggregates (including identifier vs generic recall).
    """
    counters = [_install_call_counter(client) for client in _iter_llm_clients(service)]
    results: list[dict[str, object]] = []
    started = time.monotonic()

    for row in rows:
        question = str(row.get("question") or "")
        expected_raw = row.get("expected_urls")
        expected_urls = [str(u) for u in expected_raw] if isinstance(expected_raw, list) else []
        prov: list[dict] = []
        t0 = time.monotonic()
        calls_before = sum(c["calls"] for c in counters if c is not None)
        row_result: dict[str, object] = {
            "id": row.get("id", ""),
            "intent": row.get("intent", ""),
            "question": question,
        }
        try:
            answer = await service.answer(
                question,
                provenance=prov,
                bypass_cache=True,
                expected_urls=expected_urls,
                retrieval_only=True,
            )
            elapsed = (time.monotonic() - t0) * 1000.0
            calls = sum(c["calls"] for c in counters if c is not None) - calls_before
            row_result.update(_row_metrics(question, expected_urls, answer, elapsed, calls))
        except Exception as exc:  # noqa: BLE001 - record per-row failures, keep going
            row_result["error"] = str(exc)
            row_result["source_recall"] = 0.0
            row_result["mrr"] = 0.0
            row_result["latency_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
            row_result["provider_calls"] = sum(c["calls"] for c in counters if c is not None) - calls_before
            row_result["candidate_count"] = 0
            row_result["duplicate_rate"] = 0.0
            row_result["source_coverage"] = 0
            row_result["final_context_size"] = 0
        results.append(row_result)

    total_ms = (time.monotonic() - started) * 1000.0
    provider_calls_total = sum(c["calls"] for c in counters if c is not None)

    def _mean(rows_subset: list[dict[str, object]], key: str) -> float | None:
        values: list[float] = []
        for r in rows_subset:
            if "error" in r:
                continue
            value = r.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    ok_results = [r for r in results if "error" not in r]
    identifier_rows = [r for r in ok_results if r.get("intent") in _IDENTIFIER_INTENTS]
    generic_rows = [r for r in ok_results if r.get("intent") in _GENERIC_INTENTS]

    return {
        "rows": len(rows),
        "rows_ok": len(ok_results),
        "rows_failed": len(results) - len(ok_results),
        "total_ms": round(total_ms, 1),
        "provider_calls_total": provider_calls_total,
        "source_recall_mean": _mean(ok_results, "source_recall"),
        "mrr_mean": _mean(ok_results, "mrr"),
        "latency_ms_mean": _mean(ok_results, "latency_ms"),
        "duplicate_rate_mean": _mean(ok_results, "duplicate_rate"),
        "source_coverage_mean": _mean(ok_results, "source_coverage"),
        "final_context_size_mean": _mean(ok_results, "final_context_size"),
        "identifier_recall": _mean(identifier_rows, "source_recall"),
        "generic_recall": _mean(generic_rows, "source_recall"),
        "results": results,
    }


def compare_benchmarks(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    """Compare a candidate benchmark against the baseline with the fixed gates.

    Deltas are candidate-minus-baseline (positive = better for recall/MRR,
    negative for provider calls and duplicate rate). Gate booleans implement
    the plan's fixed thresholds exactly.
    """

    def _delta(key: str) -> float | None:
        b = baseline.get(key)
        c = candidate.get(key)
        if not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
            return None
        return round(float(c) - float(b), 4)

    source_recall_delta = _delta("source_recall_mean")
    mrr_delta = _delta("mrr_mean")
    identifier_delta = _delta("identifier_recall")
    generic_delta = _delta("generic_recall")

    def _pct_delta(key: str) -> float | None:
        b = baseline.get(key)
        c = candidate.get(key)
        if not isinstance(b, (int, float)) or not isinstance(c, (int, float)) or b == 0:
            return None
        return round(((float(c) - float(b)) / float(b)) * 100.0, 1)

    provider_calls_delta_pct = _pct_delta("provider_calls_total")
    duplicate_rate_delta_pct = _pct_delta("duplicate_rate_mean")

    return {
        "source_recall_delta": source_recall_delta,
        "mrr_delta": mrr_delta,
        "identifier_recall_delta": identifier_delta,
        "generic_recall_delta": generic_delta,
        "provider_calls_delta_pct": provider_calls_delta_pct,
        "duplicate_rate_delta_pct": duplicate_rate_delta_pct,
        "gates": {
            "recall_regression_ok": source_recall_delta is not None and source_recall_delta >= -RECALL_REGRESSION_LIMIT,
            "mrr_regression_ok": mrr_delta is not None and mrr_delta >= -MRR_REGRESSION_LIMIT,
            "identifier_improved": identifier_delta is not None and identifier_delta >= IDENTIFIER_RECALL_IMPROVEMENT,
            "provider_calls_reduced": (
                provider_calls_delta_pct is not None and provider_calls_delta_pct <= -PROVIDER_CALL_REDUCTION_PCT
            ),
            "duplicate_rate_reduced": (
                duplicate_rate_delta_pct is not None and duplicate_rate_delta_pct <= -DUPLICATE_RATE_REDUCTION_PCT
            ),
        },
    }


def _build_benchmark_settings():
    """Build ``AppSettings`` with excluded LLM providers removed from the fallback order."""
    from data_engineering_copilot.config.settings import AppSettings

    base = AppSettings(skip_provider_check=True)
    order = [p for p in base.llm_fallback_order if str(p).lower() not in _EXCLUDED_LLM_PROVIDERS]
    settings = AppSettings(llm_fallback_order=order, skip_provider_check=True)
    settings.validate_all()
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m ... --output <path> [--generation GEN]``."""
    parser = argparse.ArgumentParser(prog="rag-optimization-benchmark")
    parser.add_argument("--output", required=True, help="Path to write the benchmark report JSON")
    parser.add_argument("--generation", default=None, help="Active generation (default: resolved active generation)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Technical query dataset (default: tests/evaluation/technical_queries.jsonl)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from data_engineering_copilot.config.settings import resolve_active_generation
    from data_engineering_copilot.evaluation.provenance import config_fingerprint, eval_environment
    from data_engineering_copilot.factory import build_rag_service

    project_root = Path(__file__).resolve().parents[2]
    dataset = Path(args.dataset) if args.dataset else project_root / "tests" / "evaluation" / "technical_queries.jsonl"

    try:
        rows = load_technical_queries(dataset)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 2

    settings = _build_benchmark_settings()
    gen = args.generation or resolve_active_generation()

    def _make_env():
        env = eval_environment(settings)
        env["generation"] = gen
        return env

    try:
        service = build_rag_service(app_settings=settings)
    except Exception as exc:  # noqa: BLE001 - surface infra/config failures cleanly
        print(f"❌ Failed to build RAG service: {exc}")
        return 2

    report = asyncio.run(run_retrieval_benchmark(service, rows))
    report["generation"] = gen
    report["provenance"] = {**_make_env(), "config_fingerprint": config_fingerprint(settings)}
    report["dataset"] = str(dataset)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Benchmark complete: {report['rows']} rows, {report['rows_ok']} ok, {report['rows_failed']} failed")
    print(f"  source_recall={report.get('source_recall_mean')} mrr={report.get('mrr_mean')}")
    print(f"  identifier_recall={report.get('identifier_recall')} generic_recall={report.get('generic_recall')}")
    print(
        f"  provider_calls_total={report.get('provider_calls_total')} duplicate_rate={report.get('duplicate_rate_mean')}"
    )
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
