"""Unit tests for the RAG optimization benchmark module."""

from __future__ import annotations

import asyncio
import json

import pytest

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.evaluation.rag_optimization_benchmark import (
    _EXCLUDED_LLM_PROVIDERS,
    _build_benchmark_settings,
    compare_benchmarks,
    load_technical_queries,
    run_retrieval_benchmark,
)


def _row(id_: str, **overrides) -> dict:
    row = {
        "id": id_,
        "question": f"Question {id_}?",
        "expected_urls": [f"https://example.com/docs/{id_}.md"],
        "intent": "factual",
    }
    row.update(overrides)
    return row


def _write_rows(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


class TestLoadTechnicalQueries:
    def test_valid_rows(self, tmp_path):
        rows = [
            _row("a", intent="api_lookup"),
            _row("b", intent="code_example"),
            _row("c", intent="debugging"),
            _row("d", intent="factual"),
            _row("e", intent="how_to"),
        ]
        loaded = load_technical_queries(_write_rows(tmp_path, rows))
        assert [r["id"] for r in loaded] == ["a", "b", "c", "d", "e"]

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_technical_queries(tmp_path / "nope.jsonl")

    def test_missing_fields_rejected(self, tmp_path):
        rows = [
            _row("a", id=None),
            _row("b", question=""),
            _row("c", expected_urls=[]),
            _row("d", intent="bogus_intent"),
        ]
        with pytest.raises(ValueError) as exc:
            load_technical_queries(_write_rows(tmp_path, rows))
        message = str(exc.value)
        assert "id" in message
        assert "question" in message
        assert "expected_urls" in message
        assert "intent" in message

    def test_non_list_expected_urls_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="expected_urls"):
            load_technical_queries(_write_rows(tmp_path, [_row("a", expected_urls="https://x")]))

    def test_duplicate_ids_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate id"):
            load_technical_queries(_write_rows(tmp_path, [_row("a"), _row("a")]))

    def test_empty_file_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="No rows"):
            load_technical_queries(_write_rows(tmp_path, []))


class _FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        return "rewritten query"


class _FakeAnswer:
    def __init__(self, sources: list[DocumentChunk], context: str = "context") -> None:
        self.sources = tuple(sources)
        self.context = context


class _FakeService:
    def __init__(self, answers: dict[str, _FakeAnswer]) -> None:
        self.answers = answers
        self.llm_client = _FakeLLMClient()
        self.asked: list[str] = []

    async def answer(self, question, provenance=None, bypass_cache=False, expected_urls=None, retrieval_only=False):
        self.asked.append(question)
        await self.llm_client.generate(question)
        await self.llm_client.generate(question)
        return self.answers[question]


def _chunk(id_: str, url: str, *, hash_: str = "", source: str = "src") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=id_,
        source_name=source,
        title=f"Title {id_}",
        url=url,
        text=f"text {id_}",
        content_hash=hash_,
    )


class TestRunRetrievalBenchmark:
    def test_computes_recall_mrr_latency_context(self):
        url = "https://example.com/docs/window.md"
        service = _FakeService(
            {
                "Q1": _FakeAnswer([_chunk("c1", url), _chunk("c2", "https://example.com/other.md")]),
                "Q2": _FakeAnswer([_chunk("c3", "https://example.com/other.md")]),
            }
        )
        rows = [
            _row("q1", question="Q1", expected_urls=[url]),
            _row("q2", question="Q2", expected_urls=[url]),
        ]
        report = asyncio.run(run_retrieval_benchmark(service, rows))

        assert report["rows"] == 2
        assert report["rows_ok"] == 2
        assert report["rows_failed"] == 0
        # Q1 hits expected at rank 1 (MRR 1.0, recall 1.0); Q2 misses (0, 0).
        assert report["source_recall_mean"] == 0.5
        assert report["mrr_mean"] == 0.5
        q1 = next(r for r in report["results"] if r["id"] == "q1")
        assert q1["source_recall"] == 1.0
        assert q1["mrr"] == 1.0
        assert q1["final_context_size"] == len("context")
        assert q1["candidate_count"] == 2
        assert service.asked == ["Q1", "Q2"]

    def test_duplicate_rate_and_source_coverage(self):
        url = "https://example.com/docs/window.md"
        service = _FakeService(
            {
                "Q1": _FakeAnswer(
                    [
                        _chunk("c1", url, hash_="h1", source="a"),
                        _chunk("c2", url, hash_="h1", source="a"),
                        _chunk("c3", url, hash_="h2", source="b"),
                    ]
                )
            }
        )
        rows = [_row("q1", question="Q1", expected_urls=[url])]
        report = asyncio.run(run_retrieval_benchmark(service, rows))
        q1 = report["results"][0]
        assert q1["duplicate_rate"] == pytest.approx(0.3333, abs=1e-4)
        assert q1["source_coverage"] == 2
        assert report["duplicate_rate_mean"] == pytest.approx(0.3333, abs=1e-4)
        assert report["source_coverage_mean"] == 2

    def test_provider_call_count(self):
        url = "https://example.com/docs/window.md"
        service = _FakeService({"Q1": _FakeAnswer([_chunk("c1", url)])})
        rows = [_row("q1", question="Q1", expected_urls=[url], intent="api_lookup")]
        report = asyncio.run(run_retrieval_benchmark(service, rows))
        assert report["provider_calls_total"] == 2  # llm_client.generate called twice

    def test_identifier_vs_generic_recall_split(self):
        url = "https://example.com/docs/window.md"
        service = _FakeService(
            {
                "ID1": _FakeAnswer([_chunk("c1", url)]),
                "GEN1": _FakeAnswer([_chunk("c2", "https://example.com/other.md")]),
            }
        )
        rows = [
            _row("id1", question="ID1", expected_urls=[url], intent="api_lookup"),
            _row("gen1", question="GEN1", expected_urls=[url], intent="factual"),
        ]
        report = asyncio.run(run_retrieval_benchmark(service, rows))
        assert report["identifier_recall"] == 1.0
        assert report["generic_recall"] == 0.0

    def test_row_error_recorded_but_benchmark_continues(self):
        service = _FakeService({})

        async def _boom(question, provenance=None, bypass_cache=False, expected_urls=None, retrieval_only=False):
            raise RuntimeError("query failed")

        service.answer = _boom
        rows = [_row("q1", question="Q1"), _row("q2", question="Q2")]
        report = asyncio.run(run_retrieval_benchmark(service, rows))
        assert report["rows_failed"] == 2
        assert all("error" in r for r in report["results"])
        assert report["rows_ok"] == 0


class TestCompareBenchmarks:
    def _report(self, **overrides) -> dict:
        report = {
            "rows": 1,
            "rows_ok": 1,
            "source_recall_mean": 0.8,
            "mrr_mean": 0.6,
            "identifier_recall": 0.7,
            "generic_recall": 0.9,
            "provider_calls_total": 10,
            "duplicate_rate_mean": 0.2,
        }
        report.update(overrides)
        return report

    def test_all_gates_pass(self):
        baseline = self._report()
        candidate = self._report(
            source_recall_mean=0.82,
            mrr_mean=0.62,
            identifier_recall=0.80,
            generic_recall=0.89,
            provider_calls_total=6,
            duplicate_rate_mean=0.1,
        )
        cmp = compare_benchmarks(baseline, candidate)
        assert cmp["source_recall_delta"] == 0.02
        assert cmp["identifier_recall_delta"] == 0.1
        assert cmp["provider_calls_delta_pct"] == -40.0
        assert cmp["duplicate_rate_delta_pct"] == -50.0
        assert cmp["gates"] == {
            "recall_regression_ok": True,
            "mrr_regression_ok": True,
            "identifier_improved": True,
            "generic_recall_regression_ok": True,
            "provider_calls_reduced": True,
            "duplicate_rate_reduced": True,
        }

    def test_identifier_not_improved(self):
        baseline = self._report(identifier_recall=0.7)
        candidate = self._report(identifier_recall=0.72)
        cmp = compare_benchmarks(baseline, candidate)
        assert cmp["gates"]["identifier_improved"] is False

    def test_recall_regression_fails(self):
        baseline = self._report(source_recall_mean=0.8)
        candidate = self._report(source_recall_mean=0.78)
        cmp = compare_benchmarks(baseline, candidate)
        assert cmp["gates"]["recall_regression_ok"] is False

    def test_generic_recall_regression_fails(self):
        baseline = self._report(generic_recall=0.9)
        candidate = self._report(generic_recall=0.88)
        cmp = compare_benchmarks(baseline, candidate)
        assert cmp["gates"]["generic_recall_regression_ok"] is False

    def test_generic_recall_regression_within_tolerance_passes(self):
        baseline = self._report(generic_recall=0.9)
        candidate = self._report(generic_recall=0.89)
        cmp = compare_benchmarks(baseline, candidate)
        assert cmp["gates"]["generic_recall_regression_ok"] is True

    def test_provider_calls_not_reduced_enough(self):
        baseline = self._report(provider_calls_total=10)
        candidate = self._report(provider_calls_total=9)
        cmp = compare_benchmarks(baseline, candidate)
        assert cmp["gates"]["provider_calls_reduced"] is False


class TestExcludedProviders:
    def test_excluded_providers_constant(self):
        assert "opencodego" in _EXCLUDED_LLM_PROVIDERS
        assert "opencodezen" in _EXCLUDED_LLM_PROVIDERS

    def test_build_settings_strips_excluded_providers(self):
        settings = _build_benchmark_settings()
        assert all(p not in settings.llm_fallback_order for p in _EXCLUDED_LLM_PROVIDERS)
