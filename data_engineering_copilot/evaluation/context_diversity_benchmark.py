"""Offline candidate-diversity benchmark for context assembly.

Locks the current deduplication/selection behavior (content-hash + same-parent
sibling collapse + lexical overlap dedup, then confidence ordering) and
compares it against the existing lexical MMR reranker on deterministic
synthetic chunk pools. Records, per strategy:

* duplicate rate (chunks whose text is ~identical to an earlier selected chunk)
* source coverage (distinct sources selected / distinct sources available)
* final context size (items and characters)
* groundedness (fraction of required facts present in the final context)

``ContextDiversityReport.passes()`` is the fixed gate from the plan: enable a
strategy only when it reduces duplicate rate by ``>=10%`` without source
coverage or groundedness regression. The plan's fixed decision keeps
content-hash dedup, sibling collapse, and lexical MMR; this module only
measures and never changes production routing. No LLM calls, no infra.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.context_assembler import ContextAssembler
from data_engineering_copilot.services.reranker import mmr_rerank

# Fixed gate thresholds (plan Task 10).
DUPLICATE_RATE_REDUCTION = 0.10  # candidate must cut duplicate rate by >= 10% relative
OVERLAP_DUPLICATE_THRESHOLD = 0.70  # token Jaccard above which two chunks are duplicates

DEFAULT_TOP_K = 6
DEFAULT_MMR_LAMBDA = 0.5

_CURRENT = "current"
_MMR = "mmr"


def _tokenize(text: str) -> frozenset[str]:
    """Lowercased alphanumeric token set (deterministic)."""
    return frozenset(word for word in text.lower().replace("\n", " ").split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class DiversityScenario:
    """A synthetic candidate pool with the facts that must survive selection."""

    id: str
    query: str
    chunks: tuple[RetrievedChunk, ...]
    required_facts: tuple[str, ...] = ()
    top_k: int = DEFAULT_TOP_K
    description: str = ""

    @property
    def expected_sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.chunk.source_name for c in self.chunks))


@dataclass(frozen=True)
class StrategyMetrics:
    """Metrics for one selection strategy on one scenario."""

    strategy: str
    scenario_id: str
    selected_items: int
    selected_chars: int
    duplicate_rate: float
    source_coverage: float
    groundedness: float


@dataclass(frozen=True)
class ScenarioResult:
    """Comparison of the two strategies on one scenario."""

    scenario_id: str
    description: str
    current: StrategyMetrics
    mmr: StrategyMetrics
    improvement: bool


@dataclass(frozen=True)
class ContextDiversityReport:
    """Aggregate report over all scenarios plus the fixed gate decision."""

    scenarios: tuple[ScenarioResult, ...]
    current_duplicate_rate: float
    mmr_duplicate_rate: float
    current_source_coverage: float
    mmr_source_coverage: float
    current_groundedness: float
    mmr_groundedness: float
    duplicate_rate_reduction: float
    gate_passed: bool
    decision: str
    notes: tuple[str, ...] = ()

    def passes(self) -> bool:
        """True when MMR beats the fixed gate: >=10% duplicate-rate reduction
        without source-coverage or groundedness regression."""
        return self.gate_passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_duplicate_rate": self.current_duplicate_rate,
            "mmr_duplicate_rate": self.mmr_duplicate_rate,
            "current_source_coverage": self.current_source_coverage,
            "mmr_source_coverage": self.mmr_source_coverage,
            "current_groundedness": self.current_groundedness,
            "mmr_groundedness": self.mmr_groundedness,
            "duplicate_rate_reduction": self.duplicate_rate_reduction,
            "gate_passed": self.gate_passed,
            "decision": self.decision,
            "notes": list(self.notes),
        }


def _select_current(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """The current production selection path (no reranker): sibling + overlap
    dedup via ContextAssembler, then confidence order, capped at top_k."""
    assembler = ContextAssembler(max_context_chars=10_000_000)
    deduped = assembler._deduplicate_chunks(sorted(chunks, key=lambda c: c.confidence, reverse=True))
    return deduped[:top_k]


def _select_mmr(chunks: list[RetrievedChunk], top_k: int, lambda_param: float) -> list[RetrievedChunk]:
    """Lexical MMR diversity selection."""
    return mmr_rerank(chunks, top_k=top_k, lambda_param=lambda_param)


def _measure(
    strategy: str, scenario_id: str, selected: list[RetrievedChunk], scenario: DiversityScenario
) -> StrategyMetrics:
    """Compute duplicate rate, source coverage, context size, groundedness."""
    selected_texts = [_tokenize(c.chunk.text) for c in selected]
    duplicate_count = 0
    seen: list[frozenset[str]] = []
    for tokens in selected_texts:
        if any(_jaccard(tokens, earlier) >= OVERLAP_DUPLICATE_THRESHOLD for earlier in seen):
            duplicate_count += 1
        seen.append(tokens)
    duplicate_rate = duplicate_count / len(selected) if selected else 0.0

    selected_sources = {c.chunk.source_name for c in selected}
    available_sources = set(scenario.expected_sources)
    source_coverage = len(selected_sources & available_sources) / len(available_sources) if available_sources else 0.0

    joined = " ".join(c.chunk.text for c in selected).lower()
    facts_hit = sum(1 for fact in scenario.required_facts if fact.lower() in joined)
    groundedness = facts_hit / len(scenario.required_facts) if scenario.required_facts else 1.0

    return StrategyMetrics(
        strategy=strategy,
        scenario_id=scenario_id,
        selected_items=len(selected),
        selected_chars=sum(len(c.chunk.text) for c in selected),
        duplicate_rate=round(duplicate_rate, 4),
        source_coverage=round(source_coverage, 4),
        groundedness=round(groundedness, 4),
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def run_context_diversity_benchmark(
    scenarios: list[DiversityScenario] | None = None,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> ContextDiversityReport:
    """Run both selection strategies on every scenario and apply the gate.

    Returns a report with the per-scenario comparison, aggregate metrics, and
    the enable/reject decision. Deterministic; no LLM or infra calls.
    """
    if scenarios is None:
        scenarios = default_scenarios()
    scenario_results: list[ScenarioResult] = []
    for scenario in scenarios:
        chunks = sorted(scenario.chunks, key=lambda c: c.confidence, reverse=True)
        current_selected = _select_current(chunks, scenario.top_k)
        mmr_selected = _select_mmr(chunks, scenario.top_k, mmr_lambda)
        current_metrics = _measure(_CURRENT, scenario.id, current_selected, scenario)
        mmr_metrics = _measure(_MMR, scenario.id, mmr_selected, scenario)
        scenario_results.append(
            ScenarioResult(
                scenario_id=scenario.id,
                description=scenario.description,
                current=current_metrics,
                mmr=mmr_metrics,
                improvement=mmr_metrics.groundedness >= current_metrics.groundedness
                and mmr_metrics.source_coverage >= current_metrics.source_coverage,
            )
        )

    current_dup = _mean([r.current.duplicate_rate for r in scenario_results])
    mmr_dup = _mean([r.mmr.duplicate_rate for r in scenario_results])
    current_cov = _mean([r.current.source_coverage for r in scenario_results])
    mmr_cov = _mean([r.mmr.source_coverage for r in scenario_results])
    current_ground = _mean([r.current.groundedness for r in scenario_results])
    mmr_ground = _mean([r.mmr.groundedness for r in scenario_results])

    reduction = (current_dup - mmr_dup) / current_dup if current_dup > 0 else 0.0
    reduction = round(reduction, 4)

    notes: list[str] = []
    gate_passed = False
    decision = "keep_current"
    if reduction >= DUPLICATE_RATE_REDUCTION:
        if mmr_cov < current_cov:
            notes.append("MMR regresses source coverage; rejected")
        elif mmr_ground < current_ground:
            notes.append("MMR regresses groundedness; rejected")
        else:
            gate_passed = True
            decision = "enable_mmr"
            notes.append("MMR reduces duplicate rate >= 10% without coverage/groundedness regression")
    else:
        notes.append("No strategy meets the duplicate-rate reduction gate; keep current behavior")

    return ContextDiversityReport(
        scenarios=tuple(scenario_results),
        current_duplicate_rate=current_dup,
        mmr_duplicate_rate=mmr_dup,
        current_source_coverage=current_cov,
        mmr_source_coverage=mmr_cov,
        current_groundedness=current_ground,
        mmr_groundedness=mmr_ground,
        duplicate_rate_reduction=reduction,
        gate_passed=gate_passed,
        decision=decision,
        notes=tuple(notes),
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_name: str,
    confidence: float,
    parent_chunk_id: str = "",
) -> RetrievedChunk:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        source_name=source_name,
        title=f"Title {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        text=text,
        content_hash=f"hash_{chunk_id}",
        parent_chunk_id=parent_chunk_id,
    )
    return RetrievedChunk(chunk=chunk, distance=1.0 - confidence, confidence=confidence)


def default_scenarios() -> list[DiversityScenario]:
    """Deterministic synthetic pools exercising sibling, near-duplicate, and
    lexical-overlap-with-distinct-facts patterns."""
    parent_text = (
        "window functions compute a result for each input row based on a window of other rows "
        "partitioned by partition columns and ordered within each partition"
    )
    scenarios: list[DiversityScenario] = []

    # 1) Same-parent siblings: five children of one parent with identical
    # (substituted) parent text, plus two distinct real chunks.
    sibling_chunks: list[RetrievedChunk] = []
    for i in range(5):
        sibling_chunks.append(
            _chunk(
                f"sib-{i}",
                parent_text,
                source_name="spark",
                confidence=0.95 - i * 0.01,
                parent_chunk_id="parent-window",
            )
        )
    sibling_chunks.append(
        _chunk(
            "real-1",
            "rowsBetween defines the window frame boundaries in Spark SQL",
            source_name="spark",
            confidence=0.9,
        )
    )
    sibling_chunks.append(
        _chunk(
            "real-2",
            "partitionBy groups rows into partitions before window evaluation",
            source_name="sql-guide",
            confidence=0.88,
        )
    )
    scenarios.append(
        DiversityScenario(
            id="sibling_collapse",
            query="how do window functions partition rows",
            chunks=tuple(sibling_chunks),
            required_facts=("rowsBetween", "partitionBy"),
            top_k=4,
            description="five identical same-parent siblings plus two distinct chunks",
        )
    )

    # 2) Near-duplicate copies: identical text from four different sources.
    copy_text = "spark.sql.shuffle.partitions controls the number of partitions for shuffled data"
    dup_chunks: list[RetrievedChunk] = []
    for i, source in enumerate(("spark", "sql-guide", "tuning", "databricks")):
        dup_chunks.append(_chunk(f"dup-{i}", copy_text, source_name=source, confidence=0.9 - i * 0.02))
    dup_chunks.append(
        _chunk(
            "distinct-1",
            "enable adaptive query execution with spark.sql.adaptive.enabled",
            source_name="spark",
            confidence=0.85,
        )
    )
    dup_chunks.append(
        _chunk(
            "distinct-2",
            "broadcast join hint broadcasts a small table to all executors",
            source_name="sql-guide",
            confidence=0.8,
        )
    )
    scenarios.append(
        DiversityScenario(
            id="near_duplicate_copies",
            query="how many partitions for shuffle",
            chunks=tuple(dup_chunks),
            required_facts=("spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled"),
            top_k=4,
            description="identical copy from four sources plus two distinct chunks",
        )
    )

    # 3) Lexically similar chunks carrying different required facts.
    similar_texts = (
        "filter a DataFrame using the isNotNull method to keep rows where a column has a value",
        "filter a DataFrame with isNotNull to drop rows whose column is null in the dataset",
        "aggregate rows by grouping on the partition key and summing the amount column",
        "aggregate rows grouped by the partition key while computing the sum of the amount column",
    )
    overlap_chunks = [
        _chunk("sim-0", similar_texts[0], source_name="spark", confidence=0.93),
        _chunk("sim-1", similar_texts[1], source_name="sql-guide", confidence=0.91),
        _chunk("sim-2", similar_texts[2], source_name="spark", confidence=0.89),
        _chunk("sim-3", similar_texts[3], source_name="sql-guide", confidence=0.87),
    ]
    scenarios.append(
        DiversityScenario(
            id="lexical_overlap_distinct_facts",
            query="filter null rows and aggregate by partition key",
            chunks=tuple(overlap_chunks),
            required_facts=("isNotNull", "amount"),
            top_k=4,
            description="pairs with high lexical overlap but distinct required facts",
        )
    )

    return scenarios


def _print_report(report: ContextDiversityReport) -> None:
    print(f"current_duplicate_rate={report.current_duplicate_rate} mmr_duplicate_rate={report.mmr_duplicate_rate}")
    print(f"current_source_coverage={report.current_source_coverage} mmr_source_coverage={report.mmr_source_coverage}")
    print(f"current_groundedness={report.current_groundedness} mmr_groundedness={report.mmr_groundedness}")
    print(f"duplicate_rate_reduction={report.duplicate_rate_reduction}")
    print(f"decision={report.decision}")
    for note in report.notes:
        print(f"  note: {note}")
    for result in report.scenarios:
        print(
            f"  {result.scenario_id}: current(dup={result.current.duplicate_rate}, "
            f"cov={result.current.source_coverage}, ground={result.current.groundedness}) "
            f"mmr(dup={result.mmr.duplicate_rate}, cov={result.mmr.source_coverage}, "
            f"ground={result.mmr.groundedness}) improvement={result.improvement}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline candidate-diversity benchmark")
    parser.add_argument(
        "--mmr-lambda", type=float, default=DEFAULT_MMR_LAMBDA, help="MMR relevance/diversity trade-off"
    )
    args = parser.parse_args(argv)
    report = run_context_diversity_benchmark(mmr_lambda=args.mmr_lambda)
    _print_report(report)
    return 0 if report.passes() else 1


if __name__ == "__main__":
    raise SystemExit(main())
