# 005: Namespace-aware BM25 tokenizer

## Status

Rejected (gate failed 2026-08-18)

## Context

Technical queries (dotted identifiers like `pyspark.sql.functions.col`, version-qualified lookups like `Spark 4.0`, path-like strings like `sql/catalyst/expressions`) are matched by the default BM25 tokenizer as ordinary word tokens. This loses the information that `pyspark.sql.functions.col` is a single API identifier, fragmenting it into `pyspark`, `sql`, `functions`, `col` — which overlap with many unrelated prose chunks.

A namespace-aware tokenizer was proposed that:
1. Preserves full dotted/path identifiers as single tokens.
2. Emits safe sub-components (so `pyspark.sql.functions` still matches prose containing `pyspark`).
3. Uses versioned persistence (`namespace-v1`) with a store-level mismatch guard.

## Decision

Build and validate a candidate generation with the namespace-v1 tokenizer (same nvidia nemotron-3-embed-1b embeddings as the active generation). Benchmark identifier recall on `technical_queries.jsonl` (20 rows: 12 identifier-intent, 8 generic).

**Result**: identifier recall 0.3333 → 0.3333 (delta +0.0, gate requires ≥ +0.05). The namespace tokenizer did not improve retrieval of technical identifiers on this dataset. The candidate generation was rolled back and deleted.

## Consequences

- `namespace_bm25_enabled` defaults to `False` and remains off.
- The `BM25Tokenizer` retains the `namespace-v1` code path and versioned persistence (tested, committed), but it is not active in production.
- A future attempt could evaluate on a larger, more identifier-heavy dataset, or combine namespace BM25 with weighted RRF (identifier_sparse_rrf) which also showed no recall improvement on its own.
- No production disruption: the active generation is unchanged.
