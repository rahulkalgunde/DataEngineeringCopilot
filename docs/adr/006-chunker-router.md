# 006: Metadata-based chunker router

## Status

Accepted (implemented 2026-08-18, commit 26f6c95)

## Context

The ingestion pipeline needs to choose the right chunker for each source (Spark docs → SparkChunker, Airflow/Delta → HeaderAwareChunker, Claude → HeaderAwareChunker with RST support, etc.). Previously the chunker was selected per-source in the factory, making it hard to add new sources without modifying the router.

## Decision

Add a `ChunkerRouter` that selects chunkers based on existing source metadata (source family, document type, content signals) at ingestion time. The routing priority:

1. **spark** — `SparkChunker` (full fidelity with SQL function registry)
2. **structured** — `StructuredChunker` (JSON/table structured data)
3. **code** — code-aware chunking for code-heavy content
4. **guide** — `HeaderAwareChunker` (RST/MDX heading support)
5. **generic** — fallback chunking

## Consequences

- New sources are routed by adding metadata entries to the router, not by modifying the factory.
- All existing chunkers are reused unchanged (no behavioral regression).
- The router is deterministic and explainable (no LLM-based classification).
- The `StructuredChunker` (Task 2) is only activated when source metadata signals structured data (JSON tables), keeping it isolated from prose-heavy sources.
