# 011: Context assembler — six-phase deterministic pipeline with drop provenance

Date: 2026-08-21 · Status: Accepted

## Context

Post-rerank context assembly previously did only Jaccard dedup + truncation.
Research (`docs/context_assembly_research.md`) identified silent-dedup
regressions, sibling fragmentation, lost-in-the-middle degradation, and
unobservable budget drops as the failure classes that matter at our corpus
sizes.

## Decision

`services/context_assembler.py::assemble` runs six ordered phases:

1. Content-hash dedup (SHA-256 of chunk text; `assembly_content_hash_dedup`).
2. Adjacent sibling merge by `parent_chunk_id` + `segment_index`
   (`assembly_enable_sibling_merge`).
3. Diversity: MMR **or** Jaccard>0.70 dedup — mutually exclusive
   (`assembly_mmr_enabled`, default False pending its benchmark gate).
4. Two-pass source-coverage budget: coverage pass places the top chunk of
   every distinct source URL first; depth pass fills remaining budget capped
   at `max_chunks_per_source`.
5. Lost-in-the-middle boustrophedon reorder (>3 chunks).
6. XML formatting with breadcrumbs (`assembly_breadcrumb_format`), `&`/`<`
   escaping (`prompt_xml_content_escape`), hard truncate at
   `item_limit_chars`.

Drops are recorded with machine-readable reasons
(`dropped_due_total_context_budget` / `dropped_due_per_source_cap`) surfaced
in Answer provenance.

## Consequences

- The assembler raises on oversized segments: ingestion's lossless splitter
  (`infrastructure/token_budget.py`) guarantees reconstruction, so an
  oversized segment is an invariant violation, not a truncation opportunity.
- When post-rerank compression runs (`context_compression_enabled`), the
  assembler's own dedup phase is suppressed to avoid double-dedup.
