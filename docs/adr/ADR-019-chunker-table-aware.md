# ADR-019: Table-Aware + Prose-Tail-Merge Chunking

- **Status:** Accepted (2026-09-06)
- **Applies to:** `HeaderAwareChunker` (`data_engineering_copilot/services/header_aware_chunker.py`)
- **Relation:** ADR-018 (`docs/adr/ADR-018-chunking-corpus-gate.md`); Section 7 of `plans/2026-09-06_10-16_chunking_corpus_adequacy_plan.md` (Task 3)

## Context

The 2026-09-06 chunking study (plus independent review) confirmed two defect classes in the
header-aware chunker:

1. **Mid-row HTML table cuts.** `_split_oversized_section` only treated fenced code blocks as
   atomic. Prose word windows (`_split_prose_chunk`) fell back to raw `str.split()` windows, so an
   oversized paragraph containing an HTML table was cut at an arbitrary word boundary — frequently
   **inside a `<tr>` row**. A row opened in window N and closed in window N+1; the markdown-to-HTML
   renderer then dropped the truncated half, so retrieval silently lost rows. Confirmed in the live
   corpus (security.md / configuration.md tables) and by code inspection (`_split_oversized_section`,
   pre-change, iterated only `_FENCE_RE`).
2. **Word-window nubs.** `_split_prose_chunk` emitted every `chunk_size_words` window unchanged, so
   a paragraph whose length was not a clean multiple produced a trailing 1–9 word nub. These feed
   the corpus `tiny_rate` gate (ADR-018) and carry near-zero retrieval value.

## Decision

1. Treat HTML `<table …>…</table>` blocks as atomic split units in `_split_oversized_section`,
   alongside fenced code. Oversized tables split greedily at `<tr>` row boundaries, never inside a
   row; the `<table>` opener stays with the first fragment and `</table>` with the last, so every
   emitted fragment is self-delimiting.
2. `_split_prose_chunk` gains a `tail_floor: int = 25` parameter: a trailing word window smaller
   than `tail_floor` words is merged into the previous window instead of emitted as a nub.

## Behavior Contract

- Splitting never opens a `<tr>` it does not close in the same chunk — for any chunk,
  `text.count("<tr>") == text.count("</tr>")`.
- Word-window splitting never emits a trailing chunk of 1–9 words (nub) — sub-`tail_floor` windows
  are absorbed.
- No mid-fence cuts (unchanged, verified by regression test).
- Overlapping span safety: a table nested inside a fenced block is processed once (overlap guard).

## Verification

- Unit: `TestTableAwareAndTailMerge` in `tests/unit/test_header_aware_chunker.py` (small-table
  intact, oversized-table row-boundary, no-fence-cut, prose-tail-merge).
- Cross-module: `test_chunking_factory`, `test_chunker_router`, `test_chunker_invariants`,
  `test_context_fragmentation_guards` — all green.
- Expected corpus-gate impact (ADR-018): table fractures go to zero for table-bearing pages; the
  `table_fracture` and (indirectly) `tiny_rate` gates tighten toward their thresholds.

## Trade-offs

- Table fragments packed at row granularity can land slightly above `chunk_size_words` when one row
  alone exceeds the budget; that oversized row stays whole (fail-safe) rather than being cut.
- Prose-tail merge can push one accumulated chunk to ~`chunk_size_words + tail_floor` words — still
  well under the `1.5×` oversized gate.

## Open

- `_split_prose_chunk` normalizes internal whitespace when windowing (`" ".join`); offset fidelity
  (Task 5.2) will derive emitted offsets from chunk text rather than raw spans.
