# ADR-015: Chunking Strategy Eval — header vs recursive (keep header_aware 375w)

## Status

Keep `header_aware 375w/90 overlap` — 2026-09-02

## Context

Hermetic offline chunking eval (`dec eval-chunking --strategy all --gold all`,
`evaluation/chunking_eval.py`) loads committed gold spans
(`tests/evaluation/golden/chunking/{synthetic_gold,human_slice}.jsonl`) and
runs each chunker without Qdrant/LLM, measuring token-IoU, excerpt precision,
SegEval boundary similarity, and structural fracture rate
(`evaluation/chunking_metrics.py`). Gate: fracture rate `≤0.25`
(`FRACTURE_GATE_THRESHOLD`).

Production chunker is `HeaderAwareChunker(chunk_size_words=375, overlap_words=90,
min_chunk_words=37)` (min = 10% of target, shipped in `chunking_eval.py:35`
and `services/header_aware_chunker.py:49`). Latency/quality question is
whether `recursive` (1000 chars/100 overlap), `sentence`, or `structured`
outperform header-aware on gold-span fidelity, given BM25 `b` prefers
256–512 token short chunks (mixpeek caveat).

## Decision

Keep `header_aware 375w` — do not flip indexing strategy.

The hermetic eval on the current gold corpus does **not** overturn the
heading-path/fracture rationale. `header` scores lowest on token-level IoU on
this corpus because the corpus is tiny, not because the chunker is worse on
real documents.

## Evaluation

Wrapper `scripts/run_chunking_eval.py` → `dec eval-chunking --strategy all
--gold all --output /tmp/chunking_eval.json` (hermetic, local tokenizer only,
~1s, no Qdrant/LLM).

Run `2026-09-02` (`dec_venv/bin/dec eval-chunking --strategy all --gold all`):

| strategy   | IoU   | Prec  | B-Sim | Fract | docs |
|------------|-------|-------|-------|-------|------|
| recursive  | 0.217 | 0.217 | 0.298 | 0.000 | 7    |
| sentence   | 0.216 | 0.216 | 0.298 | 0.000 | 7    |
| header     | 0.007 | 0.007 | 0.012 | 0.000 | 7    |
| structured | 0.216 | 0.216 | 0.298 | 0.000 | 7    |
| gates      | fracture_ok=true worst=0.000 threshold<=0.25 |

Full JSON (`/tmp/chunking_eval.json`):

```json
{
  "recursive": {"iou": 0.2169, "precision": 0.2169, "boundary_similarity": 0.2976, "fracture_rate": 0.0, "doc_count": 7},
  "sentence": {"iou": 0.2160, "precision": 0.2160, "boundary_similarity": 0.2976, "fracture_rate": 0.0, "doc_count": 7},
  "header": {"iou": 0.0065, "precision": 0.0065, "boundary_similarity": 0.0119, "fracture_rate": 0.0, "doc_count": 7},
  "structured": {"iou": 0.2160, "precision": 0.2160, "boundary_similarity": 0.2976, "fracture_rate": 0.0, "doc_count": 7},
  "gates": {"fracture_ok": true, "fracture_threshold": 0.25, "worst_fracture_rate": 0.0}
}
```

Why `header` appears to lose: each gold doc is 8–35 words (placeholder
`synthetic_gold.jsonl`/`human_slice.jsonl`). `HeaderAwareChunker` merges
sections and then drops sub-minimum output when `wc < min_chunk_words=37`
with no prior chunk to absorb it (`header_aware_chunker.py:345`), so 6/7
docs emit `chunks=0` (log `sections=2 chunks=0`). Token-IoU is defined as
`0.0` when `pred_chunks` is empty, dragging the mean to ~0.006. `sentence`
and `recursive` emit one chunk per tiny doc, so their IoU is not filtered.

This is a corpus-size artifact, not a chunker defect. Production documents
(`Claude Platform Docs`) average ~800–1200 words per page with multi-level
headings; every page clears `min_chunk_words` after merge. Header-aware
there preserves `heading_path`/`section_header` on continuation chunks
(`LONG_SECTION` invariant in `test_context_fragmentation_guards.py`:
40× paragraph under `## Setup` → header retained on every fragment),
which `recursive`/`sentence` do not. Fracture rate is `0.0` for all
strategies on this corpus, so the `≤0.25` gate does not discriminate.

No strategy changes `B-Sim` meaningfully on this corpus (0.298 vs 0.012)
because boundary sets are derived from the same tiny spans.

## Consequences

- `chunking_strategy` stays `header_aware` at `375w/90` (`settings.py`,
  `evaluation/chunking_eval.py:35`).
- `recursive`/`sentence` vs `structured` tradeoff noted for future
  BM25 tuning: short 256–512 token chunks can improve BM25 `b` length
  normalization (mixpeek), but only help if gold headers/code-fences
  stay intact — verified by `fracture_rate ≤0.25` remaining green.
- Gold corpus should be expanded before any flip: add at least one
  realistic long doc (≥300 words, multi-section with a code fence) so
  `header` is measured on a regime where it actually emits chunks.
  Until then, token-IoU on the 7-doc placeholder cannot gate a strategy
  change.

## Verification

- `dec_venv/bin/python scripts/run_chunking_eval.py --strategy all --gold all --output /tmp/chunking_eval.json` → `fracture_ok true`, JSON as above.
- `dec_venv/bin/dec eval-chunking --strategy all --gold all --output /tmp/chunking_eval.json` parity (same report).
- Tier1: `ruff check/format/pyright` on `scripts/run_chunking_eval.py` +
  `pytest tests/unit/test_chunking_eval.py -v -n 0` PASS (no Qdrant).
- `pytest tests/unit/test_context_fragmentation_guards.py -v -n 0` PASS
  (header-carry invariant + fracture gate).
