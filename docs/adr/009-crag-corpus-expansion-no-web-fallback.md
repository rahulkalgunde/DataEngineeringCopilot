# 009: CRAG corrective loop expands within the corpus; no web-search fallback

Date: 2026-08-21 · Status: Accepted

## Context

CRAG (arXiv:2401.15884) prescribes a retrieval evaluator with three actions:
use the documents / discard and fall back to large-scale web search / use
both. Our pipeline implements the evaluator (`services/relevance_grader.py`):
chunks are LLM-graded for relevance, and when the grade falls below 0.5 the
pipeline re-retrieves once with `top_k*2` and fuses the expanded results back
(`async_rag.py::_relevance_guarded_chunks`). The web-search leg of CRAG is
absent.

The comparison against 2026 RAG best practices
(`docs/research/rag_best_practices_comparison_2026-08-21.md`, gap item 11)
flagged this as a divergence from CRAG-as-published.

## Decision

Keep correction offline: expand retrieval inside the indexed corpus only. Do
not add a web-search fallback.

Rationale:

1. The product contract is "answers from the ingested documentation only".
   Questions outside the corpus must produce evidence-based refusals (scope
   gate, low-confidence gate), not uncontrolled web content.
2. Web fallback would break provenance: every citation is verified against
   retrieved `source_name`s; web results have no place in that chain.
3. Cache scoping (`services/query_cache.py::scope_fingerprint`) keys answers
   to the index generation and corpus config; web-derived answers would be
   unkeyable.
4. Security posture: retrieved-content injection scanning, blocked-URL
   lists, and domain whitelists all assume a curated corpus. The open web
   multiplies the injection surface the spotlighting defenses must cover.

## Consequences

- Global/thematic questions outside the corpus are refused, not answered
  from web knowledge.
- If product requirements change, implement web fallback behind a dark flag
  with its own injection scan, citation policy, cache-scope treatment, and a
  benchmark gate — tracked in a future ADR.
