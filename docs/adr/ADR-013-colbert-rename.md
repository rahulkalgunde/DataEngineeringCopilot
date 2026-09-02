# ADR-013: Rename `colbert` → `lexical_ngram` Proxy (Docs == Code) — Not Neural

Date: 2026-09-02 · Status: Accepted

## Context

`services/colbert_reranker.py` ships a deterministic char-3gram MaxSim proxy
(`_char_ngram_overlap`) — no token embeddings, no PLAID/ColBERTv2 late interaction.
It was exposed as `reranker_type="colbert"` (see `010-colbert-proxy-reranker.md`
and `ADR-010-freeze-dark-flags.md:42`), which misleads operators into expecting
neural late-interaction quality and diverges from code reality. The plan
`plans/2026-09-02_rag_pipeline_simplification_plan.md:Task 4` requires docs == code:
rename the proxy and keep the old value as a deprecated alias.

## Decision

- Canonical name: `reranker_type="lexical_ngram"` — `LexicalNgramReranker`
  (Char-3gram MaxSim lexical proxy — NOT neural ColBERT).
- Alias: `reranker_type="colbert"` still works but emits `DeprecationWarning`
  (`colbert → lexical_ngram, not neural`) via `settings.py:_warn_colbert_alias`
  (model_validator after). No mapping — value stays `"colbert"` for back-compat;
  `factory.py` routes both `("colbert", "lexical_ngram")` to `LexicalNgramReranker`
  and logs `reranker_lexical_ngram_proxy_not_neural` (warning level for the alias).
- Domain enum: `RerankerType.LEXICAL_NGRAM = "lexical_ngram"`; `COLBERT` retained
  as deprecated alias.
- Module header `colbert_reranker.py:1` states `Char-3gram MaxSim lexical proxy — NOT neural ColBERT`
  with alias note and ADR-013 pointer.
- Docs `RAG_SYSTEM_LEARNER_GUIDE.md:836` reranker table uses `lexical_ngram` as
  canonical with a `colbert (deprecated alias)` row; config reference lists
  `cross_encoder | lexical_ngram | colbert | pylate_colbert`.

## Consequences

- New configs should use `lexical_ngram`; existing `colbert` configs keep working
  with a warning — no breaking change.
- `tests/unit/test_colbert_rename.py` gates alias + header + enum + factory routing.
- `pylate_colbert` (true neural via PyLate) stays separate and dark behind its
  own eval gate (`make eval-rerank` nDCG + latency).

## Alternatives Considered

- Hard rename without alias: rejected — breaks deployed `.env` / `RagConfig`.
- Keep `colbert` only: rejected — audit finding "talks fancy tech but not implement
  correctly" requires docs == code.

## Provenance

- Plan: `plans/2026-09-02_rag_pipeline_simplification_plan.md:Task 4`.
- Prior: `docs/adr/010-colbert-proxy-reranker.md`, `docs/adr/ADR-010-freeze-dark-flags.md`.
- Files: `services/colbert_reranker.py:1`, `config/settings.py:1268`, `domain/models.py:RerankerType`,
  `factory.py:2031`, `RAG_SYSTEM_LEARNER_GUIDE.md:836`.
