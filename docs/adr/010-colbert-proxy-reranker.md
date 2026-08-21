# 010: Lexical trigram proxy shipped under reranker_type="colbert"

Date: 2026-08-21 · Status: Accepted

## Context

The rerank optimization plan wanted a ColBERT-style option without adding a
PyTorch/transformers dependency to the CPU-only deployment. The shipped
implementation (`services/colbert_reranker.py`) scores via per-query-token
char-3gram MaxSim overlap — a deterministic lexical heuristic, not neural
late-interaction (no token embeddings, no PLAID-class optimizations).

## Decision

Ship the lexical proxy behind `reranker_type="colbert"` (value kept for
backward compatibility) but name the class honestly:
`LexicalNgramReranker`. Module and settings comments state it is a proxy,
not neural late-interaction.

## Consequences

- Operators must not expect PLAID/ColBERTv2 quality from this setting.
- True late-interaction (Qdrant multivectors + `hnsw_config(m=0)`) remains a
  deferred experiment gated on `dec eval-rerank` nDCG@K gains over both the
  cross-encoder and this proxy.
