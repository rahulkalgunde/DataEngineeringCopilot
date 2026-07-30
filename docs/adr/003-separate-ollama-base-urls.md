# ADR-003: Separate Embedding and LLM Ollama Base URLs

**Date:** 2026-07-30  
**Status:** Accepted  
**Deciders:** Architecture Team  

## Context

Ollama can serve embeddings (`/api/embed`) and text generation (`/v1/chat/completions`) on the same or different ports. Using the same base URL for both creates coupling and prevents independent scaling or routing.

## Decision

Add two configuration settings in `settings.py`:

- `embedding_ollama_base_url` — default: `http://localhost:11434`
- `llm_ollama_base_url` — default: `http://localhost:11434`

The `factory.py` `_build_embedder` function routes to `embedding_ollama_base_url` and `_build_purpose_llm_client` routes to `llm_ollama_base_url`. Both default to the same address for backward compatibility.

## Consequences

- Positive: Users can run embedding and LLM models on separate Ollama instances
- Positive: Independent scaling — LLM instance can be GPU-backed, embedding instance CPU-only
- Negative: Adds two config keys; users migrating must review their `.env` files
