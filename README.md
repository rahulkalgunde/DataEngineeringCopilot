# DataEngineeringCopilot

Question answering over data engineering documentation (Spark, Airflow, Databricks, Delta Lake, Claude docs). RAG pipeline built on Qdrant, Redis, Celery, and Streamlit — multi-provider LLM routing with Ollama as the always-available local fallback. No LangChain or LlamaIndex.

## Project Structure

```text
DataEngineeringCopilot/
  main.py                       # CLI entry point
  Makefile                      # Docker, test, lint, eval targets
  AGENTS.md                     # Agent guide & architecture
  data_engineering_copilot/
    config/
      settings.py               # All runtime settings (providers, limits, flags)
      documentation_sources.json
      logging.py, naming.py, pinned_sources.json, spark_sources.json, ...
    domain/
      models.py                 # Dataclasses shared across the app
      exceptions.py, protocols.py
    infrastructure/
      llm_client.py             # Unified OpenAI-compatible LLM client
      provider_fallback.py      # ProviderFallbackChain (all LLM/embedding calls)
      provider_selector.py, provider_health.py, provider_capabilities.py
      rate_limiter.py           # RPM/RPD coordination
      async_qdrant_store.py, async_crawler.py, async_embeddings.py
      async_rag_cache.py, embedding_cache.py, crawl_cache.py, crawl_db.py
      rerank_clients.py, pii_redactor.py, bm25_tokenizer.py, token_budget.py
    services/
      async_rag.py              # RAG pipeline orchestration
      query_rewriting.py        # Intent, decomposition, expansion, gated HyDE
      reranker.py, colbert_reranker.py, llm_reranker.py
      context_assembler.py, context_compression.py
      groundedness.py, scope_verifier.py, relevance_grader.py
      input_guardrails.py, output_guardrails.py, prompt_injection.py
      structured_output.py, prompt_builder.py
      async_ingestion.py, claude_docs_ingestion.py
      chunker.py, hierarchical_chunker.py, semantic_chunker.py, ...
    workers/
      celery_app.py             # Celery config & signal handlers
      tasks.py                  # Ingestion tasks with zombie recovery
    api/
      app.py, routes.py         # FastAPI backend
    ui/
      streamlit_app.py          # Chat UI
    observability/
      langfuse_client.py, otel_telemetry.py, structured_logging.py
    evaluation/                 # Eval harnesses (retrieval, generation, rerank, ...)
    profiler/
      cli.py, report_generator.py
    cli.py                      # CLI dispatcher
    factory.py                  # DI: build_rag_service(), fallback chains, ...
    utils/
      text.py
```

## Setup

# Package Management Constraints
- NEVER use standard 'pip' or 'python -m venv' commands.
- This project exclusively uses 'uv' as its Python package and environment manager.
- To create or manage virtual environments, use: `uv venv dec_venv`
- To install packages, use: `uv pip install -e ".[dev]"`
- To add a single package to the environment, use: `uv pip install <package_name>`
- Always ensure you target the correct local virtual environment binary path: `dec_venv/bin/python`

Linux/macOS:

```bash
uv venv dec_venv
source dec_venv/bin/activate
uv pip install -e ".[dev]"
```

Ollama models (pulled automatically by `make dev`; also via `make pull-models`):

- `phi4-mini:3.8b` — general LLM
- `qwen2.5-coder:7b` — code generation

No additional embedding model download is required — embeddings go through the Ollama HTTP API (or the configured cloud embedding provider).

## Configuration

Settings load from three `.env` files in order (later files override earlier):

1. `.env` — defaults (committed)
2. `.env.secrets` — sensitive keys (gitignored)
3. `.env.local` — personal overrides (gitignored)

### Models

- **Class defaults**: `llm_provider=ollama`, `llm_model=llama3.2:3b`, `embedding_provider=ollama` (`data_engineering_copilot/config/settings.py`). Ollama is the last-resort degraded fallback in every chain.
- **Answer purpose**: defaults to Groq (`answer_llm_provider=groq`, `answer_llm_model=openai/gpt-oss-20b`); rewrite/groundedness/intent/enrichment/evaluation/code also default to Groq with the same model for fastest free_forever latency.
- **LLM fallback chain** (`llm_fallback_order`): `groq → cerebras → nvidia → cloudflare → openrouter → gemini → agnes → ollama_cloud → ollama` — kept in sync with the live free_forever probe. Providers without configured keys are skipped, so with no API keys everything routes to Ollama.
- **Free_forever catalog** (`config/free_tier_models.json:1` → `data/provider_catalog.json:1`): curated 11 `$0-forever` models (not `$1 credit`), live-probed by `dec probe-catalog` (keeps fastest OK per provider, per-purpose via `services/provider_catalog.py:114`). When `CATALOG_AUTO_ORDER=true` (`.env:40`) the factory prefers that latency-sorted order over `LLM_FALLBACK_ORDER` (fail-open to static order when missing/stale). See `docs/provider_catalog.md` for inventory, ranking, and probe results (2026-08-27: 9 OK, groq 535ms fastest).
- **Embedding fallback chain**: nvidia → openrouter → huggingface → local-hf (local-hf is offline, always available).

All LLM/embedding/rerank calls route through `ProviderFallbackChain` (`infrastructure/provider_fallback.py:162`, health-scored, Redis-backed). See `docs/RAG_SYSTEM_LEARNER_GUIDE.md` for the full tour and `docs/provider_catalog.md` for the catalog.

## Docker

First-time setup (builds image, starts stack, pulls Ollama models):

```bash
make dev
```

Day-to-day:

```bash
make up              # Start all services (infra + app profile)
make down            # Stop all services
make status          # Containers, status, health checks
make logs            # Stream logs
make logs-worker     # Stream worker logs
make rebuild         # Rebuild app services (required after pyproject.toml/uv.lock changes)
make streamlit       # Run the Streamlit UI locally
```

Bare `docker compose up` starts **infrastructure only** — `backend-api` and `celery_worker` are gated behind the `app` profile. `make up` includes the profile.

Legacy aliases (`make docker-up`, `docker-down`, `docker-status`, `docker-logs`, ...) still exist and map to the targets above — see `docs/makefile_guide.md`.

### Services

- Infra: redis, qdrant, minio (+ minio-init), ollama, clickhouse, postgres (crawl frontier), langfuse, langfuse-postgres, langfuse-worker
- App profile: backend-api (FastAPI), celery_worker

## CLI

Run via `dec_venv/bin/dec` (or `python main.py`). Full reference with per-command infra requirements: `docs/cli_guide.md`.

```bash
# Core (in-process)
dec ask "How does Delta Lake time travel work?"
dec health / dec config / dec inspect-db / dec status

# Ingestion
dec ingest ...                # Celery path (needs API + worker + full stack)
dec ingest-claude-docs ...    # In-process

# Eval harnesses (in-process, frozen inputs)
dec eval-fast                 # Zero-LLM retrieval integrity check
dec eval-retrieval / eval-generation / eval-rerank / eval-assembly
dec eval-prompt-aug / eval-chunking / evaluate / eval-coverage

# Index generation lifecycle
dec gen-manifest → gen-build → gen-validate → gen-activate   # atomic alias switch
dec gen-rollback / gen-stale / gen-reset
# Spark-only mirror: spark-manifest / spark-render / spark-build / spark-activate ...

# Provider catalog (free_forever probes + smart fallback)
dec probe-catalog --json      # live probe 14 models → data/provider_catalog.json (fastest OK per provider)
dec probe-llm --json          # probe wired LLM + embedding providers
dec probe-catalog --offline   # skeleton, no network (CI)

# Reset granularity (coarse → fine)
dec reset-index               # Qdrant + BM25 + Redis + PG
dec reset-qdrant              # Collection + BM25
dec reset-crawler-db          # Redis/PG crawl state, keeps Qdrant
dec clear-cache [--query|--embedding|--crawl|--bm25|--all]
```

## RAG Pipeline at a Glance

Query path in `services/` (full tour: `docs/RAG_SYSTEM_LEARNER_GUIDE.md`):

- Two-tier query cache (exact + semantic) checked before any retrieval
- Query rewriting: intent classification, decomposition, expansion, gated HyDE
- Multi-query hybrid retrieval: dense + BM25 fused via Qdrant RRF
- Injection scan on retrieved context
- Reranking (cross-encoder / LLM / ColBERT)
- CRAG relevance gate on candidates
- Context assembly: dedup, sibling merge, MMR, source coverage, lost-in-the-middle reorder
- Schema-enforced structured generation (strict JSON for doc-intent answers)
- Guardrails: groundedness check, scope gate, citation verification, PII redaction
- Langfuse scoring/tracing on the way out

## Ingestion

The crawler downloads documentation pages and stores chunks in Qdrant (Celery path via `dec ingest`, in-process via `dec ingest-claude-docs`).

### When to Reset and Re-Ingest

Run `dec reset-index` before re-ingesting in these scenarios:

- **Switching embedding provider** (e.g. Ollama → OpenRouter) — different providers produce different vector dimensions
- **Changing embedding model** — models may use incompatible vector spaces
- **Changing chunking strategy** — new chunks have different boundaries than old ones
- **Corrupted or incomplete index** — if ingestion partially failed or Qdrant reports errors
- **Major documentation restructure** — if source URLs changed significantly

```bash
dec reset-index
dec ingest --max-pages 40
```

**Do NOT reset** for incremental ingestion — the system deduplicates via content hash and only updates changed pages.

### Switching Embedding Providers

Set these in `.env`:

```bash
# Switch from Ollama to OpenRouter
EMBEDDING_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_EMBEDDING_DIMENSION=2048
```

Then reset and re-ingest:

```bash
dec reset-index
dec ingest --max-pages 20
```

The configured documentation sources live in `data_engineering_copilot/config/documentation_sources.json` (Apache Spark, Apache Airflow, Databricks, Delta Lake).

## Rate Limiting

- **Sliding Window Rate Limiter**: Shared `SlidingWindowRateLimiter` coordinates RPM and RPD between embeddings and LLM clients. Configured per-provider in `settings.py` (OpenRouter: 18 RPM / 900 RPD; NVIDIA NIM: 36 RPM; defaults for all other providers are in `settings.py`).
- **429 Handling**: LLM calls are fail-fast and failover-first — no same-provider retries, no circuit breaker. `Retry-After` is parsed into a category-based provider cooldown and the adaptive router fails over to the next provider in `llm_fallback_order`, ending at Ollama. The rate limiter acts as a non-blocking pre-flight gate so an over-limit provider is skipped without a paid API call. Embeddings still block on the limiter and retry transient errors.

## Ingestion & Workers

- **Worker Time Limits**: Soft limit=36000s (10h), Hard limit=43200s (12h) (`workers/celery_app.py`)
- **Zombie Task Recovery**: Celery `task_failure`/`task_revoked` signal handlers catch hard time limit kills and mark Redis status as FAILED
- **Qdrant Batch Splitting**: `upsert_chunks` splits into 256-chunk sub-batches to prevent 32MB payload limit errors
- **Ingestion Lock**: Released before flush to prevent unbounded chunk accumulation

## Architecture

This project intentionally does not use LangChain or LlamaIndex (except `langchain-text-splitters`).

- `config`: source URLs, runtime settings, logging
- `domain`: dataclasses, exceptions, and protocols shared by the app
- `infrastructure`: adapters for crawling, HTML parsing, embeddings, Qdrant, Redis, BM25, rerank clients, PII redaction, and provider fallback/health/rate limiting
- `services`: the RAG pipeline (retrieval, rewriting, reranking, context assembly, guardrails, structured output) plus ingestion workflows
- `workers`: Celery tasks with zombie recovery
- `api`: FastAPI backend (auth, routes, middleware)
- `observability`: Langfuse tracing/scoring, OpenTelemetry, structured logging, token tracking
- `evaluation`: eval harnesses (retrieval, generation, rerank, assembly, chunking, prompt augmentation)
- `profiler`: performance profiling and reporting
- `cli`: command-line interface dispatcher
- `ui`: Streamlit chat interface

Dependency injection goes through `factory.py` (`build_rag_service()`, `build_llm_fallback_chain()`, ...) — never hand-instantiate services.

Local generation can take time on CPU. The timeout and generation limits are configured in `data_engineering_copilot/config/settings.py` as `ollama_timeout_seconds`, `ollama_num_ctx`, `ollama_num_predict`, `retrieval_top_k`, and `max_context_chars`.

If Ollama fails due to prompt or output length, the service automatically retries with reduced repository context and then with a larger output budget. Tune with `ollama_retry_context_ratio`, `ollama_retry_extra_num_predict`, and `ollama_retry_max_num_predict` in the same settings file.

Runtime logs are written under `logs/` in the project workspace:

- `logs/app.log` captures CLI, Streamlit, ingestion, retrieval, vector store, and Ollama events for troubleshooting.
- `logs/ingestion_refresh.log` captures detailed UI refresh events and fetched documentation URLs.
