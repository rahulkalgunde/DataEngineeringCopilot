# `dec` CLI Guide

The `dec` command-line utility drives the Data Engineering Copilot from a terminal. This guide documents every subcommand, option, exit code, and gotcha, with realistic examples.

---

## Table of Contents

- [Overview](#overview)
- [Commands](#commands)
  - [dec ingest](#dec-ingest)
  - [dec ingest-claude-docs](#dec-ingest-claude-docs)
  - [dec ask](#dec-ask)
  - [dec reenrich](#dec-reenrich)
  - [dec retry-failed](#dec-retry-failed)
  - [dec unskip](#dec-unskip)
  - [dec reset-index](#dec-reset-index)
  - [dec reset-qdrant](#dec-reset-qdrant)
  - [dec reset-crawler-db](#dec-reset-crawler-db)
  - [dec clear-query-cache](#dec-clear-query-cache)
  - [dec clear-cache](#dec-clear-cache)
  - [dec spark-config-check](#dec-spark-config-check)
  - [dec spark-manifest](#dec-spark-manifest)
  - [dec spark-render](#dec-spark-render)
  - [dec spark-build](#dec-spark-build)
  - [dec spark-validate](#dec-spark-validate)
  - [dec spark-activate](#dec-spark-activate)
  - [dec spark-rollback](#dec-spark-rollback)
  - [dec gen-config-check](#dec-gen-config-check)
  - [dec gen-manifest](#dec-gen-manifest)
  - [dec gen-build](#dec-gen-build)
  - [dec gen-validate](#dec-gen-validate)
  - [dec gen-activate](#dec-gen-activate)
  - [dec gen-rollback](#dec-gen-rollback)
  - [dec gen-reset](#dec-gen-reset)
  - [dec gen-stale](#dec-gen-stale)
  - [dec ui](#dec-ui)
  - [dec profile](#dec-profile)
  - [dec health](#dec-health)
  - [dec status](#dec-status)
  - [dec evaluate](#dec-evaluate)
  - [dec eval-coverage](#dec-eval-coverage)
  - [dec gen-synthetic-eval](#dec-gen-synthetic-eval)
  - [dec rag-plan](#dec-rag-plan)
  - [dec config](#dec-config)
  - [dec langfuse-seed-prompts](#dec-langfuse-seed-prompts)
  - [dec langfuse-evaluate](#dec-langfuse-evaluate)
  - [dec langfuse-seed-score-configs](#dec-langfuse-seed-score-configs)
  - [dec langfuse-metrics](#dec-langfuse-metrics)
  - [dec langfuse-review-queue](#dec-langfuse-review-queue)
  - [dec inspect-db](#dec-inspect-db)
  - [dec cancel](#dec-cancel)
  - [dec ingestion-monitor](#dec-ingestion-monitor)
  - [dec probe-llm](#dec-probe-llm)
  - [dec version (not implemented)](#dec-version-not-implemented)
- [Cheat sheet](#cheat-sheet)
- [Common workflows](#common-workflows)

---

## Overview

`dec` is the console entry point defined in `pyproject.toml` (`dec = "main:main"`), backed by `data_engineering_copilot/cli.py`. You can also invoke it explicitly with the venv launcher:

```bash
dec <command> [options]
dec_venv/bin/dec <command> [options]
```

All commands read configuration from `.env` → `.env.secrets` → `.env.local` (later files win). Run `dec --help` for the full command list.

### Infrastructure requirements by command

| Command(s) | Needs API on `:8000` + Celery worker | Needs Qdrant | Needs Redis | Needs PG (`CRAWL_DB_URL`) | Needs Ollama/embedder/LLM |
|---|---|---|---|---|---|
| `ingest`, `cancel`, `ingestion-monitor`, `profile` | Yes | Yes (via worker) | Yes | Yes (crawl) | Yes |
| `ask`, `evaluate` | No (direct RAG) | Yes | Yes | No | Yes |
| `reenrich`, `retry-failed`, `unskip` | No (in-process, direct ingestion) | Yes | Yes | Yes | Yes |
| `reset-index` | No | Yes | Yes | If set | No |
| `reset-qdrant` | No | Yes | No | No | No |
| `clear-query-cache` | No | No | Yes | No | No |
| `clear-cache` | No | No | Yes* | No | No |

\* `dec clear-cache --bm25` needs only disk access (no Redis).
| `spark-config-check` | No | No | No | No | No |
| `spark-manifest` | No | No | No | No | No |
| `spark-render` | No | No | No | No | No |
| `spark-build` | No | Yes | No | No | Yes (embedder) |
| `spark-validate` | No | Yes | No | No | No |
| `spark-activate` / `spark-rollback` | No | Yes | No | No | No |
| `gen-config-check` | No | No | No | No | No |
| `gen-manifest` | No | No | No | No | No |
| `gen-build` | No | Yes | No | No | Yes (embedder) |
| `gen-validate` | No | Yes | No | No | No |
| `gen-activate` / `gen-rollback` | No | Yes | No | No | No |
| `gen-reset` | No | Yes | No | No | No |
| `gen-stale` | No | Yes | No | No | No |
| `health`, `status`, `config`, `inspect-db` | `status` checks workers | Yes (except health/config degrade gracefully) | Yes | Yes (status) | Yes (health) |

> `dec ingest`, `dec cancel`, and `dec ingestion-monitor` talk to the FastAPI server at `http://localhost:8000`. Start it with `make dev` (first time) or `make up` (subsequent).

---

## Commands

### `dec ingest`

Crawl documentation sources and build the Qdrant vector index. Dispatches a Celery task through the production API and then polls Redis progress.

```
usage: dec ingest [-h] [--max-pages MAX_PAGES] [--source SOURCE]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--max-pages N` | int | `None` → `settings.max_pages_per_source` (100,000) | Maximum pages to crawl **per source**. Hard cap: `settings.max_pages_hard_cap`. |
| `--source "Name"` | str (repeatable) | all sources | Documentation source name. Repeat to select multiple. |

**Examples**

```bash
# Ingest all configured sources (unbounded pages)
dec ingest

# Ingest only Spark with a 50-page cap per source
dec ingest --source "Apache Spark Documentation" --max-pages 50

# Ingest two specific sources
dec ingest --source "Apache Spark Documentation" --source "Databricks Documentation"

# Ingest all sources with a small page budget (quick smoke run)
dec ingest --max-pages 5
```

**Behavior**
- POSTs `{"source_names": [...], "max_pages": N}` to `POST /api/v1/ingest` (Celery dispatch via a 60s `SETNX` lock; concurrent runs are rejected).
- Prints `Dispatched ingestion task <task_id>`, then polls `GET /api/v1/ingest/status/<task_id>` every 2s, printing status transitions:
  ```
  Status: DISPATCHED | Pages: 0 | Chunks: 0
  Status: PROCESSING | Pages: 12 | Chunks: 0
  ```
- **Resilient polling**: transient poll failures (timeouts, network errors, HTTP 5xx) are retried up to 3 times with backoff (`2s/4s/8s`), so a single slow status response does not kill the CLI while the task keeps running. Only after retries are exhausted does it give up — printing a pointer to `dec ingestion-monitor --task-id <id>` and `dec cancel <id>` instead of failing with a traceback.
- Terminates when status is `COMPLETED`, `FAILED`, or `CANCELLED`.
- **Ctrl-C** attempts to cancel the running task via `POST /api/v1/ingest/<task_id>/cancel` and exits with code `130`. If the cancel call fails it prints a manual curl fallback:
  ```bash
  curl -X POST http://localhost:8000/api/v1/ingest/<task_id>/cancel
  ```

**Exit codes**: `0` success; `1` dispatch failure (HTTP error, unreachable API); `130` SIGINT.

**Gotchas**
- If the API is unreachable it instructs you to start `backend-api` + `celery_worker`.
- URL dedup is handled by the `AsyncUrlRegistry` in Redis; re-ingesting mostly skips cached pages (`pages_skipped`).

---

### `dec ingest-claude-docs`

Ingest Anthropic Claude documentation (the Claude platform and API "LLMs docs") directly into the Qdrant vector index — no Celery, no crawler/Redis/Postgres. Downloads the markdown sources in-process (via `httpx`), strips YAML frontmatter, then runs them through the same `HeaderAwareChunker` + embedding fallback chain + `AsyncQdrantVectorStore` as the standard ingestion pipeline.

```
usage: dec ingest-claude-docs [-h] [--site {platform,code,all}] [--max-docs MAX_DOCS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--site` | `platform` \| `code` \| `all` | `all` | Which site(s) to ingest. `platform` = `platform.claude.com` LLM docs; `code` = `code.claude.com` docs. |
| `--max-docs N` | int | `None` (all) | Cap the number of `.md` docs ingested **per site** (useful for a quick smoke run). |

**Sources indexed** (routed to Qdrant with `doc_type` `guide`/`api_reference` per file path):
- `Claude Platform Documentation` — `https://platform.claude.com/docs/en/llms*.md`, `url_prefix` `build-with-claude/`.
- `Claude Code Docs` — `https://code.claude.com/docs/llms*.md`, `url_prefix` matching Claude Code markdown pages.

**Examples**

```bash
# Ingest ALL Claude docs (platform + code)
dec ingest-claude-docs --site all

# Quick smoke run: only the platform site, capped at 5 pages
dec ingest-claude-docs --site platform --max-docs 5

# Only Claude Code docs
dec ingest-claude-docs --site code
```

**Note on re-ingesting over an existing index**: this command upserts into the existing collection and does **not** rebuild already-stored chunks. If the embedding model or its `input_type` mode changed (see the note under [`dec reset-index`](#dec-reset-index)), run `dec reset-index`/`dec reset-qdrant` first so the whole corpus is re-embedded in the correct mode — otherwise old chunks stay in the incompatible subspace.

**Behavior**
- Pulls the site's `llms*.md` index (an index of markdown links), then downloads each `.md` page to a cache dir under `settings.data_dir/claude_docs/<site>/` (skips files already downloaded). Pages under ~100 chars of text are skipped.
- Chunks with `HeaderAwareChunker` and upserts with deterministic chunk IDs, so re-running is idempotent (re-upsert overwrites the same IDs — no duplicates).
- Prints a summary on completion: per-source document counts, chunked documents, total chunks uploaded, and any fetch failures.
- Does **not** require the crawler stack (Redis/Postgres/celery_worker) — only Qdrant + an available embedder.

**Exit codes**: `0` success; `1` on error (unreachable Qdrant, missing embedder, download failures).

**Gotchas**
- Requires the LLM docs index to be reachable; the CLI warns per-site on download failures and continues with the other site.
- Weighty pages are chunked identically to regular ingestion, so queries against this data use the same routers (`--claude-only`/`--sources` in `dec ask`).

---

### `dec ask`

Ask a question against the indexed documentation. Builds the full RAG service in-process (no API needed).

```
usage: dec ask [-h] [--user-id USER_ID] [--session-id SESSION_ID] question
```

| Argument | Description |
|---|---|
| `question` | The question to answer (quote it). |
| `--user-id` | User identifier recorded on the Langfuse trace. |
| `--session-id` | Session identifier recorded on the Langfuse trace. |

**Examples**

```bash
dec ask "What is a Spark DataFrame?"
dec ask "How do I schedule DAGs in Airflow?"
dec ask "Explain Delta Lake time travel"
dec ask --user-id=u1 --session-id=s1 "What is a Spark DataFrame?"
```

**Behavior** — runs the RAG pipeline: query rewriting → vector retrieval → reranking → context assembly → LLM → groundedness verification. Prints:

```
<answer text>

Sources:
- <source title>: <url>
- ...

Confidence: 0.87
```

**Gotchas**
- Requires Qdrant, Redis, the embedding provider, and a reachable LLM chain (or the fallback chain to Ollama).
- An empty Ollama response usually means the output budget was exhausted (see `OLLAMA_NUM_PREDICT`).
- When `--user-id`/`--session-id` are omitted and Langfuse tracing is enabled, the trace is recorded without user/session context.

---

### Conversational RAG (Chat)

Multi-turn conversational RAG is available through the API and the Streamlit UI
(no dedicated CLI command). The system keeps a conversation history per session
and uses it to rewrite follow-up questions into standalone queries and to ground
every turn in freshly retrieved context.

**API**

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat` | SSE streamed chat turn. Body: `{"message": ..., "session_id": ...}`. |
| `GET` | `/api/v1/sessions` | List sessions for the caller (`X-User-ID`, anonymous fallback). |
| `GET` | `/api/v1/sessions/{session_id}` | Full message thread for a session. |
| `DELETE` | `/api/v1/sessions/{session_id}` | Delete a session and its messages. |

`POST /api/v1/chat` emits `data: {...}` SSE events (`session_created`,
`status`, `sources`, `token`, `done`, `error`) followed by `data: [DONE]`.
Send `X-User-ID: <user>` to scope sessions; omit `session_id` on the first turn
and reuse the returned `session_id` on follow-ups.

**Behavior** — each turn runs: history-aware query rewriting (turn 1 skips
rewriting) → multi-query retrieval (expansion + HyDE + rank fusion) →
reranking → history-injected prompt → **true token streaming** (the fallback
chain now streams from the first available provider instead of collapsing to a
single chunk). Conversation history is cached in Redis (hot) and durably
persisted to Postgres (`chat_sessions`/`chat_messages` tables in the
`crawl_frontier` database; DSN from `chat_db_url`, defaulting to
`crawl_db_url`).

**Speed features**
- **True streaming**: `ProviderFallbackChain.generate_stream` walks providers
  (same gates/cooldown/degraded logic as `execute`) and yields tokens as they
  arrive; a provider that fails before the first token falls through, a failure
  after tokens are emitted surfaces as an `error` event (no silent truncation).
- **Turn-1 cache**: the first turn (empty history, standalone question) reads
  and writes the shared two-tier answer cache — exact and semantic repeat
  questions return instantly (measured 0.34s vs ~7s full pipeline). Follow-up
  turns never touch the cache, so context-dependent answers are never stale.
- **Local reranking** (`chat_rerank_local`, default true): chat reranks with the
  local `BAAI/bge-reranker-v2-m3` cross-encoder instead of the ~5s cloud LLM
  rerank chain. The single-turn Ask pipeline is unaffected.
- **Smart-cache recall tier** (`chat_cache_recall_enabled`, default **off**):
  on follow-up turns, similar cached (question→answer) pairs are retrieved via
  embedding similarity and re-synthesized by the local model, gated by the
  scope verifier; falls through to the full pipeline on any failure. Enable
  only after real usage has populated the cache, then monitor `cache.hit_rate`.

**Streamlit UI** — the `💬 Chat` tab provides a ChatGPT-style interface:
`st.chat_message` bubbles with avatars, `st.chat_input` (with a Stop button via
`submit_mode="stop"`), live `st.status` pipeline progress, token-by-token
streaming, a sidebar session manager (New/Delete + session list), per-turn
sources, and stable anonymous user scoping (`anon-<uuid>` sent as
`X-User-ID`).

**Settings** — session/storage: `chat_enabled`, `chat_session_ttl_seconds`
(default 72h), `chat_history_max_turns` (default 10), `chat_history_max_tokens`
(default 2048), `chat_db_url`, `chat_title_max_chars` (default 60). Speed:
`chat_rerank_local` (default true), `chat_rewrite_local` / `chat_scope_local`
(default **false** — CPU-only Ollama is ~6 tok/s, so routing medium-length
rewrites/HyDE to it is slower than cloud; enable only on GPU-backed Ollama),
`chat_answer_local` (default false), `chat_cache_recall_enabled` (default
false), `chat_cache_top_k` (3), `chat_cache_recall_threshold` (0.70),
`chat_cache_max_age_seconds` (86400). See `config/settings.py`.

**Cache toggles** — per-type enable/disable (all default **on**):
`query_cache_enabled` (master switch for the two-tier RAG query cache),
`query_cache_exact_enabled` / `query_cache_semantic_enabled` (individual
tiers), `embedding_cache_enabled` (embedder L1/L2 cache), and
`crawl_cache_enabled` (crawler HTTP-header cache). When a cache is disabled
the corresponding store is neither read nor written; the crawler falls back to
a no-op cache that always re-fetches. `dec clear-cache` clears the stores for
all of them.

**Gotchas**
- Requires Qdrant, Redis, Postgres, the embedding provider, and a reachable LLM chain.
- Follow-up turns never read or write the shared single-turn answer cache, so
  context-dependent follow-ups are never served stale single-turn answers.
- Streaming falls back gracefully: the answer purpose resolves to a single
  provider client for true per-token streaming; if all providers fail before
  the first token, the chain degrades to the next provider / error event.

---

### `dec reenrich`

Re-enrich pages whose contextual enrichment previously failed (summaries generated by `LLMContextSummarizer`). It clears the vector-store chunks and the Redis URL-registry entry for each URL, resets the frontier rows to `DISCOVERED`, then re-runs ingestion for the source in-process (no API/Celery needed).

```
usage: dec reenrich [-h] --source SOURCE [--urls URLS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--source "Name"` | str (required) | — | Documentation source to re-enrich. |
| `--urls FILE` | path | Redis set | File with one URL per line (`#` comments allowed). Defaults to the Redis set `ingest:enrichment_failed:<source>` written by the pipeline's enrichment failure recorder. |

**Examples**

```bash
# Re-enrich from the Redis failure tracker (populated by the pipeline)
dec reenrich --source "Apache Spark Documentation"

# Re-enrich from a URL file (e.g. plans/reingest_failed_enrichment_2026-08-01.txt)
dec reenrich --source "Apache Spark Documentation" --urls plans/reingest_failed_enrichment_2026-08-01.txt
```

**Behavior**
1. Resolves the URL set (file takes precedence over the Redis set).
2. For each URL: `delete_by_url` on Qdrant (removes existing chunks and the stored content hash so dedup won't skip it), `HDEL crawl:url_registry:<source>` in Redis (so the crawler re-fetches instead of 304-skipping), and `requeue_urls` on the PostgreSQL frontier (state → `DISCOVERED`, attempts reset).
3. Runs `ingest(source_names=[source])` in-process with a large page cap; the crawler drains the requeued `DISCOVERED` rows and re-enriches under the current router settings.

**Exit codes**: `0` success (even if 0 URLs were requeued — prints a note); `1` on errors (e.g. missing `CRAWL_DB_URL`, missing URLs, ingestion failure).

**Gotchas**
- Requires `CRAWL_DB_URL` (the crawler frontier is PostgreSQL-backed).
- Re-enriching URLs that were only *indexed without a summary* is safe — their chunks are deleted first, so no duplicates are created.
- The failure recorder only writes URLs after all transient retries are exhausted, so the Redis set only contains docs that genuinely failed enrichment.
- Use `--category all` to retry all failure types (fetch, embed, upsert, enrichment).

---

### `dec retry-failed`

Retry all FAILED pages for a source, optionally filtered by failure category. This is the general-purpose recovery command for any ingestion failure.

```
usage: dec retry-failed [-h] --source SOURCE [--category {fetch,embed,upsert,all}]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--source "Name"` | str (required) | — | Documentation source to retry. |
| `--category` | choice | all | Filter by failure category: `fetch` (HTTP errors), `embed` (embedding failures), `upsert` (vector store failures), `all` (everything). |

**Examples**

```bash
# Retry all failed pages
dec retry-failed --source "Apache Spark Documentation"

# Retry only fetch failures (HTTP errors, timeouts)
dec retry-failed --source "Apache Spark Documentation" --category fetch

# Retry only embedding failures
dec retry-failed --source "Apache Spark Documentation" --category embed
```

**Behavior**
1. Queries PostgreSQL for all FAILED URLs matching the category filter.
2. For each URL: deletes Qdrant chunks, clears Redis URL registry, resets frontier to `DISCOVERED`.
3. Re-runs ingestion for the source.

**When to use**: After an ingestion run where some pages failed due to transient errors (Ollama overload, network issues, etc.). This retries ALL failed pages, not just enrichment failures.

---

### `dec unskip`

Re-process SKIPPED pages for a source. SKIPPED pages are those where parsing returned no readable content (e.g., navigation-only pages, empty pages).

```
usage: dec unskip [-h] --source SOURCE
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--source "Name"` | str (required) | — | Documentation source to unskip. |

**Examples**

```bash
# Re-process all skipped pages
dec unskip --source "Apache Spark Documentation"
```

**Behavior**
1. Queries PostgreSQL for all SKIPPED URLs for the source.
2. For each URL: deletes Qdrant chunks (in case some were partially indexed), clears Redis URL registry, resets frontier to `DISCOVERED`.
3. Re-runs ingestion for the source.

**When to use**: After improving the parser or content extractor, re-process pages that were previously skipped due to no readable content.

---

### `dec reset-index`

Full clean rebuild. Recreates the Qdrant collection with the current dimension + hybrid config, deletes the persisted BM25 cache, clears Redis `crawl:*` keys, and drops the PostgreSQL crawl-frontier tables.

```
usage: dec reset-index [-h]
```

**Destructive scope** (all wiped):
1. Qdrant collection (`data_engineering_docs`) recreated with `dense` (+ `sparse` when hybrid) vectors.
2. Persisted BM25 tokenizer cache (`.bm25_cache/<collection>.json`).
3. Redis keys matching `crawl:*` (URL registry + HTTP conditional-GET cache).
4. PostgreSQL `crawl_frontier` and `sitemap_edges` tables — only if `CRAWL_DB_URL` is set.

**Example**

```bash
dec reset-index
```

**Behavior**
- Deletes then recreates the collection first, so a failure aborts before any frontier history is dropped.
- Recreating with a **different embedding provider/model** than the original collection is the main reason to reset (dimension mismatch).

**Note on embedding mode (`input_type`)** — dual-mode embedding models (e.g. `nemotron-3-embed-1b`) encode passages and queries in different, non-interchangeable subspaces:
- Index-time chunks are embedded with `input_type="passage"`; live search prompts with `input_type="query"`.
- **Existing chunks ingested before this mode was wired are in the wrong/mixed subspace** and will not match query-mode vectors. After a `reset-index` (or a full re-ingest), the index is rebuilt in the correct mode automatically.
- Re-ingesting **over an existing collection** (e.g. `dec ingest-claude-docs --site code` without `reset-index`) only *adds/overwrites* points — it does **not** fix already-stored chunks that were embedded in the old mode. Use `reset-index` (or `reset-qdrant`) first to rebuild the corpus, then re-ingest.
- After switching to `input_type`, also **clear the embedding query cache** (Redis + in-process) so stale pre-change query embeddings are not served for new passage-mode chunks.

**Gotchas**
- Destructive and irreversible for the index/crawl state. Run `dec inspect-db` first if unsure.
- If Redis is down, crawl-key clearing is skipped (logged at debug) — the rest still runs.
- If `CRAWL_DB_URL` is not set, the PG step is skipped with a warning.

---

### `dec reset-qdrant`

Lighter than `reset-index`: only recreates the Qdrant collection and deletes the BM25 cache. Does **not** touch Redis or PostgreSQL.

```
usage: dec reset-qdrant [-h]
```

**Example**

```bash
dec reset-qdrant
```

**Behavior**
- Prints `Deleted collection ...` and `Created collection ... (dim=<N>, hybrid=<bool>)`.
- If the collection does not exist, prints "nothing to reset" (404 handled) and recreates it anyway.

**Gotchas**
- Safe for preserving crawl frontier state, but the index must be rebuilt afterward with `dec ingest`.

---

### `dec reset-crawler-db`

Clears crawler state without touching Qdrant. Resets Redis `crawl:*` keys (URL registry + HTTP cache) and PostgreSQL frontier tables (`crawl_frontier` + `sitemap_edges`). Qdrant is preserved so the dedup mechanism (`content_hash` in Qdrant payloads) still works — re-crawled pages with unchanged content are skipped.

```
usage: dec reset-crawler-db [-h]
```

**Example**

```bash
dec reset-crawler-db
```

**Behavior**
1. Clears Redis keys: `crawl:url_registry:*`, `crawl:<hash>`, `ingest:enrichment_failed:*`
2. Drops PostgreSQL tables: `crawl_frontier`, `sitemap_edges` (recreated on next ingestion)
3. Preserves Qdrant vector store and BM25 cache

**What gets cleared**

| Store | What | Keys/Tables |
|-------|------|-------------|
| Redis | URL registry | `crawl:url_registry:<source>` |
| Redis | HTTP cache | `crawl:<url_hash>` |
| Redis | Enrichment failures | `ingest:enrichment_failed:<source>` |
| PostgreSQL | URL frontier | `crawl_frontier` |
| PostgreSQL | Link graph | `sitemap_edges` |

**What is preserved**
- Qdrant vector store (all indexed chunks + content hashes)
- BM25 tokenizer cache
- RAG query cache

**When to use**: When failed/errored pages were not tracked and you need to re-crawl everything. The dedup at Qdrant store level handles duplicates — re-crawled pages with unchanged content are automatically skipped.

**Gotchas**
- Requires `CRAWL_DB_URL` to be set for PostgreSQL reset
- Safe to run multiple times (idempotent)
- After reset, run `dec ingest --source <name>` to re-crawl

---

### `dec clear-query-cache`

Clears the RAG query cache (both exact-match and semantic tiers) without touching the vector index, BM25 cache, or crawler state.

```
usage: dec clear-query-cache [-h]
```

**Example**

```bash
dec clear-query-cache
```

**Behavior**
1. Deletes all Redis `rag:cache:*` keys:
   - `rag:cache:exact:<fingerprint>:<sha256>` — exact-match tier
   - `rag:cache:semantic:<fingerprint>:<id>` — semantic-similarity tier
   - `rag:cache:semantic:counter` — semantic id counter
2. Preserves Qdrant, BM25 cache, and `crawl:*` / PostgreSQL frontier state
3. Gracefully degrades with a warning if Redis is unreachable (never raises)

**What gets cleared**

| Store | What | Keys |
|-------|------|------|
| Redis | Exact-match answers | `rag:cache:exact:*` |
| Redis | Semantic-similarity answers | `rag:cache:semantic:*` |

**What is preserved**
- Qdrant vector store (all indexed chunks + content hashes)
- BM25 tokenizer cache
- Redis `crawl:*` and PostgreSQL frontier state

**When to use**: After a provider/model change, an answer-quality fix, or when stale cached answers must not be served. Next `dec ask` / UI query re-generates fresh answers.

**Gotchas**
- Running API or Streamlit processes keep their own in-memory L1 copy of recently cached answers; restart those services for a fully cold cache.
- The in-memory caches also expire on their own TTL, so clearing Redis is safe and idempotent.

---

### `dec clear-cache`

Clears one or more cache stores. With no `--type` flag (or `--all`), every cache store is cleared. The RAG query cache (`rag:cache:*`), embedding cache (`embed:cache:*`), and crawl cache (`crawl:*` + `ingest:enrichment_failed:*`) live in Redis; the BM25 cache is a persisted tokenizer under `.bm25_cache/` on disk. Qdrant and PostgreSQL are never touched.

```
usage: dec clear-cache [-h] [--query] [--embedding] [--crawl] [--bm25] [--all]
```

**Examples**

```bash
dec clear-cache              # clear every cache store
dec clear-cache --all        # same as above
dec clear-cache --query      # RAG query cache only
dec clear-cache --embedding  # embedding cache only
dec clear-cache --crawl      # crawl cache only
dec clear-cache --bm25       # persisted BM25 tokenizers only
```

**What gets cleared**

| Store | What | Keys/Files |
|-------|------|-----------|
| Redis | RAG query cache | `rag:cache:*` (exact + semantic tiers) |
| Redis | Embedding cache | `embed:cache:*` |
| Redis | Crawl cache | `crawl:*` (URL registry + HTTP headers) |
| Redis | Enrichment failures | `ingest:enrichment_failed:*` |
| Disk | BM25 tokenizers | `.bm25_cache/*.json` |

**What is preserved**
- Qdrant vector store (all indexed chunks + content hashes)
- PostgreSQL frontier tables

**Gotchas**
- Running API or Streamlit processes keep in-memory L1 copies of the query/embedding caches; restart those services for a fully cold cache.
- The crawl cache can be re-disabled and re-enabled via `CRAWL_CACHE_ENABLED`; clearing it does not affect Qdrant dedup.
- See the [Cache toggles](#cache-toggles) settings section for per-type enable/disable.

---

### `dec spark-config-check`

Validates the pinned Spark source configuration (`data_engineering_copilot/config/spark_sources.json`) without network access. Does not download or mutate anything.

```
usage: dec spark-config-check [-h]
```

**Example**

```bash
dec spark-config-check
```

**Exit codes**: `0` valid, `1` invalid configuration.

---

### `dec spark-manifest`

Downloads/materializes the pinned Spark source (from the configured Git commit tarball) and writes a file manifest. Does not index.

```
usage: dec spark-manifest [-h] [--output OUTPUT]
```

**Options**
- `--output`: Manifest output path (default `data/spark_corpus/<generation>/manifest.json`).

**Example**

```bash
dec spark-manifest
```

**Exit codes**: `0` success, `2` invalid config, `5` operational failure (download/extract).

---

### `dec spark-render`

Builds the pinned rendered Spark documentation (Jekyll guides + PySpark Sphinx API docs) and writes `rendered_manifest.json` into `data/spark_corpus/<gen>/`. Requires the pinned native source to be materialized first and the `dec_pydocs_venv` Sphinx toolchain for the PySpark API build.

```
usage: dec spark-render [-h] [--generation GENERATION]
```

**Options**
- `--generation`: Generation identifier (default derived from `spark-<ref>-<commit8>-<embedder-sha12>`).

**Example**

```bash
dec spark-render --generation spark-4.0.0-fa33ea00-abc123
```

**Exit codes**: `0` success, `5` operational failure (missing materialized source, build command exits non-zero, expected output root missing, or timeout).

---

### `dec spark-build`

Builds a Spark generation collection in Qdrant (dense + sparse vectors) without activating it. Fits BM25 on the complete corpus before writing sparse vectors.

When a rendered manifest exists at `data/spark_corpus/<gen>/rendered_manifest.json`, the build runs the hybrid merge (rendered pages preferred over native counterparts, native-only records retained) and writes the exact losslessly-split segment list to `data/spark_corpus/<gen>/chunks.jsonl` before embedding. The build also writes per-file `coverage.json` (status, representation, source path, canonical URL, chunk count) and a `build_report.json` (commit, manifest hashes, chunk/point counts, BM25 size, embedding dimension, validation result).

```
usage: dec spark-build [-h] [--generation GENERATION]
```

**Options**
- `--generation`: Generation identifier (default derived from `spark-<ref>-<commit8>-<embedder-sha12>`).

**Example**

```bash
dec spark-build --generation spark-4.0.0-fa33ea00-abc123
```

**Exit codes**: `0` success, `2` invalid config, `5` operational failure.

---

### `dec spark-validate`

Validates a built generation collection without mutating it. Runs the strict artifact checks (`coverage.json` zero-chunk files, missing rendered outputs, manifest-path and chunk-ID uniqueness, per-chunk generation/commit metadata, Qdrant point count vs `chunks.jsonl`, per-segment token/char budgets ≤ 3800/6000, contiguous segment indices, consistent `segment_total`, lossless reconstruction vs parent hash) plus the store-level checks (dense/sparse vector configuration, BM25 readiness, doc_type metadata presence, payload text == persisted/embedded text). Writes a validation report that `dec spark-activate` requires.

```
usage: dec spark-validate [-h] --generation GENERATION
```

**Options**
- `--generation`: Generation identifier (required).

**Example**

```bash
dec spark-validate --generation spark-4.0.0-fa33ea00-abc123
```

**Exit codes**: `0` success, `2` invalid config/args, `3` validation failure, `5` operational failure.

---

### `dec spark-activate`

Atomically repoints the logical Qdrant alias `data_engineering_docs` to a validated generation collection. Requires a passing validation report from `dec spark-validate`, and interactive confirmation (or `FORCE=1`).

```
usage: dec spark-activate [-h] --generation GENERATION
```

**Options**
- `--generation`: Generation identifier (required).

**Behavior**
- Refuses to activate if the generation has no passing validation report.
- Prompts `[y/N]` on a TTY; non-interactive execution requires `FORCE=1`.
- Atomically switches the alias via Qdrant `ChangeAliases` (delete + create in one request).
- Records active state to `.index_state/active.json` and appends to `.index_state/history.jsonl`.

**Exit codes**: `0` success/aborted, `3` missing/failed validation, `5` operational failure.

---

### `dec spark-rollback`

Points the logical alias back to the previously recorded generation in `.index_state/history.jsonl`. Requires the requested generation to be the current active generation, and interactive confirmation (or `FORCE=1`).

```
usage: dec spark-rollback [-h] --generation GENERATION
```

**Options**
- `--generation`: Generation identifier (required).

**Exit codes**: `0` success/aborted, `4` not the active generation / no previous generation, `5` operational failure.

---

### `dec gen-config-check`

Validates the combined pinned sources configuration (`config/pinned_sources.json` → `pinned_sources_path`) without network access. Checks that every source is `github` or `url_index`, names/slugs are unique, GitHub sources carry an HTTPS repository + 40-hex commit + non-empty streams, and `url_index` sources carry `index_url`/`url_prefix`/`base_url`/`cache_dir`/`doc_type`.

```
usage: dec gen-config-check [-h]
```

**Exit codes**: `0` valid, `1` invalid.

---

### `dec gen-manifest`

Materializes every pinned source (GitHub tarballs at their pinned commits + `llms.txt` url-index pages) and writes per-source manifests `manifest-<slug>.json` plus a combined `manifest.json` under `pinned_corpus_dir/<generation>/`. The generation defaults to `_default_generation()` (a `pinned-{sha12}` digest over the embedder name and the full pinned config).

```
usage: dec gen-manifest [-h] [--generation GENERATION]
```

**Options**
- `--generation`: Generation identifier (default: derived `pinned-{sha12}`).

**Exit codes**: `0` success, `5` materialization failure.

---

### `dec gen-build`

Builds a combined frozen generation collection (not activated) from all five pinned sources: Spark (full SparkChunker fidelity incl. SQL function registry), Airflow + Delta (HeaderAwareChunker with RST/MDX heading support), Claude Platform + Claude Code (url-index pages). Prepares each source via `GithubSourcePreparer`/`UrlIndexPreparer`, then `PinnedIndexBuilder` combines them: per-source commit validation, global content-hash dedup, batch embedding through the fallback chain, one combined `fit_bm25_corpus`, frozen upsert into `data_engineering_docs__<generation>`, and writes `chunks.jsonl` / `coverage.json` / `build_report.json`.

```
usage: dec gen-build [-h] [--generation GENERATION]
```

**Infrastructure**: Qdrant + embedder (via fallback chain).

**Exit codes**: `0` success, `5` build failure.

---

### `dec gen-validate`

Validates a built combined generation without mutation. Runs the pinned artifact checks (chunk generation stamp, source commit ∈ pinned commits, coverage coverage of every chunk `file_path`, chunk-ID uniqueness, Qdrant point count vs `chunks.jsonl`) plus store-level checks (sparse config, BM25 readiness, doc_type metadata, payload text). Writes the validation report required by `dec gen-activate`.

```
usage: dec gen-validate [-h] --generation GENERATION
```

**Exit codes**: `0` passed, `2` invalid identifier, `3` failed/missing artifacts, `5` operational failure.

---

### `dec gen-activate`

Atomically repoints the logical alias to a validated combined generation. Shares the validation-report gate, alias change, and `.index_state/active.json` write with `dec spark-activate`.

```
usage: dec gen-activate [-h] --generation GENERATION
```

**Exit codes**: `0` success/aborted, `3` missing/failed validation, `5` operational failure.

---

### `dec gen-rollback`

Points the logical alias back to the previously recorded generation.

```
usage: dec gen-rollback [-h] --generation GENERATION
```

**Exit codes**: `0` success/aborted, `4` not active / no previous generation, `5` operational failure.

---

### `dec gen-reset`

Wipes all generation state: drops the active alias, deletes every `data_engineering_docs__*` collection, removes `.index_state/active.json`, `.index_state/history.jsonl`, and `validation-*.json` reports, deletes `.bm25_cache/data_engineering_docs*.json` tokenizers, then runs the full `reset-index` crawl-state purge. Disk source caches (`data/spark_src`, `data/raw_sources`, `data/pinned_src`) are preserved. Requires confirmation or `FORCE=1`.

```
usage: dec gen-reset [-h]
```

**Exit codes**: `0` success/aborted, `5` collection purge failure.

---

### `dec gen-stale`

Reports every generation collection in Qdrant as `active`, `stale` (built but not the active index and still backed by local `chunks.jsonl` artifacts), or `orphan` (no local artifacts).

```
usage: dec gen-stale [-h]
```

**Exit codes**: `0` success, `5` collection list failure.

---

### `dec ui`

Prints the command to launch the Streamlit UI (does not launch it).

```
usage: dec ui [-h]
```

**Example**

```bash
dec ui
# Run: python -m streamlit run data_engineering_copilot/ui/streamlit_app.py
```

**Gotchas**
- Prefer the Makefile wrapper (`make streamlit`), which uses the venv interpreter: `dec_venv/bin/streamlit run data_engineering_copilot/ui/streamlit_app.py`.

---

### `dec profile`

Runs a concurrency/load sweep of the ingestion pipeline through the production API, collecting host resource metrics and producing a report with per-stage tuning recommendations.

```
usage: dec profile [-h] [--sources [SOURCES ...]] [--load-sweep LOAD_SWEEP] [--output-dir OUTPUT_DIR]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--sources [S...]` | list[str] (nargs=*) | all | Documentation sources to profile. |
| `--load-sweep` | str (comma-separated ints) | `10,20,50,100` | `max_pages` values to test sequentially. |
| `--output-dir` | str | `./profiler_reports` | Directory for the generated reports. |

**Examples**

```bash
# Default sweep over all sources
dec profile

# Profile two sources at three load levels
dec profile --sources "Apache Spark Documentation" "Databricks Documentation" --load-sweep 5,10,25

# Write reports somewhere else
dec profile --load-sweep 50,100 --output-dir /tmp/profiler_reports
```

**Behavior**
- For each sweep value, dispatches a Celery ingestion task, samples host CPU/resource metrics, and reads production progress (pages, chunks, errors) from Redis.
- Prints per-sweep summary (`Duration`, `Peak CPU`, `Pages fetched`, `Chunks indexed`, `Status`) and per-stage tuning advice:
  ```
  crawler: 20 → 24 (increase)
  parser: 8 → 8 (keep)
  ```
- Saves `telemetry_report.json` + `telemetry_report.md` into the output dir.
- Prints the overall best configuration (load level + throughput + actions).

**Gotchas**
- Requires the API + Celery worker (`:8000`).
- The `dec profile` wrapper only forwards `--sources`, `--load-sweep`, `--output-dir`. The underlying profiler also accepts `--sample-interval` (default `1.0`) and `--poll-interval` (default `2.0`), which are **not** wired through `dec`. To use them, run the profiler module directly:
  ```bash
  dec_venv/bin/python -m data_engineering_copilot.profiler.cli \
      --sources "Databricks Documentation" \
      --load-sweep 20,40 \
      --sample-interval 0.5 \
      --poll-interval 1.0
  ```

---

### `dec health`

Checks the health of all services and exits accordingly.

```
usage: dec health [-h]
```

**Checks performed**
- **Qdrant** — `GET {qdrant_url}/` must return 200. (Note: `/health` is 404; the root path is the health probe.)
- **Redis** — `PING` must return `PONG`.
- **Embedding provider** — prints the configured provider/model (OpenRouter / NVIDIA / Gemini / else).
- **LLM provider** — prints the configured provider/model; if Ollama, also hits `GET {ollama_base_url}/api/tags` for a liveness check.

**Example**

```bash
dec health
# Qdrant:  ✅ Healthy (200 OK)
# Redis:   ✅ Healthy (PONG)
# ...
# ✅ All services healthy
```

**Exit codes**: `0` all healthy; `1` any service unhealthy/unreachable (output also prints `❌ Some services are unhealthy`).

---

### `dec status`

Shows ingestion and system status across the stack.

```
usage: dec status [-h]
```

**Sections reported**
- **Qdrant collection** — name, status, vector count, segment count. A 404 prints "Collection does not exist (run `dec ingest` to create)".
- **Celery workers** — runs `celery -A data_engineering_copilot.workers.tasks inspect active` (needs the worker reachable) and shows active tasks or `No active tasks`.
- **Crawl frontier (PG)** — if `CRAWL_DB_URL` set: page/edge counts and a breakdown of frontier states (`pending`, `processing`, `done`, `failed`, ...).
- **Redis cache** — connection status and DB0 key count.

**Example**

```bash
dec status
```

**Gotchas**
- The Celery probe shells out to the `celery` binary and can report `❌ Workers not responding` when the broker is up but the worker is busy/down.

---

### `dec evaluate`

Runs RAG evaluation over a golden dataset (default `tests/evaluation/eval_dataset.jsonl`).

```
usage: dec evaluate [-h] [--verbose] [--dataset DATASET] [--source SOURCE]
                    [--experiment-name EXPERIMENT_NAME]
                    [--dataset-name DATASET_NAME] [--spark] [--output-dir OUTPUT_DIR]
```

**Behavior**
- Loads evaluation queries (`question`/`query`) from the dataset file, runs each through the RAG service.
- `--dataset <path>` points at any JSONL dataset. The repo ships per-source datasets: `eval_dataset.jsonl` (Apache Spark, 12 queries), `eval_dataset_airflow.jsonl` (3), `eval_dataset_delta_lake.jsonl` (3). Recall-format sets (used with `--spark`) live alongside: `eval_dataset_spark.jsonl`, `recall_claude.jsonl`, `recall_spark_api.jsonl`, `recall_airflow.jsonl`, `recall_delta.jsonl`, `recall_multi_hop.jsonl`, `recall_oos.jsonl`. (`eval_dataset_databricks.jsonl` was removed — Databricks is not in the active generation.)
- `--source <name>` filters loaded rows by their `source_name` field (e.g. `dec evaluate --source "Apache Airflow Documentation"`). Exit code 1 if no rows match.
- Prints per-query `Answer` snippet, `Confidence`, and retrieved-context count.
- Prints summary: total queries, average confidence, **INSUFFICIENT_CONTEXT rate**, and average answer **correctness** (token-F1 vs `ground_truth`) for rows that have ground truth.
- `--output-dir <dir>` writes `per_question_results.jsonl` (id, question, answer, confidence, correctness, contexts) for drift/bisection.
- **RAGAS metrics** (if the `ragas` package is installed): `context_recall`, `context_precision`, `faithfulness`, `answer_relevancy`, `overall`. Otherwise prints "RAGAS evaluation skipped".
- **Langfuse dataset upload** (if Langfuse is reachable): evaluated rows are uploaded to a dataset named `dec-evaluate-{source}-{timestamp}` (override with `--dataset-name`) with `input.query`, `expected_output.answer`, and metadata (confidence, latency_ms, contexts). Prints the dataset name on success.
- **Drift detection** (if `DRIFT_DETECTION_ENABLED`): records a snapshot into `data/eval_history.jsonl`, compares against the trailing window (`DRIFT_WINDOW_DAYS`, default 7), and prints per-metric deltas, `DRIFT DETECTED`, `No drift detected`, or "First eval recorded".

**`dec evaluate --spark`** runs the Spark retrieval-recall evaluation
(`tests/evaluation/eval_dataset_spark.jsonl`, 51 queries) and additionally
measures retrieval-stage diagnostics:

- `term_recall` — expected terms present in the assembled context.
- `source_recall` — expected URLs present in the final context.
- `candidate_source_recall` — expected URLs present in the **fused candidate pool** (pre-rerank). The gap between candidate and final recall is the rerank/truncation diagnostic.
- `expected_fused_ranks` / `dropped_expected_urls` — per-query, which expected sources were retrieved but dropped from the final context.
- `forbidden_term_hits` — terms that must never surface in the evidence (e.g. Delta/Airflow in a Spark row); any hit fails the eval.
- `out_of_scope` / `insufficient_context` — Delta Lake and Airflow rows are explicitly out of scope and must produce a scope refusal ("cannot answer"); a non-refusal fails the eval.
- `insufficient_context` — answer flagged as missing information.
- Stage latency (`retrieval_ms`, `rerank_ms`, `total_ms`) from provenance stage times.
- `--output-dir <dir>` writes machine-readable JSON: `retrieval_provenance.json`
  (per-query per-variant/fused/final candidate records from the opt-in
  provenance capture in `AsyncRagService.answer(..., provenance=...)`) and
  `retrieval_metrics.json` (aggregates: avg recalls over in-scope rows,
  insufficient-context rate, out-of-scope refusal rate, queries dropping
  expected sources or hitting forbidden terms, median/p95 retrieval latency).
- Spark eval exits `1` when avg term or source recall over **in-scope** rows is
  below `0.9`, when any forbidden term surfaces in evidence, or when any
  out-of-scope row is not refused.

**Experiments** (`--experiment-name`)

`dec evaluate --experiment-name "My Run"` uploads the evaluated rows (as above)
and then runs a Langfuse experiment over that dataset via the v4
`dataset.run_experiment` API: each item's `query` is answered by the production
RAG service, a term-overlap `faithfulness` evaluator scores output vs
`expected_output.answer`, and offline RAGAS metrics are scored onto each item
trace (`ragas_context_recall` etc.). Requires reachable Langfuse; the pipeline
still needs Qdrant + embedder + LLM for the RAG task itself.

`dec evaluate --experiment-name "My Run" --dataset-name "some-existing-dataset"`
runs the experiment **directly** against an existing Langfuse dataset, skipping
the eval loop and upload entirely (exit code 1 if Langfuse is unavailable).

**Example**

```bash
dec evaluate
dec evaluate --source "Apache Spark Documentation"
dec evaluate --dataset tests/evaluation/eval_dataset_airflow.jsonl
dec evaluate --spark --output-dir .rag_eval/baseline-01
dec evaluate --experiment-name "baseline-2026-08"
```

**Exit codes**: `0` on completion; `1` if the evaluation dataset is missing, `--source` matches no rows, or (Spark mode) recall is below threshold.

### `dec eval-coverage`

Validates every in-scope recall eval row against the **active generation's**
indexed corpus (`data/pinned_corpus/<gen>/chunks.jsonl`): each `expected_url`
must resolve to an indexed chunk and each `expected_term` must occur in the
corpus. Out-of-scope rows pass by design (they carry no evidence).

```
usage: dec eval-coverage [--dataset DATASET] [--generation GEN] [--json]
```

- `--dataset <path>`: validate a single recall file (default: all recall-format
  files in `tests/evaluation/`).
- `--generation <gen>`: corpus to validate against (default: active generation).
- `--json`: machine-readable report.
- Exit codes: `0` all pass, `1` any row fails, `2` bad input / no corpus.

Additions to a dataset must pass this gate before merge
(`make eval-coverage`); schema/slug/evidence gates run in CI
(`make test-eval-data`).

### `dec gen-synthetic-eval`

Generates a corpus-grounded **synthetic recall set** from the active
generation and filters every row through the coverage validator.

```
usage: dec gen-synthetic-eval --source NAME [--generation GEN] [--limit N]
                              [--out PATH] [--testset-size N]
```

- Deterministic by default (offline): rows are derived from chunk headings and
  every `expected_term` is verified to appear in the chunk text, so rows pass
  `dec eval-coverage` by construction.
- Output: `tests/evaluation/recall_synthetic_<source>.jsonl` (override with
  `--out`).
- The Ragas path (`TestsetGenerator.generate_with_langchain_docs`) exists in
  `evaluation/synthetic_generator.py` and needs factory-wired LLM + embeddings.

**Gotchas**
- Runs the real RAG pipeline (Qdrant + embedder + LLM) — build the index first with `dec ingest`.
- Use a per-source dataset (or `--source`) that matches what you've actually ingested — evaluating Airflow queries against a Spark-only index yields meaningless ~0 scores for every metric.
- RAGAS metrics route through the repo's **adaptive provider routing** (no fixed local model):
  - **LLM**: no provider is pinned as judge. If `EVALUATION_LLM_PROVIDER` is set it becomes the primary of the purpose-`evaluation` chain; when empty (default) every call routes through `llm_fallback_order` and picks the first provider that is currently available (has an API key, not cooling down, inside its rate window) — local Ollama only as the degraded last resort (skipped after consecutive failures). Each ragas metric/query makes one LLM call per requested generation (`answer_relevancy` requests `n=3`, so it makes 3). High-volume runs (RAGAS makes ~20 LLM calls/query) can exhaust a provider's per-minute rate window; the judge then adaptively shifts to the next available provider, so don't pin a low-limit provider.
  - **Embeddings**: NVIDIA then OpenRouter — both default to `nvidia/nemotron-3-embed-1b` (2048-dim), so a mid-run failover keeps the dimension constant (required for ragas cosine similarity). Providers without API keys are skipped; local Ollama (`nomic-embed-text`) is used only when no external key is configured. These eval embeddings never touch Qdrant and are independent of the production index embedder.
  - This uses the same paid providers as the app; a full-dataset run costs real LLM calls (≈18 calls/query on the 12-query Spark set). The eval harness (mocked embedder, no infra) is available separately via `make test-eval`.

---

### `dec rag-plan`

FLASH-executor driver for the general RAG improvement plan
(`plans/rag_general_improvement_execution_plan.md`). Orchestrates the plan's
phases 0-7 with checkpoint/resume, dry-run, and a standard failure schema.

```
usage: dec rag-plan [-h] [--phase {0,1,2,3,4,5,6,7}] [--dry-run] [--force]
                    [--run-id RUN_ID]
                    [--candidate-generation CANDIDATE_GENERATION] [--json]
```

**Behavior**
- Auto-discovers the active generation (same resolution as the RAG service), runs
  `git status`, `dec status`, `dec spark-config-check`, and
  `dec spark-validate --generation <active>` in Phase 0.
- Each run writes artifacts under `.rag_eval/runs/<run_id>/`: `artifacts/`,
  `logs/`, `trials/`, plus `checkpoint.json`, `result.json`, and on failure
  `failure.json` (schema_version-pinned, machine-readable).
- `--phase <n>` runs a single phase; without it, all remaining phases run.
- Completed phases are checkpointed; re-running resumes from the checkpoint.
  `--force` re-runs already-completed phases.
- `--dry-run` prints the exact commands each phase would run without executing
  anything and without writing a checkpoint.
- `--json` prints the final summary as JSON.
- Phases 3 and 7 need `--candidate-generation`; without `--force` they are
  skipped (no candidate) or blocked (candidate present). Phase 7 activation uses
  `FORCE=1` internally. Phases 4-6 halt with exit code 10 because they require a
  code change (candidate provenance instrumentation) that is not implemented yet.

**Phases**
| Phase | Name | What it runs | Destructive |
|---|---|---|---|
| 0 | preflight | git status, dec status, spark-config-check, spark-validate | No |
| 1 | baseline-eval | `dec evaluate --spark` + `dec evaluate` (general dataset) | No |
| 2 | chunk-audit | `dec spark-manifest` + structural distribution analysis | No |
| 3 | contextual-index | `dec spark-build` + `dec spark-validate` (candidate) | No |
| 4 | multi-query-fusion | halted — code change required | No |
| 5 | rerank-context | halted — code change required | No |
| 6 | tuning | halted — code change required | No |
| 7 | rollout | `dec spark-activate` + post-activation `dec evaluate --spark` | Yes |

**Examples**

```bash
dec rag-plan --dry-run                 # preview all planned commands
dec rag-plan --phase 0                 # run the reproducibility gate
dec rag-plan --phase 1 --run-id baseline-01   # baseline evaluation, resume-safe
dec rag-plan --phase 3 --candidate-generation <gen> --force   # build + validate
dec rag-plan --json                    # full run, machine-readable summary
```

**Exit codes**: `0` success (or remaining phases done/skipped); `2` usage;
`4` gate failure (validation, Qdrant, active generation); `5` command failure;
`10` halted — code change required (phases 4-6); `11` blocked — `--force`
required for a candidate generation.

**Gotchas**
- Runs real commands against Qdrant/Redis and, for phases 1/7, makes real
  evaluation LLM calls. It never calls `dec probe-llm`.
- Uses `dec_venv/bin/dec` (the console script) for subcommands — not
  `python -m dec`, which is not a module.

---

### `dec config`

Validates configuration and prints the effective resolved settings.

```
usage: dec config [-h]
```

**Checks performed**
- **Required settings** — `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `QDRANT_URL`, `REDIS_URL` present.
- **URL reachability** — Qdrant (`GET /`) and Redis (`PING`).
- **Embedding configuration** — provider, model, and resolved dimension (from `embedding_model_dimensions` or the fallback).
- **Per-purpose LLM configuration** — resolved provider/model for Answer, Rewrite, Groundedness, Intent, Enrichment, Evaluation, Code (empty provider → `(global default)`).
- **Collection** — existence check against Qdrant.

**Example**

```bash
dec config
# Required Settings:
#   ✅ LLM_PROVIDER: openrouter
#   ...
# Per-Purpose LLM Configuration:
#   Answer: openrouter/openrouter/free
#   Rewrite: groq/llama-3.1-8b-instant
#   ...
# ✅ Configuration valid
```

**Exit codes**: `0` valid (or valid-with-warnings); `1` invalid (unreachable URLs, missing required settings).

**Gotchas**
- The per-purpose model display shows `(global model)` when the model override is empty — the real resolved model may still come from the provider default. See `.env.example` for the resolution priority.

---

### `dec langfuse-seed-prompts`

Idempotently creates (or creates a new version of) every Langfuse-managed prompt via the public API, labeled `production` by default. Requires a reachable, authenticated Langfuse instance.

```
usage: dec langfuse-seed-prompts [-h] [--label LABEL] [--commit-message COMMIT_MESSAGE]
```

**Seeded prompts** (all `type=text`):
- `rag-answer`
- `query-intent-classify`
- `query-rewrite`
- `query-expand`
- `query-hyde`
- `groundedness-nli`
- `chunk-enrichment-summary`
- `eval-faithfulness`
- `judge-faithfulness`
- `judge-relevance`
- `judge-out-of-scope`
- `rag-json-retry-suffix`

**Example**

```bash
dec langfuse-seed-prompts
# seeded rag-answer (version 1)
# seeded query-intent-classify (version 1)
# ...
```

**Behavior**
- **Idempotent** — re-running creates a new version of each prompt under the same label; the runtime resolves to the newest.
- Prints `seeded <name> (version N)` for each prompt.

**Exit codes**: `0` all prompts seeded; `1` Langfuse is unavailable/disabled or the seed fails.

**Gotchas**
- Runtime code never depends on these being seeded — every prompt has a hardcoded fallback that is byte-identical to the Langfuse template (see `data_engineering_copilot/observability/langfuse_prompts.py`).
- Requires `.env.secrets` Langfuse keys and the Langfuse stack up (`make up`).

---

### `dec langfuse-evaluate`

Runs LLM-as-a-judge evaluation over production `rag-query-pipeline` traces: three judges (`faithfulness`, `relevance`, `out_of_scope`) score each trace and write the scores back onto it via the v4 `run_batched_evaluation` API.

```
usage: dec langfuse-evaluate [-h] [--filter FILTER] [--max-items MAX_ITEMS]
                             [--max-concurrency MAX_CONCURRENCY] [--verbose]
```

- `--filter <json>` trace filter array (default: `[{"type": "string", "column": "name", "operator": "=", "value": "rag-query-pipeline"}]`). Streaming traces (`rag-query-pipeline-stream`) are excluded by default.
- `--max-items N` caps the number of traces judged. **Passing `--max-items` bypasses the sampling gate.**
- `--max-concurrency N` concurrent evaluator runs (default 5).
- `--verbose` prints SDK runner progress.

**Behavior**
- Judges run through the purpose-`evaluation` LLM fallback chain (no pinned provider — local Ollama last). Retrieved context for the faithfulness judge comes from the trace's `retrieval` observation (bounded to 12000 chars).
- **Cost gating**: without `--max-items`, the run is skipped entirely with probability `1 - LANGFUSE_SAMPLE_RATE`. `--max-items` always runs (explicit operator intent).
- Prints `Scores created: N` and per-evaluator success stats.

**Example**

```bash
dec langfuse-evaluate --max-items 10
dec langfuse-evaluate --verbose
```

**Exit codes**: `0` run executed or sampled out; `1` error.

---

### `dec langfuse-seed-score-configs`

Idempotently creates any missing Langfuse score configs (and reconciles drifted categories/types) so scores display with proper types/ranges in the UI. Requires a reachable, authenticated Langfuse instance.

```
usage: dec langfuse-seed-score-configs [-h] [--description-suffix DESCRIPTION_SUFFIX]
```

- `--description-suffix <s>` appends a suffix to created config descriptions.

**Seeded score configs**: `confidence`, `groundedness`, `relevance`, `faithfulness`, `user_feedback`, `completeness` (NUMERIC 0–1); `cache_hit`, `out_of_scope` (BOOLEAN); `intent` (CATEGORICAL: factual, code_example, api_lookup, comparative, debugging, how_to); `ragas_*` (NUMERIC 0–1).

**Behavior**
- Matches existing configs by name and only creates missing ones (idempotent).
- If an existing categorical config's categories drifted from the catalog (e.g. `intent` missing newly added labels), it is updated in place.
- Prints `seeded <name>` for created configs and `already exists: <name>` for the rest.

**Example**

```bash
dec langfuse-seed-score-configs
```

**Exit codes**: `0` complete; `1` Langfuse is unavailable/disabled or the seed fails.

---

### `dec langfuse-metrics`

Queries the Langfuse Metrics API v2 for cost, latency, volume, and score analytics. Requires a reachable, authenticated Langfuse instance.

```
usage: dec langfuse-metrics [-h] [{cost-by-model,daily-volume-latency,score-summary}]
                            [--days DAYS] [--score-name SCORE_NAME] [--json]
```

**Preset queries**
- `cost-by-model` — total cost grouped by model, most expensive first.
- `daily-volume-latency` — daily request count and p95 latency (requires observations with recorded latency in the window).
- `score-summary` — average numeric score + count grouped by score name; `--score-name <name>` restricts to one score.

**Options**
- `--days N` look-back window (default 7).
- `--json` pretty-print the raw rows as JSON instead of a TSV table.

**Examples**

```bash
dec langfuse-metrics cost-by-model --days 7
dec langfuse-metrics score-summary --score-name confidence --days 30
dec langfuse-metrics daily-volume-latency --days 1 --json
```

Running with no preset prints the list of presets and exits 0.

**Exit codes**: `0` success (including empty result); `1` Langfuse unavailable.

---

### `dec langfuse-review-queue`

Lists low-confidence production answers from the OSS-compatible
`low-confidence-review` Langfuse dataset. This replaces Enterprise annotation
queues for this deployment.

```
usage: dec langfuse-review-queue [-h] [--limit LIMIT] [--json]
```

- `--limit N` displays at most `N` items (default 100).
- `--json` prints structured JSON containing the item ID, question, answer,
  source trace ID, status, and creation timestamp.

**Examples**

```bash
dec langfuse-review-queue
```

Open the `source_trace_id` in Langfuse to inspect the full retrieval context,
observations, scores, and evaluator results.

---

### `dec inspect-db`

Scrolls the Qdrant collection and prints a payload/source/chunk-type analysis plus a sample chunk.

```
usage: dec inspect-db [-h]
```

**Behavior**
- **Collection overview** — status, points, indexed vectors, segments, mode (`hybrid`/`dense`), dense vector dimension + distance, whether sparse (BM25) is enabled.
- **Embedding model info** — configured provider/model, expected dimension, and a `✅`/`⚠️` dimension match indicator against the collection.
- **Payload distribution** — counts by `source_name` and `chunk_type`, and the top-10 URLs by chunk count.
- **Sample payload** — fields of the first point (`chunk_id`, `source_name`, `title`, `url`, `chunk_type`, `word_count`, `content_hash`, `section_header`, `heading_path`) and the first 300 chars of `text`.

**Example**

```bash
dec inspect-db
```

**Gotchas**
- Scrolls in batches of 1000 points; large collections take a while.
- A missing collection prints an error and returns (exit 0) — no exception.

---

### `dec cancel`

Cancels a running ingestion task via the API.

```
usage: dec cancel [-h] task_id
```

| Argument | Description |
|---|---|
| `task_id` | The Celery task ID returned by `dec ingest` / `dec monitor`. |

**Example**

```bash
dec cancel 9f8a2e1b-3c4d-4e5f-8a9b-0c1d2e3f4a5b
```

**Behavior** — `POST /api/v1/ingest/<task_id>/cancel`, prints `Task <id> cancelled: <status>`.

**Exit codes**: `0` success; `1` HTTP error from the API or the API unreachable (prints the `docker compose up -d backend-api celery_worker` hint).

---

### `dec ingestion-monitor`

Live ingestion dashboard that auto-refreshes.

```
usage: dec ingestion-monitor [-h] [--api-url API_URL] [--task-id TASK_ID] [--interval INTERVAL]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--api-url` | str | `http://localhost:8000` | API base URL. |
| `--task-id` | str | latest task | Specific task ID to monitor. |
| `--interval` | int | `30` | Refresh interval in seconds. |

**Examples**

```bash
# Monitor the most recent ingestion task
dec ingestion-monitor

# Monitor a specific task, refresh every 5s
dec ingestion-monitor --task-id 9f8a2e1b-3c4d-4e5f-8a9b-0c1d2e3f4a5b --interval 5
```

**Behavior**
- Renders a full-screen dashboard: status icon + task + elapsed time, aggregate metrics (pages, chunks, skipped, errors) with deltas and rates, current crawl URL, per-source breakdown, recent events, and the final error message.
- Uses `GET /api/v1/ingest/latest` when no task id is given, else `GET /api/v1/ingest/status/<task_id>`.
- **Resilient polling**: transient fetch failures (timeouts, network errors, HTTP 5xx) are retried up to 3 times with backoff before the connection is considered lost.
- Exits when the task reaches a terminal state (`COMPLETED`/`FAILED`/`CANCELLED`), printing a final line.

**Exit codes**: `0` on clean completion/Ctrl-C; `1` if no task is found or the connection is lost mid-run.

---

### `dec probe-llm`

Probe every configured LLM/embedding provider with **one real request each** and report the actual result. Use this to find out *which provider works and which doesn't* when the app silently falls back (429, auth errors, timeouts, deprecated models). It does **not** go through the adaptive router's gate, so providers in cooldown are still genuinely tested.

```
usage: dec probe-llm [-h] [--providers [PROVIDERS ...]] [--purpose PURPOSE]
                     [--prompt PROMPT] [--timeout TIMEOUT] [--json]
                     [--verbose] [--no-embeddings]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--providers` | str list | all configured | Probe only these providers, e.g. `--providers openrouter groq`. |
| `--purpose` | str | all | Probe only the chain for one purpose: `answer`, `rewrite`, `groundedness`, `intent`, `enrichment`, `evaluation`, `code`. Implies `--no-embeddings`. |
| `--prompt` | str | `Reply with exactly: pong` | Prompt sent to each provider. |
| `--timeout` | float | `10` | Per-provider request timeout in seconds. |
| `--json` | flag | off | Machine-readable JSON output. |
| `--verbose` | flag | off | Show request headers (auth redacted), response preview, embedding dimension, token usage. |
| `--no-embeddings` | flag | off | Skip the embedding-provider probe. |

**What it probes**
- Exactly **one call per unique provider** — providers are deduplicated, so a provider is never probed once per purpose.
- Each provider is probed with its **default model** (`{provider}_model`, falling back to global `llm_model`) — per-purpose model overrides do not trigger extra calls.
- The global LLM provider plus every per-purpose provider that's set and every provider in `llm_fallback_order` (each deduplicated to one call); the `Roles` column shows all purposes/roles that use that provider.
- The embedding provider (one call, reports returned vs expected dimension).
- `nvidia` is skipped automatically (never used in LLM chains); probe it explicitly with `--providers nvidia`.
- A provider with no API key set shows as `SKIP` / `CONFIG:` — keys are never printed.

**Example output**
```text
Kind       Provider    Model                       Roles                                                Status  HTTP  Latency  Verdict
---------  ----------  --------------------------  ---------------------------------------------------  ------  ----  -------  -------
llm        openrouter  openrouter/free             global, answer, enrichment, fallback                 OK      200   1,670ms  OK
llm        groq        llama-3.1-8b-instant        rewrite, groundedness, intent, evaluation, fallback  OK      200   263ms    OK
llm        gemini      gemini-2.5-flash            code                                                 FAIL    404   527ms    FAIL
llm        cerebras    gpt-oss-120b                fallback                                             OK      200   538ms    OK
llm        ollama      llama3.2:3b                 fallback                                             OK      200   747ms    OK
embedding  nvidia      nvidia/nemotron-3-embed-1b  embedding                                            OK      200   373ms    OK
```

**Examples**
```bash
# Check every configured provider
dec probe-llm

# Only check two providers, machine-readable
dec probe-llm --providers openrouter groq --json

# Diagnose why the code purpose keeps failing
dec probe-llm --purpose code --verbose

# Skip the embedding probe
dec probe-llm --no-embeddings
```

**Behavior & gotchas**
- Uses the same model resolution and request shape as production (via `_build_purpose_llm_client`), but fires **one** raw request per target — no gate, no rate limiter, no failover — so a failing provider fails fast and reports the real error instead of silently falling back.
- Reports the same error categories the adaptive router uses (`rate_limited`, `authentication_error`, `quota_exceeded`, `permanent_error`, …) plus the HTTP status and `Retry-After` on 429.
- Each probe consumes 1 request against the provider's rate limits (OpenRouter 20 RPM / 1000 RPD) — repeated `--providers all` runs can eat into your quota.

**Exit codes**: `0` all probes OK; `1` at least one probe failed; `2` invalid `--purpose` or no targets.

---

### `dec version` (not implemented)

`dec version` is **not** a valid command — argparse rejects it:

```text
dec: error: argument command: invalid choice: 'version'
```

The API exposes the build/version info instead: `GET /api/v1/version` (git SHA + image build time + `deps_fingerprint_ok`). Use `docker inspect` on the running image or `curl http://localhost:8000/api/v1/version` for version details.

---

## Cheat sheet

| Task | Command |
|---|---|
| Build/rebuild the index | `dec ingest` |
| Ingest one source, capped | `dec ingest --source "Apache Spark Documentation" --max-pages 50` |
| Ingest Claude LLM docs (no crawler) | `dec ingest-claude-docs --site all` |
| Ask a question | `dec ask "What is a Spark DataFrame?"` |
| Re-enrich failed summaries | `dec reenrich --source "Apache Spark Documentation"` |
| Retry all failed pages | `dec retry-failed --source "Apache Spark Documentation"` |
| Retry only fetch failures | `dec retry-failed --source "..." --category fetch` |
| Re-process skipped pages | `dec unskip --source "Apache Spark Documentation"` |
| Wipe everything and rebuild | `dec reset-index` |
| Recreate only Qdrant + BM25 | `dec reset-qdrant` |
| Reset crawler DB only (keep Qdrant) | `dec reset-crawler-db` |
| Validate Spark source config | `dec spark-config-check` |
| Materialize Spark source + manifest | `dec spark-manifest` |
| Build a Spark generation | `dec spark-build --generation <gen>` |
| Validate a Spark generation | `dec spark-validate --generation <gen>` |
| Activate a Spark generation | `dec spark-activate --generation <gen>` |
| Roll back a Spark generation | `dec spark-rollback --generation <gen>` |
| Validate pinned sources config | `dec gen-config-check` |
| Materialize all pinned sources + manifest | `dec gen-manifest` |
| Build the combined pinned generation | `dec gen-build` |
| Validate the combined generation | `dec gen-validate --generation <gen>` |
| Activate the combined generation | `dec gen-activate --generation <gen>` |
| Roll back the combined generation | `dec gen-rollback --generation <gen>` |
| Wipe all generation state + re-run reset | `dec gen-reset` |
| List active/stale/orphan generations | `dec gen-stale` |
| Spark retrieval-recall evaluation | `dec evaluate --spark` |
| Preview plan phase commands | `dec rag-plan --dry-run` |
| Run RAG improvement plan phase | `dec rag-plan --phase <0-7>` |
| Profile ingestion | `dec profile --load-sweep 10,50,100` |
| Health check | `dec health` |
| System status | `dec status` |
| Validate config | `dec config` |
| Seed Langfuse-managed prompts | `dec langfuse-seed-prompts` |
| Run LLM-as-a-judge over production traces | `dec langfuse-evaluate --max-items N` |
| Seed Langfuse score configs | `dec langfuse-seed-score-configs` |
| Query Langfuse metrics (cost/latency/scores) | `dec langfuse-metrics cost-by-model --days 7` |
| RAG evaluation | `dec evaluate` |
| Inspect the vector DB | `dec inspect-db` |
| Cancel a task | `dec cancel <task-id>` |
| Live dashboard | `dec ingestion-monitor --task-id <task-id> --interval 5` |
| Check which LLM providers work | `dec probe-llm` |
| Launch Streamlit | `dec ui` (then run the printed command) |

---

## Common workflows

**Full dev loop (API-based ingestion)**
```bash
make dev                                # first time: build image + start stack + pull models
dec ingest --source "Apache Spark Documentation" --max-pages 20
dec ingestion-monitor --interval 5     # watch progress live
dec ask "What is Apache Spark?"
```

**After changing dependencies**
```bash
make install                            # update venv
make rebuild                            # rebuild image so deps_fingerprint matches
```

**Clean rebuild after changing the embedding provider/model**
```bash
# Embedding dimension changes require a fresh collection
dec reset-qdrant
dec ingest
```

**Local-only RAG (no API/Docker app services)**
```bash
# Start infra only: docker compose up -d qdrant redis ollama
dec ask "How do I write a Spark UDF?"
dec evaluate
```

**Iterating on config** — edit `.env`/`.env.secrets`/`.env.local`, then confirm the resolved values with `dec config` before running ingestion.

**Diagnosing LLM provider health** — when `dec ask` feels slow or answers degrade (silent fallback), find out exactly which provider works and why others fail:
```bash
dec probe-llm                       # every LLM + embedding provider, one call each
dec probe-llm --purpose code        # just the code chain (e.g. a failing code_llm_provider)
dec probe-llm --json                # machine-readable, pipe to jq / log it
```
