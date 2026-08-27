# DataEngineeringCopilot — Agent Guide

Python 3.12+. RAG over data-engineering docs: Qdrant + Ollama + FastAPI + Celery + Redis + Streamlit. Tooling: `uv` only, Ruff, Pyright, Pytest.

## Verification Loop (two-tier)
**Tier 1 — after every edit (~5–10s), only on touched files:**
1. `dec_venv/bin/python -m ruff check <files> --fix`
2. `dec_venv/bin/python -m ruff format <files>`
3. `dec_venv/bin/python -m pyright <files>`
4. `dec_venv/bin/python -m pytest tests/unit/<specific_test> -v -n 0`

**Tier 2 — milestone only (feature complete, before commit), run once:**
`ruff check data_engineering_copilot/ tests/ --fix` → `ruff format …` → `pyright …` → `pytest tests/unit/ -n 6` (all via `dec_venv/bin/python -m …`).

The full suite between milestones only burns time; Tier 2 exists to catch cross-module surprises (import wiring, shared fixtures). CI (`.github/workflows/test.yml`) is **hermetic only**: lint → unit → eval-data/schema gates. It does NOT run pyright (local-only gate) and does NOT run integration/e2e/smoke/retrieval-gate — anything needing Docker/Ollama/testcontainers stays local (`make test-integration`, `make test-e2e`, `make test-real`, `dec gen-*`, `eval-retrieval-gate`).

### Key Rules (from opencode.json)
- **RULE 1**: Commands >90–120s MUST run in background with `setsid <cmd> & disown`, output to log file, poll for status — never block foreground
- **RULE 11**: Check CI health (`gh run list`) at first session daily; fix red runs before other work
- **RULE 13**: Two-tier cadence — Tier 1 after every edit, Tier 2 only at milestones
- **RULE 23**: ASK before long commands (>60s or loading local models); NEVER kill without asking
- **RULE 24**: API Contract Testing — write contract test before any class tests (pins constructor, methods, properties, invariants)

## Tooling & Setup

### Local Development Setup
First-time setup (builds image, starts stack, pulls Ollama models):
```bash
make dev
```

Day-to-day operations:
- `make up` — Start everything (uses last built image)
- `make down` — Stop everything
- `make status` — Containers and health status
- `make logs` — Stream logs
- `make rebuild` — Rebuild after dependency changes

### CLI Commands (`dec_venv/bin/dec`)
Core (in-process):
- `dec ask "query"` — RAG query
- `dec health` / `dec config` / `dec inspect-db` / `dec status` — Service health

Ingestion:
- `dec ingest --max-pages 40` — Celery ingestion (needs worker)
- `dec ingest-claude-docs` — In-process ingestion

Eval harnesses (in-process):
- `dec eval-fast` — Zero-LLM retrieval integrity check
- `dec eval-retrieval` — Retrieval benchmark
- `dec eval-generation` — LLM quality tests
- `dec eval-rerank` — Reranker smoke test
- `dec eval-chunking` — Chunk quality tests

Generation lifecycle:
- `dec gen-manifest → gen-build → gen-validate → gen-activate` — Atomic index generation
- `dec reset-index` — Clear all indexes
- `dec clear-cache --query` — Clear specific caches

### Package Management
- NEVER use `pip` or `python -m venv` - use `uv` exclusively
- Create venv: `uv venv dec_venv`
- Install dev: `uv pip install -e ".[dev]"`

### Testing Commands
- `make test-unit` — Unit tests (parallel, xdist)
- `make test-unit-serial` — Unit tests serial (debug xdist)
- `make test-integration` — Integration tests
- `make test-e2e` — End-to-end tests
- `make test-real` — Hard gate with live infra (`REQUIRE_INFRA=1`)

## Testing Strategy

### Unit Testing (hermetic)
- Use `make_settings()` factory for hermetic settings
- Tests are auto-skipped when services are down
- Use `unique_collection_name()` for isolation
- Rate limiter isolation via `_isolate_rate_limiter()` fixture

### Integration Testing
Requires live services:
- `make test-integration` needs Qdrant + Ollama
- `make test-real` needs full Docker stack
- Use `require_qdrant()`/`require_ollama()` for guards

### Evaluation Gates
- **Retrieval regression gate**: `make eval-retrieval-gate` (compares to baseline)
- **Generation fidelity gate**: Configurable thresholds in `settings.py`
- **Dataset schema gate**: `test-eval-data` (hermetic)

### Test Structure
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.rag`, `@pytest.mark.ingestion`, `@pytest.mark.e2e`
- xdist default: `-n 6` (never `-n auto`)
- Cache doubles: `tests/unit/test_doubles_fidelity.py` enforces fidelity contracts
- API contracts: `tests/unit/test_api_contracts.py` (pin down exact interfaces)

## RAG System Operations

### Query Path
1. Two-tier cache (exact + semantic)
2. Query rewriting (intent, decomposition, HyDE)
3. Multi-query hybrid retrieval (dense + BM25 via Qdrant RRF)
4. Reranking (cross-encoder / LLM)
5. CRAG relevance gate
6. Context assembly (dedup, sibling merge, MMR)
7. Guardrails (groundedness, scope, PII redaction)

### Index Generation
1. `dec gen-manifest` — Create generation manifest
2. `dec gen-build` — Build index (population)
3. `dec gen-validate` — Validate against corpus
4. `dec gen-activate` — Atomic alias switch

### Retrieval Benchmarking
```bash
make eval-retrieval-gate  # Compare against baseline_inscope.json
```
- Threshold: R@10 >= baseline - 0.02 (absolute floor 0.25)
- Per-intent gate: R@10 >= max(0, baseline_intent - 0.05)
- Run after every RAG pipeline change

## Cache Discipline

### Cache Strategy
- Two-tier query cache (exact + semantic)
- Embedding cache (`embedding_cache_enabled`)
- Crawl cache (`crawl_cache_enabled`)
- Clear with: `dec clear-cache --query --embedding --crawl --bm25 --all`

### Cacheability Rules
- `QueryCache.is_cacheable` requires non-empty `sources` + minimum confidence
- Empty sources silently prevent caching
- Semantic cache uses `semantic_cache_threshold: 0.95`

## Retrieval Flags (Dark until Gated)

### Identifier-Aware Hybrid Search
- `identifier_sparse_rrf_enabled` (default False) — Technical queries use weighted RRF
- Benchmark gate: identifier recall >= +0.05

### Namespace-Aware BM25
- `namespace_bm25_enabled` (default False) — Namespace-aware tokenization
- Benchmark gate: identifier recall >= +0.05, generic recall <= -0.01

### Late Chunking (MRL)
- `late_chunking_enabled` (default False) — Matryoshka retrieval
- Benchmark gate: Recall@10 within -0.01 baseline + p95 latency improvement >= 20%

## Configuration Management

### Settings Loading Order
1. `.env` — defaults (committed)
2. `.env.secrets` — sensitive keys (gitignored)
3. `.env.local` — personal overrides (gitignored)

### Provider API Keys
- API-key-gated providers validated in `settings.py validate_all()`
- Only `skip_provider_check=True` for .env imports (tests use `make_settings()`)
- Free-tier budget monitoring via rate limiters

### Provider Fallback Chains
- LLM: `groq → cerebras → nvidia → cloudflare → openrouter → gemini → agnes → ollama_cloud → ollama`
- Embedding: `nvidia → openrouter → huggingface → local-hf`

## Session Management

### Plan & Context Files
- Implementation plans: `plans/YYYY-MM-DD_HH-MM_plan.md`
- Session context: `sessions/YYYY-MM-DD_HH-MM_session.md`
- Resume: load latest of both files

### Pre-Flight Checklist
1. Check CI health (`/check-ci` or `gh run list`)
2. Verify environment: `dec_venv/bin/python -c "import data_engineering_copilot"`
3. Run Tier-1 gate after every edit (~5–10s)
4. Run Tier-2 gate at milestone completion (~1–2 min)

### Session Cleanup
- Register heavy jobs: `make runcheck` (writes to `/tmp/opencode/ACTIVE_RUNS.md`)
- Background jobs: `setsid <cmd> & disown` (RULE 1)
- Heavy CPU jobs: serialize via `make rebuild` before xdist suite

### Skill Usage (Required)
**ALWAYS load relevant skills BEFORE responding or taking action** — including clarifying questions. Use the `skill` tool to load:
- `brainstorming` — before any creative work / new features
- `systematic-debugging` / `investigate-first` — before fixing bugs
- `testing` — before writing tests
- `codebase-design` — when designing module interfaces
- `safe-refactor` / `surgical-patch` — when refactoring
- `verification-before-completion` — before claiming work is done

If any skill might apply (1% chance), you MUST invoke it. Check available skills with the skill tool.

## Architecture & Design Patterns

### Dependency Injection
- DI via `factory.py`: `build_rag_service()`, `build_llm_fallback_chain()`, etc.
- Never hand-instantiate services directly

### Three-Valued Returns
- `extract_sentences` returns `None` (unsupported) vs `[]` (empty) vs list
- Check with `is None` explicitly

### ProviderFallbackChain
- All LLM/embedding/rerank calls route through `ProviderFallbackChain`
- Per-purpose LLM chains (answer, rewrite, groundedness, intent, enrichment, evaluation, code)
- Provider selection: health-scored, Redis-backed, cached 15s

### Error Categorization
- `_default_categorizer` inspects `LLMClientError.response_body` for model-not-supported patterns
- 401 → `INVALID_REQUEST` not `AUTH_ERROR`
- Ollama is always `degraded_fallback` (last resort, max 2 consecutive failures)

## Common Pitfalls

### Test-Related
- **Frozen Pydantic models**: `AppSettings` cannot be patched → use `make_settings()`
- **MagicMock without spec**: makes `hasattr` always return True → always pass `spec=[...]`
- **Ambient env vars**: raise `RuntimeError` instead of silently overriding → never export provider keys
- **Rate limiter**: module-global in-memory store shared across tests → use `_isolate_rate_limiter()`

### Configuration-Related
- **`.env` overrides**: `.env` beats `.env.local` beats class defaults in pydantic-settings
- **Embedding dimensions**: unknown models fail toward `default_embedding_dimension: 2048`
- **Retrieval flags**: flip only after benchmark gate passes

### Performance-Related
- **Ollama local**: CPU-bound, use `processing_concurrency: 4` (ROLLBACK to 3 if overloaded)
- **xdist**: never use `-n auto`, use `-n 6` (or `-n 0` for debugging)
- **Rate limiting**: shared `SlidingWindowRateLimiter` coordinates RPM/RPD

## Debugging Tools

### Service Health Checks
```bash
# Check service availability
make health  # CLI health check
make status  # Container status + health
```

### Log Locations
- `logs/app.log` — CLI, Streamlit, ingestion, retrieval, vector store
- `logs/ingestion_refresh.log` — UI refresh events

### CI Health
At first session of day: check `gh run list`, investigate failures before other work.

## Best Practices

### Code Quality
- Surgical edits for >100 line files
- Tier-1 gate after every edit (ruff, format, pyright, one targeted test)
- Tier-2 gate before commit (full suite)
- Contract tests before writing tests against any class

### Retrieval Pipeline Changes
- Run `make eval-fast` after every RAG pipeline change
- Run `make eval-retrieval-gate` before any retrieval flag flip
- Compare against baseline: `tests/evaluation/benchmarks/baseline_inscope.json`

### Long-Running Tasks
- Background with output to log file and poll
- Register in `/tmp/opencode/ACTIVE_RUNS.md`
- Use `setsid <cmd> & disown` not plain `nohup ... &`

### Provider Onboarding
- Update `tests/conftest.py`: add provider API key to `make_settings()` defaults AND to `_AMBIENT_PROVIDER_VARS`
- Verify runtime fallback order when adding providers
- Never add paid/anthropic models to `llm_fallback_order`

## Environment
- Always `dec_venv/bin/python` / `dec_venv/bin/dec` — never bare `python`. Install: `uv pip install -e ".[dev]"` (`make install`).
- **Embeddings**: `local-hf` = in-process HF sentence-transformers (`nvidia/Nemotron-3-Embed-1B-BF16`, 2048-dim) — Ollama is NOT an embedding provider; it serves LLMs only. `eval-fast` hardwires local-hf.
- `dec_pydocs_venv/` is a second venv used only by `dec spark-render` (Sphinx toolchain for PySpark API docs). Ignore it otherwise.
- Settings load `.env` → `.env.secrets` → `.env.local`. `_env_file=None` does **not** reliably isolate (third-party `load_dotenv()` re-injects `.env` into `os.environ`, which beats env files) — pass explicit kwargs to override.
- Commands expected to take >90–120s: run in background with output to a log file and poll — never block the foreground (opencode.json RULE 1). Detach with `setsid <cmd> & disown`; a plain `nohup … &` child can be killed when the tool call times out.
- Serialize heavy CPU jobs: let `make rebuild`/`make dev` finish before running the xdist suite — concurrent runs starve pytest workers (`node down: Not properly terminated`).
- Shared box, possibly parallel sessions: run `make runcheck` and register every heavy job (pid/log/eta) in `/tmp/opencode/ACTIVE_RUNS.md` before launching; deregister when done.

## No-Leak Protocol (defect-class → gate)
Every recurring defect class gets an executable gate; gates run in `make lint`/unit suite or as named targets:
| Class | Gate |
|---|---|
| Lying test doubles | fidelity registry `tests/unit/test_doubles_fidelity.py` (+ rule above); doubles travel with consumer changes |
| Config mutation | `scripts/lint_env.py` in `make lint` + `tests/unit/test_env_lint.py`; edit .env only via anchored `^KEY=` edits |
| Stale derived goldens | provenance sidecars + `make eval-data-stale`; generators must write `.provenance.json` |
| Multi-path pin divergence | `test_purpose_pin_precedence.py` parametrized over ALL purposes |
| Container/env drift | `make env-verify` after ANY .env edit or container recreate |
Ratchet: a defect class recurring twice MUST get a gate in the fixing commit.

## Session conventions (from opencode.json `instructions`)
- Implementation plans → `plans/YYYY-MM-DD_HH-MM_plan.md`; on session exit save context → `sessions/YYYY-MM-DD_HH-MM_session.md`; "resume" = load latest of both and continue.
- Check CI health (`gh run list`) at first session of the day and fix red runs before other work; `/check-ci` drives the same loop.
- `.clinerules/` targets low-power executor models (one-edit-per-turn, single-command rules) — apply only when driving such a model.

## Testing
- Tests are hermetic: conftest no-ops `load_dotenv` and **raises** on ambient provider env vars/API keys. Build settings only via `make_settings()` (Ollama-only, no env files); provider-routing tests pass `_test_allow_non_ollama=True` with placeholder keys.
- **`make_settings()` hardcodes provider keys to `""`.** Tests that need env-file aliasing (e.g. `HF_TOKEN` → `huggingface_api_key`) must construct `AppSettings(_env_file=...)` directly — `make_settings` overrides the env file with explicit empty strings.
- **When adding a new LLM provider**, update `tests/conftest.py`: add `"{provider}_api_key": ""` to `make_settings()` defaults AND `"{PROVIDER}_API_KEY"` to `_AMBIENT_PROVIDER_VARS`. Skipping this causes silent env-var leakage into tests.
- **`.env` overrides class defaults.** `pydantic-settings` reads env vars from `os.environ` first; if `.env` has a hardcoded `LLM_FALLBACK_ORDER`, it overrides the class-level default in `settings.py`. Always verify the actual runtime fallback order when adding providers.
- xdist default is `-n 6`; never `-n auto` (destabilizes the machine). Use `-n 0` (`make test-unit-serial`) to debug xdist-order or shared-resource failures.
- Integration/E2E spin up testcontainers (Qdrant/Redis/Ollama). The ambient Docker stack counts only under `REQUIRE_INFRA=1`; `make test-real` is the hard gate that fails when any service is down.
- Host-side Redis probe needs auth: `redis://:local_secure_password_123@localhost:6379/0` (compose Redis runs with `requirepass`).
- Markers are strict (`--strict-markers`); the vocabulary lives in `pyproject.toml`.
- Refactors must ship a behavioral test with a real object; test-double contracts are pinned in `tests/unit/test_doubles_contract.py`.
- **Test doubles are input-faithful**: output derives from the received input (or calls are recorded and asserted). Constant-output doubles only where the real contract is genuinely constant. Fidelity contracts live in `tests/unit/test_doubles_fidelity.py` and travel with consumer changes.

## Docker
- `backend-api` and `celery_worker` are gated behind `--profile app`; bare `docker compose up` starts infra only. Use `make dev` (first time: build + pull Ollama models) / `make up`.
- The image bakes a dependency hash: after `pyproject.toml`/`uv.lock` changes, `make rebuild` — the bind mount is not enough.
- Destructive make targets and destructive `dec` subcommands (gen/spark activate, rollback, reset) prompt for confirmation; non-interactive shells need `FORCE=1`.

## Architecture conventions
- No LangChain/LlamaIndex (except `langchain-text-splitters`).
- DI via `factory.py` (`build_rag_service()`, …) — never hand-instantiate services.
- All LLM/embedding/rerank calls route through `ProviderFallbackChain` (`infrastructure/provider_fallback.py`); obtain via `build_llm_fallback_chain()` / `build_embedding_fallback_chain()`. The factory builds separate per-purpose LLM chains (`answer`, `rewrite`, `groundedness`, `intent`, `enrichment`, `evaluation`, `code`) — don't override globally.
- **Provider selection**: `ProviderFallbackChain` uses `ProviderSelector` (health-scored, Redis-backed, cached best 15s). Error categorizer (`_default_categorizer`) inspects `LLMClientError.response_body` for model-not-supported patterns (401 → `INVALID_REQUEST` not `AUTH_ERROR`). Ollama is always `degraded_fallback` (last resort, max 2 consecutive failures). Single-provider chains still get `ProviderFallbackChain` wrapping for health tracking.
- Redis connections: always `get_shared_redis_client()`.
- Three-valued returns: e.g. `extract_sentences` returns `None` (unsupported) vs `[]` (empty) vs list — check `is None` explicitly.
- Non-test code: call `settings.validate_all()` after constructing `AppSettings`.
- **Generation layer**: per-purpose tuning via `generation_temperature` (0.15) / `code_generation_temperature` (0.20) / `generation_seed` / penalties; `provider_capabilities.py` gates which params are emitted per provider (silently omitted, never errored). Doc-intent answers use schema-enforced structured output (`services/structured_output.py`, strict JSON schema; Ollama gets `format=`, others `response_format=json_schema`).
- **Retrieval flags ship dark until their benchmark gate passes** (`identifier_sparse_rrf_enabled`, `namespace_bm25_enabled` default False with acceptance criteria in `settings.py` comments). Never flip a retrieval flag on without running its eval harness and comparing against baseline.
- **Fail-open vs fail-closed is contractual**: auxiliary verifiers (groundedness, scope, CRAG grader, sibling rejoin, reranker init) fail open with logged warnings; only evidence-based refusals are hard (empty retrieval, low confidence, explicit scope `does_not_cover`). State the posture in module docstrings.
- Package layout and per-module tour: `README.md`. RAG techniques tour: `docs/RAG_SYSTEM_LEARNER_GUIDE.md`.

## API Contract Testing (before writing tests, pin down the API)
**Key principle:** Before writing tests against any class, write a contract test that pins down its constructor signature, method names, property vs method, and behavioral invariants. This is cheap (22 tests, 3s to run) and prevents expensive debugging cycles. See `tests/unit/test_api_contracts.py`.

Findings from prior sessions that cost ~2.5 hours of debugging:
- **Dataclass constructors have specific field names** — `RetrievedChunk` takes `distance` + `confidence`, NOT `score`. `CachedAnswer` takes `sources` (tuple), NOT `citations` (list). Always verify with `inspect.signature` or `dataclasses.fields` before writing tests.
- **Method names must be verified** — `RelevanceGrader.grade_chunks` (not `grades_relevance`), `QueryCache.aget`/`aset_exact` (not `get_or_compute`). Check `hasattr` before using.
- **Property vs method** — `QueryCache.stats` is a property, not a method. Calling `stats()` raises `TypeError`. Use `inspect.getattr_static` to check.
- **Async vs sync** — `QueryCache.aget` is async, `get_exact` is sync. Verify with `inspect.iscoroutinefunction`.
- **MagicMock without `spec=`** makes `hasattr` always return True. Always pass `spec=[...]` when mocking interfaces.
- **Frozen Pydantic models** (`AppSettings`) cannot be patched. Use `make_settings()` with explicit kwargs instead.
- **Cacheability requirements** — `QueryCache.is_cacheable` requires non-empty `sources` AND minimum confidence. Empty sources silently prevent caching.
- **Error constructors** — `ProviderError(category, provider, model)` takes positional args, NOT keyword args. `LLMClientError` uses `ProviderErrorCategory` enum, not raw strings.
- **Numeric contracts** — `ndcg_at_k` with binary relevance returns 1.0 when ALL expected items are present, regardless of position.

## Key make targets
| Target | Description |
|--------|-------------|
| `make test-quick` | Unit tests minus `@slow`, parallel |
| `make test-unit` / `make test-unit-serial` | All unit tests, parallel / `-n 0` for xdist debugging |
| `make test-integration` | Integration in 3 legs: serial, light parallel, Ollama-heavy |
| `make test-real` | Hard gate: `REQUIRE_INFRA=1`, fails if any service down |
| `make test-e2e` | E2E (serial + parallel legs) |
| `make test-eval` / `make test-eval-data` | Eval harness (mocked embedder) / dataset schema gates (both run in CI) |
| `make eval-fast` | Zero-LLM retrieval integrity check (Qdrant + local embedder only) — run after RAG-pipeline changes |
| `make test-chunking` / `test-chunking-serial` | Chunking evaluator suite (gold-span metrics, invariants, snapshots) |
| `make streamlit` | Run the Streamlit UI locally |
| `make dev` / `make up` / `make down` | First-time setup / start / stop stack |
| `make rebuild` | Rebuild image + restart app services (after dependency changes) |
| `make status` / `make logs` / `make logs-worker` | Health / log tailing |
| `make shell svc=redis` | Shell into a service |
| `make FORCE=1 prune` | Remove project containers/images/build cache (volumes kept) |

## CLI entry points
Entry: `main.py:main` → `cli.py`. Full list with per-command infra requirements: `docs/cli_guide.md`. Behavior-changing highlights:

- **`dec probe-llm` makes live paid API calls (one per provider) — get explicit user approval before running.**
- In-process (no Celery): `ask`, `ingest-claude-docs`, `evaluate`, `eval-fast`, `eval-coverage`, `inspect-db`, `health`, `config`. Celery path (needs API + worker + full stack): `ingest`, `profile`.
- Isolated eval harnesses (in-process, frozen inputs): `eval-retrieval` (recall/MRR gate vs baseline), `eval-generation` (faithfulness/relevance/rubric with retrieval frozen), `eval-rerank` (nDCG@K/MRR/P@K on frozen candidate pools), `eval-assembly` (duplicate rate/coverage/compression/needle-loss), `eval-prompt-aug` (template/LLM modes), `eval-chunking` (gold-span chunker quality).
- Generation lifecycle (immutable index gens): `gen-manifest` → `gen-build` → `gen-validate` → `gen-activate` (atomic alias switch); plus `gen-rollback`, `gen-stale`, `gen-reset`. Spark-only mirror: `spark-config-check`, `spark-manifest`, `spark-render` (Sphinx/Jekyll), `spark-build`/`spark-validate`/`spark-activate`/`spark-rollback`.
- Reset granularity: `reset-index` (Qdrant + BM25 + Redis + PG) > `reset-qdrant` (collection + BM25) > `reset-crawler-db` (Redis/PG crawl state, keeps Qdrant); `clear-cache [--query|--embedding|--crawl|--bm25|--all]` for cache stores.
- Recovery: `reenrich` (failed summaries), `retry-failed --category fetch` (failed pages), `unskip`.

## References
- CLI details: `docs/cli_guide.md` · Makefile details: `docs/makefile_guide.md` · Design decisions: `docs/adr/`
- Binding session rules: `opencode.json` `instructions` · Low-power-executor rules: `.clinerules/`

## graphify
If `graphify-out/graph.json` exists, answer codebase questions with `graphify query "<q>"` (or `path`/`explain`) before raw grep, and run `graphify update .` after modifying code (AST-only, no API cost). The directory is commonly dirty from hook updates — that is not a reason to skip it.

- **NVIDIA model discovery**: use https://build.nvidia.com/models?filters=nimType%3Anim_type_preview&q=agentic — consider ONLY models marked Free Endpoint + agentic. `settings.nvidia_model` ids not listed there 404 on the API (calibration 2026-08-24).
