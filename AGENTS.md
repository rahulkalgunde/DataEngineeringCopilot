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

## Environment
- Always `dec_venv/bin/python` / `dec_venv/bin/dec` — never bare `python`. Install: `uv pip install -e ".[dev]"` (`make install`).
- `dec_pydocs_venv/` is a second venv used only by `dec spark-render` (Sphinx toolchain for PySpark API docs). Ignore it otherwise.
- Settings load `.env` → `.env.secrets` → `.env.local`. `_env_file=None` does **not** reliably isolate (third-party `load_dotenv()` re-injects `.env` into `os.environ`, which beats env files) — pass explicit kwargs to override.
- Commands expected to take >90–120s: run in background with output to a log file and poll — never block the foreground (opencode.json RULE 1).

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
