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

The full suite between milestones only burns time; Tier 2 exists to catch cross-module surprises (import wiring, shared fixtures). CI (`.github/workflows/test.yml`) runs lint → unit/eval → integration+e2e, but **not pyright** — local pyright is the only type gate.

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
- Redis connections: always `get_shared_redis_client()`.
- Three-valued returns: e.g. `extract_sentences` returns `None` (unsupported) vs `[]` (empty) vs list — check `is None` explicitly.
- Non-test code: call `settings.validate_all()` after constructing `AppSettings`.
- Package layout and per-module tour: `README.md`.

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
| `make dev` / `make up` / `make down` | First-time setup / start / stop stack |
| `make rebuild` | Rebuild image + restart app services (after dependency changes) |
| `make status` / `make logs` / `make logs-worker` | Health / log tailing |
| `make shell svc=redis` | Shell into a service |
| `make FORCE=1 prune` | Remove project containers/images/build cache (volumes kept) |

## CLI entry points
Entry: `main.py:main` → `cli.py`. Full list with per-command infra requirements: `docs/cli_guide.md`. Behavior-changing highlights:

- **`dec probe-llm` makes live paid API calls (one per provider) — get explicit user approval before running.**
- In-process (no Celery): `ask`, `ingest-claude-docs`, `evaluate`, `eval-fast`, `eval-coverage`, `inspect-db`, `health`, `config`. Celery path (needs API + worker + full stack): `ingest`, `profile`.
- Generation lifecycle (immutable index gens): `gen-manifest` → `gen-build` → `gen-validate` → `gen-activate` (atomic alias switch); plus `gen-rollback`, `gen-stale`, `gen-reset`. Spark-only mirror: `spark-build`/`spark-validate`/`spark-activate`/`spark-rollback`.
- Reset granularity: `reset-index` (Qdrant + BM25 + Redis + PG) > `reset-qdrant` (collection + BM25) > `reset-crawler-db` (Redis/PG crawl state, keeps Qdrant); `clear-cache [--query|--embedding|--crawl|--bm25|--all]` for cache stores.
- Recovery: `reenrich` (failed summaries), `retry-failed --category fetch` (failed pages), `unskip`.

## References
- CLI details: `docs/cli_guide.md` · Makefile details: `docs/makefile_guide.md` · Design decisions: `docs/adr/`
- Binding session rules: `opencode.json` `instructions` · Low-power-executor rules: `.clinerules/`

## graphify
If `graphify-out/graph.json` exists, answer codebase questions with `graphify query "<q>"` (or `path`/`explain`) before raw grep, and run `graphify update .` after modifying code (AST-only, no API cost). The directory is commonly dirty from hook updates — that is not a reason to skip it.
