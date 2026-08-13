# DataEngineeringCopilot — Agent Guide

## Tech Stack
Python 3.12+, Pyright, Ruff, Pytest, `uv` only.

## Verification Loop (Run in order after any code change)
1. `dec_venv/bin/python -m ruff check data_engineering_copilot/ tests/ --fix`
2. `dec_venv/bin/python -m ruff format data_engineering_copilot/ tests/`
3. `dec_venv/bin/python -m pyright data_engineering_copilot/ tests/`
4. `dec_venv/bin/python -m pytest tests/unit/<specific_test> -v -n 0` (isolated check)

## Environment & Entry Points
- **VENV**: Always use `dec_venv/bin/python` or `dec_venv/bin/dec`. Never bare `python`.
- **Config**: `.env` → `.env.secrets` → `.env.local`. Passing `_env_file=None` to `Settings` does NOT reliably block env files; explicitly pass all kwargs to override.
- **CLI**: Entry point is `main.py:main` → `cli.py`. See `docs/cli_guide.md` for full command list.
- **Dangerous Command**: `dec probe-llm` makes live paid API calls. **NEVER** run without explicit user approval.

## Testing Gotchas
- **Hermeticity**: `tests/conftest.py` blocks ambient provider env vars and API keys. Never build `AppSettings` directly in tests; use `make_settings()` (Ollama-only defaults).
- **Concurrency**: Default `addopts` uses `-n 6`. **NEVER** use `-n auto` (destabilizes system). Use `-n 0` to debug xdist-order issues or shared-resource tests.
- **Infra**: Integration/E2E tests use `testcontainers` (Qdrant, Redis, Ollama). Ambient Docker stack is NOT used for tests unless `REQUIRE_INFRA=1` is set.
- **Redis Auth**: Use `redis://:local_secure_password_123@localhost:6379/0` for the host-side probe.
- **Markers**: Markers (`@pytest.mark.rag`, etc.) are strict. See `pyproject.toml`.

## Docker & Staleness
- **Staleness**: Image bakes a dependency hash. If `pyproject.toml` or `uv.lock` changes, the bind mount is NOT enough; you **must** run `make rebuild`.
- **Profiles**: `backend-api` and `celery_worker` are gated behind `--profile app`. Bare `docker compose up` only starts infra. Use `make dev`.

## Architecture & Conventions
- **No Heavy Frameworks**: No LangChain or LlamaIndex (except text splitters).
- **Factory DI**: Use `factory.py` (e.g., `build_rag_service()`). Never instantiate services manually.
- **Three-Valued Returns**: Methods like `extract_sentences` return `None` (unsupported), `[]` (empty), or `list` (content). Check `is None` explicitly.
- **Type-Only Commits**: Refactors must include a behavioral test exercising the changed code with a real object (contracts pinned in `tests/unit/test_doubles_contract.py`).
- **Unified Provider Fallback**: All LLM/embedding calls route through `ProviderFallbackChain` in `infrastructure/provider_fallback.py`. Never call a provider directly — use `build_llm_fallback_chain()` / `build_embedding_fallback_chain()` from `factory.py`.
- **Shared Redis Client**: Always use `get_shared_redis_client()` from `factory.py` instead of creating new connections.
- **Per-Purpose LLM Clients**: The factory builds separate LLM chains for each purpose (`answer`, `rewrite`, `groundedness`, `intent`, `enrichment`, `evaluation`, `code`). Do not override globally.
- **Settings Validation**: Call `settings.validate_all()` after constructing `AppSettings` in non-test code.

## Makefile Targets (Key)
| Target | Description |
|--------|-------------|
| `make install` | Install package in editable mode with dev extras |
| `make test-quick` | Fast unit tests (excludes `@slow`), parallel |
| `make test-unit` | All unit tests, parallel |
| `make test-unit-serial` | Unit tests sequential (`-n 0`) — **use for xdist debugging** |
| `make test-integration` | Integration tests in 3 legs (serial, light, Ollama-heavy) |
| `make test-real` | **Hard gate**: `REQUIRE_INFRA=1`, fails if any service down |
| `make test-e2e` | End-to-end tests (serial + parallel legs) |
| `make test-eval` | RAG evaluation harness (mocked embedder, no infra) |
| `make lint` | Ruff check |
| `make format` | Ruff format |
| `make dev` | **First-time setup**: build image + start stack + pull Ollama models |
| `make up` | Start stack (uses last built image) |
| `make rebuild` | Rebuild image + restart app services (after `pyproject.toml`/`uv.lock` changes) |
| `make down` | Stop stack (volumes kept) |
| `make logs` / `make logs-worker` | Follow logs |
| `make status` | Container + health status |
| `make shell svc=redis` | Shell into a service |
| `make FORCE=1 prune` | Remove all project containers/images/build cache (volumes kept) |

## CLI Commands (Key)
| Command | Purpose | Infra Needed |
|---------|---------|--------------|
| `dec ingest --source "Name" --max-pages N` | Crawl + index via Celery | API + worker + Qdrant + Redis + PG + Ollama |
| `dec ingest-claude-docs --site all` | Ingest Claude LLM docs (in-process, no Celery) | Qdrant + embedder |
| `dec ask "question"` | In-process RAG query | Qdrant + Redis + embedder + LLM |
| `dec reenrich --source "Name"` | Re-enrich failed summaries | Qdrant + Redis + PG + Ollama |
| `dec retry-failed --source "Name" --category fetch` | Retry failed pages | Qdrant + Redis + PG + Ollama |
| `dec reset-index` | Full clean rebuild (Qdrant + BM25 + Redis + PG) | Qdrant + Redis + PG |
| `dec reset-qdrant` | Recreate Qdrant collection + BM25 only | Qdrant |
| `dec reset-crawler-db` | Reset Redis/PG crawl state (keep Qdrant) | Redis + PG |
| `dec spark-build --generation <gen>` | Build Spark gen in Qdrant (no activate) | Qdrant + embedder |
| `dec spark-validate --generation <gen>` | Validate built generation | Qdrant |
| `dec spark-activate --generation <gen>` | Atomically switch alias to validated gen | Qdrant |
| `dec gen-manifest` | Materialize all pinned sources + manifest | Qdrant (none) |
| `dec gen-build` | Build combined pinned generation (Spark+Airflow+Delta+Claude) | Qdrant + embedder |
| `dec gen-validate --generation <gen>` | Validate combined generation | Qdrant |
| `dec gen-activate --generation <gen>` | Switch alias to combined gen | Qdrant |
| `dec gen-reset` | Purge alias + gen collections + state + BM25 caches, then reset | Qdrant + Redis + PG |
| `dec gen-stale` | List active/stale/orphan generations | Qdrant |
| `dec evaluate --spark` | Spark retrieval-recall evaluation | Qdrant + embedder + LLM |
| `dec profile --load-sweep 10,50,100` | Ingestion load sweep | API + worker + all infra |
| `dec health` | Component health check | All |
| `dec config` | Validate effective config | Qdrant + Redis |
| `dec inspect-db` | Inspect Qdrant collection | Qdrant |
| `dec probe-llm` | **Live paid API calls** — one probe per provider | All providers |

## References
- CLI details: `docs/cli_guide.md`
- Makefile targets: `docs/makefile_guide.md`
- Rules: `opencode.json`, `.clinerules/`