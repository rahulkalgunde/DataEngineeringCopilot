---
name: testing
description: Use for ANY task involving DataEngineeringCopilot tests — writing unit/integration/e2e/evaluation tests, hermetic settings via make_settings, pytest markers and xdist, mocking with respx/aresponses, testcontainers, the verification loop, or debug test failures. Triggers: test, pytest, conftest, make_settings, marker, xdist, -n 0, respx, aresponses, testcontainers, REQUIRE_INFRA, verification loop, ruff, pyright.
---

# DataEngineeringCopilot Testing

Run everything with `dec_venv/bin/python -m pytest` / `-m ruff` / `-m pyright`.
Never bare `python`.

## Verification loop (after ANY code change, in order)

```
dec_venv/bin/python -m ruff check data_engineering_copilot/ tests/ --fix
dec_venv/bin/python -m ruff format data_engineering_copilot/ tests/
dec_venv/bin/python -m pyright data_engineering_copilot/ tests/
dec_venv/bin/python -m pytest tests/unit/<specific_test> -v -n 0
```

## Test taxonomy & markers (`pyproject.toml`)

`addopts = "-n 6 --dist worksteal --strict-markers"`, `asyncio_mode="auto"`.

| Marker | Meaning | Infra |
|---|---|---|
| `unit` | fast, isolated, mocked | none |
| `integration` | live services | Qdrant/Ollama/Langfuse/etc. |
| `qdrant`/`ollama`/`langfuse` | requires that service | service |
| `rag` / `ingestion` | full pipeline | Qdrant + Ollama |
| `api` | FastAPI endpoints | Celery/Redis |
| `celery` | Celery workers | Redis broker |
| `e2e` | full Docker stack | everything |
| `evaluation` | RAG quality, mocked embedder | none |
| `xdist_group` | same-worker resource sharing | — |
| `serial` | must not run parallel | — |

`tests/conftest.py` auto-skips infra-marked tests when a service is down;
`REQUIRE_INFRA=1` turns skips into hard failures (`make test-real`).

## Hermetic settings (critical)

`tests/conftest.py` no-ops `dotenv`, patches `AppSettings.__init__` to force
`_env_file=None`, blocks ambient provider env vars and API keys (fails loudly
on violation), and rejects non-Ollama providers unless
`_test_allow_non_ollama=True`.

- **Never** build `AppSettings` directly in tests. Always
  `make_settings(**overrides)` from `tests.conftest`.
- `make_settings()` defaults every provider to Ollama with **empty** keys; add
  placeholder keys only for providers the test actually routes to.
- Never real keys/secrets in tests (pre-commit scans staged files).
- **When adding a new LLM provider**, you MUST also update `tests/conftest.py`:
  add `"{provider}_api_key": ""` to `make_settings()` defaults AND add
  `"{PROVIDER}_API_KEY"` to `_AMBIENT_PROVIDER_VARS` in `pytest_configure()`.
  Without this, ambient API keys silently leak into tests (the test passes
  but exercises the wrong provider). See provider-onboarding skill step 5.
- **`.env` overrides class defaults.** The global `settings` singleton is
  created at import time from `.env`/`.env.secrets`. If `.env` has a hardcoded
  `LLM_FALLBACK_ORDER`, it overrides the class-level default in `settings.py`.
  Tests using `make_settings()` are immune (they bypass env files), but the
  probe CLI and runtime code use the global singleton — always verify the
  actual fallback order at runtime, not just the class default.

## Concurrency / xdist

- Default `-n 6 --dist worksteal`. **Never** use `-n auto`.
- Use `-n 0` to debug shared-resource or xdist-order failures
  (`make test-unit-serial`).
- Shared resources: `xdist_group` or `serial` markers; redis/pg state is shared.

## Mocking patterns

- HTTP: `respx` (httpx mock) for provider clients; `aresponses` for aiohttp.
- Doubles live in `tests/doubles/` (`embedder.py`, `llm.py`, `vector_store.py`,
  `frontier.py`, `redis.py`). **Contracts are pinned** in
  `tests/unit/test_doubles_contract.py` — any double change must keep the
  contract and add a behavioral test with a real object.
- Integration services via `testcontainers[qdrant,redis,ollama]` (ambient
  Docker stack is NOT used unless `REQUIRE_INFRA=1`).

## Writing tests per layer

- **Factory/provider** (`tests/unit/test_provider_factory.py`): use
  `_make_settings()`; assert resolved model/base_url/`_max_tokens_field`;
  missing-key raises via `object.__setattr__(s, "<field>", SecretStr(""))`.
- **LLM client**: `respx.post(".../chat/completions")` mocking; assert payload
  fields (`max_tokens` vs `max_completion_tokens`).
- **RAG service**: fixture `rag_service` with doubled embedder/vector store/LLM;
  assert stage outputs, three-valued returns (`is None` checks).
- **Behavioral/type-only refactors**: must include a test exercising changed
  code with a real object (see `test_doubles_contract.py`).

## Gotchas

- Markers are strict (`--strict-markers`) — registering a typo'd marker fails.
- LLM call sites returning `None` content must be normalized (`.strip()` etc.).
- Never leak ambient provider env into tests; a test that silently passes
  because a key exists in `.env` will fail in CI — always use `make_settings`.
