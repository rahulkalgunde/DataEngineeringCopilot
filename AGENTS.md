# DataEngineeringCopilot — Agent Guide

Python 3.12+. RAG over data-engineering docs: Qdrant + Ollama + FastAPI + Celery + Redis + Streamlit. Tooling: `uv` only, Ruff, Pyright, Pytest.

## ⚠️ ORCHESTRATOR MANDATE — READ BEFORE EVERY ACTION

**You are an orchestrator, not a worker. Violating these = wasted tokens + user frustration.**

1. **COMMANDS >30s** → background: `setsid <cmd> > /tmp/opencode/<name>_<ts>.log 2>&1 & disown`. Poll with `tail -5` / `kill -0 <pid>`. NEVER block foreground.
2. **2+ FILES or 2+ SUBTASKS** → dispatch `task(subagent_type: "general")`. Write brief. Wait for report. NEVER inline.
3. **POLLING** → short `tail`/`ps` commands. NEVER `sleep N && tail` in foreground.
4. **BEFORE ANY ACTION** → load `orchestrator-guard` skill. Answer its 3 questions. If any = YES, dispatch/background instead.

## Verification Loop (two-tier)
**Tier 1 — after every edit (~5–10s), only on touched files:**
1. `dec_venv/bin/python -m ruff check <files> --fix`
2. `dec_venv/bin/python -m ruff format <files>`
3. `dec_venv/bin/python -m pyright <files>`
4. `dec_venv/bin/python -m pytest tests/unit/<specific_test> -v -n 0`

**Tier 2 — milestone only (feature complete, before commit), run once:**
`ruff check data_engineering_copilot/ tests/ --fix` → `ruff format …` → `pyright …` → `pytest tests/unit/ -n 6` (all via `dec_venv/bin/python -m …`).

The full suite between milestones only burns time; Tier 2 exists to catch cross-module surprises (import wiring, shared fixtures). CI (`.github/workflows/test.yml`) is **hermetic only**: lint → unit → eval-data/schema gates. It does NOT run pyright (local-only gate) and does NOT run integration/e2e/smoke/retrieval-gate — anything needing Docker/Ollama/testcontainers stays local (`make test-integration`, `make test-e2e`, `make test-real`, `dec gen-*`, `eval-retrieval-gate`).

## Rules (full text in opencode.json `instructions`)
| Rule | Summary |
|------|---------|
| **R1: Orchestrator** | Dispatch > inline, background > foreground, poll > block |
| **R2: Permission** | Ask before long commands/destructive ops. Never kill without asking |
| **R3: Verification** | Tier 1 per-edit, Tier 2 milestone only. `-n 6`, never `-n auto` |
| **R4: CI Health** | Daily `gh run list`, fix red before other work. `/check-ci` shortcut |
| **R5: Session** | Plan/save/resume via timestamp files. Pre-flight env check. Graphify on exit |
| **R6: Safety** | 3-strike breaker, surgical edits, zero silent failures, turn discipline |
| **R7: Quality** | Contract tests, prove-red-before-green, Tier-2 for eval/CI changes |
| **R8: Coaching** | Enforce velocity on human. Anti-drift. Skill suggestions (1/turn max) |
| **R9: Project** | Gen-build preflight, degradation=hypothesis-zero, one-concern/commit, no stash |
| **R10: Rate limit** | Sleep N+2 on rate-limit errors |

## Tooling & Setup

### Local Development Setup
```bash
make dev    # First-time: build image, start stack, pull Ollama models
make up     # Start with last built image
make down   # Stop everything
make rebuild # Rebuild after pyproject.toml/uv.lock changes
make status  # Container + health status
```

### CLI Commands (`dec_venv/bin/dec`)
Core: `dec ask "query"` · `dec health` · `dec config` · `dec inspect-db` · `dec status`
Ingestion: `dec ingest --max-pages 40` (Celery) · `dec ingest-claude-docs` (in-process)
Eval: `dec eval-fast` · `dec eval-retrieval` · `dec eval-generation` · `dec eval-rerank` · `dec eval-chunking`
Generation: `dec gen-manifest` → `dec gen-build` → `dec gen-validate` → `dec gen-activate`
Reset: `dec reset-index` · `dec clear-cache --query`

### Package Management
- NEVER use `pip` or `python -m venv` — use `uv` exclusively
- Create venv: `uv venv dec_venv` · Install dev: `uv pip install -e ".[dev]"`

### Testing Commands
- `make test-unit` / `make test-unit-serial` — Unit (parallel / `-n 0` for debug)
- `make test-integration` — Integration (needs Qdrant + Ollama)
- `make test-e2e` — End-to-end
- `make test-real` — Hard gate with live infra (`REQUIRE_INFRA=1`)
- `make test-ui` — Playwright browser E2E

## Testing Strategy

### Unit Testing (hermetic)
- Use `make_settings()` factory for hermetic settings
- Tests auto-skip when services are down
- Use `unique_collection_name()` for isolation
- Rate limiter isolation via `_isolate_rate_limiter()` fixture
- `make_settings()` hardcodes provider keys to `""` — tests needing env-file aliasing must construct `AppSettings(_env_file=...)` directly

### Integration Testing
- Requires live services: `make test-integration` needs Qdrant + Ollama; `make test-real` needs full stack
- Use `require_qdrant()`/`require_ollama()` for guards
- Host-side Redis: `redis://:local_secure_password_123@localhost:6379/0`

### Evaluation Gates
- Retrieval: `make eval-retrieval-gate` (R@10 >= baseline - 0.02, floor 0.25)
- Generation: Configurable thresholds in `settings.py`
- Schema: `make test-eval-data` (hermetic)

### Test Structure
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.rag`, `@pytest.mark.ingestion`, `@pytest.mark.e2e`
- xdist: `-n 6` (never `-n auto`). `-n 0` for xdist debugging.
- API contracts: `tests/unit/test_api_contracts.py` — pin down interfaces before writing tests
- Test doubles: input-faithful; fidelity contracts in `tests/unit/test_doubles_fidelity.py`

## RAG System Operations

### Query Path
1. Two-tier cache (exact + semantic) → 2. Query rewriting (intent, decomposition, HyDE) → 3. Multi-query hybrid retrieval (dense + BM25 via Qdrant RRF) → 4. Reranking (cross-encoder / LLM) → 5. CRAG relevance gate → 6. Context assembly (dedup, sibling merge, MMR) → 7. Guardrails (groundedness, scope, PII redaction)

### Index Generation
`dec gen-manifest` → `dec gen-build` → `dec gen-validate` → `dec gen-activate`

### Retrieval Benchmarking
`make eval-retrieval-gate` — threshold: R@10 >= baseline - 0.02; per-intent: R@10 >= max(0, baseline_intent - 0.05)

## Cache Discipline
- Two-tier query cache (exact + semantic) + embedding cache + crawl cache
- Clear: `dec clear-cache --query --embedding --crawl --bm25 --all`
- `QueryCache.is_cacheable` requires non-empty `sources` + minimum confidence

## Retrieval Flags (Dark until Gated)
| Flag | Default | Gate |
|------|---------|------|
| `identifier_sparse_rrf_enabled` | False | identifier recall >= +0.05 |
| `namespace_bm25_enabled` | False | identifier recall >= +0.05, generic <= -0.01 |
| `late_chunking_enabled` | False | Recall@10 within -0.01 baseline + p95 latency >= 20% |

## Configuration Management
- Settings load order: `.env` → `.env.secrets` → `.env.local`
- `.env` overrides class defaults in pydantic-settings — verify actual runtime fallback order when adding providers
- Provider fallback chains: LLM: `groq → cerebras → nvidia → cloudflare → openrouter → gemini → agnes → ollama_cloud → ollama`; Embedding: `nvidia → openrouter → huggingface → local-hf`

## Architecture & Design Patterns
- DI via `factory.py`: `build_rag_service()`, `build_llm_fallback_chain()`, etc. Never hand-instantiate.
- `ProviderFallbackChain`: all LLM/embedding/rerank calls route through it. Per-purpose LLM chains (answer, rewrite, groundedness, intent, enrichment, evaluation, code). Ollama is always `degraded_fallback`.
- Three-valued returns: `None` (unsupported) vs `[]` (empty) vs list — check `is None` explicitly.
- No LangChain/LlamaIndex (except `langchain-text-splitters`).
- `settings.validate_all()` after constructing `AppSettings` (non-test code only).
- Redis: always `get_shared_redis_client()`.
- Generation layer: per-purpose tuning via `generation_temperature`/`code_generation_temperature`; `provider_capabilities.py` gates which params are emitted per provider.
- Fail-open vs fail-closed is contractual: auxiliary verifiers fail open; only evidence-based refusals are hard.
- Package layout: `README.md`. RAG techniques: `docs/RAG_SYSTEM_LEARNER_GUIDE.md`.

## Common Pitfalls
- **Frozen Pydantic models**: `AppSettings` cannot be patched → use `make_settings()`
- **MagicMock without spec**: `hasattr` always returns True → always pass `spec=[...]`
- **Ambient env vars**: raise `RuntimeError` instead of silently overriding
- **Rate limiter**: module-global in-memory store shared across tests → use `_isolate_rate_limiter()`
- **`.env` overrides**: `.env` beats `.env.local` beats class defaults
- **xdist**: never `-n auto`, use `-n 6`
- **Docker image staleness**: after `pyproject.toml`/`uv.lock` changes, `make rebuild` — bind mount alone is not enough

## No-Leak Protocol (defect-class → gate)
Every recurring defect class gets an executable gate:
| Class | Gate |
|---|---|
| Lying test doubles | fidelity registry `tests/unit/test_doubles_fidelity.py` |
| Config mutation | `scripts/lint_env.py` in `make lint` + `tests/unit/test_env_lint.py` |
| Stale derived goldens | provenance sidecars + `make eval-data-stale` |
| Multi-path pin divergence | `test_purpose_pin_precedence.py` parametrized over ALL purposes |
| Container/env drift | `make env-verify` after ANY .env edit or container recreate |
| Structural fracture | `make eval-chunking-corpus` ≤ 0.02 / ≤ 0.01 / ≤ 0.001 |

Ratchet: a defect class recurring twice MUST get a gate in the fixing commit.

## Session Conventions
- Plans → `plans/YYYY-MM-DD_HH-MM_plan.md`; Session context → `sessions/YYYY-MM-DD_HH-MM_session.md`
- Resume: load latest of both, continue first incomplete task
- CI health at session start: `gh run list` or `/check-ci`
- `.clinerules/` targets low-power executor models — apply only when driving such a model
- `graphify update .` before ending session (refreshes graph.json)

## Planner-Worker Workflow (SDD)

**When to use:** Any task with 2+ independent subtasks. Single-file fixes go inline.

**Decomposition tree:** by file → by layer → by concern → too coupled → sequential in planner. Cross-cutting concerns (factory.py, settings.py, cli.py) = planner-only.

**Task brief:** Objective, Files, Interface Contract, Pattern, Verification, Constraints — one file per task in `.superpowers/sdd/<plan>/`.

**Dispatch:** Record BASE commit → write briefs → dispatch parallel (same response) or sequential (after dep) → workers implement+verify+report → planner reviews → fix loop (max 5 rounds, circuit breaker at 5).

**Shared state:** disk artifacts in `.superpowers/sdd/`. Never paste plans into dispatches.

## Environment
- Always `dec_venv/bin/python` / `dec_venv/bin/dec` — never bare `python`. Install: `uv pip install -e ".[dev]"`.
- `local-hf` = in-process HF sentence-transformers (`nvidia/Nemotron-3-Embed-1B-BF16`, 2048-dim). Ollama is NOT an embedding provider.
- `dec_pydocs_venv/` = second venv for `dec spark-render` only. Ignore otherwise.
- Serialize heavy CPU: let `make rebuild`/`make dev` finish before xdist suite.
- Shared box: `make runcheck` + register every heavy job in `/tmp/opencode/ACTIVE_RUNS.md`.

## Docker
- `backend-api` and `celery_worker` are behind `--profile app`. Use `make dev`/`make up`.
- Destructive targets prompt for confirmation; non-interactive needs `FORCE=1`.

## References
- CLI: `docs/cli_guide.md` · Makefile: `docs/makefile_guide.md` · Design decisions: `docs/adr/`
- Binding rules: `opencode.json` `instructions` · Low-power rules: `.clinerules/`
- NVIDIA models: https://build.nvidia.com/models?filters=nimType%3Anim_type_preview&q=agentic (Free Endpoint + agentic only)

## graphify
If `graphify-out/graph.json` exists, use `graphify query "<q>"` before raw grep, run `graphify update .` after code changes.
