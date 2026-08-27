# Provider Catalog & Free-Forever Fallback

Dedicated documentation for the **free_forever provider catalog** — the utility that researches every LLM provider in this project, validates which models are truly `$0-forever`, live-probes them, and builds smart fallback orders the RAG pipeline consumes — **LLM (per-purpose) + embedding (`embedding_fallback_order`) + rerank (`rerank_fallback_order`)**, all `free_forever` and driven by `CATALOG_AUTO_ORDER`.

## Table of Contents
- [Why](#why)
- [Architecture](#architecture)
- [Files & Data Flow](#files--data-flow)
- [Free-Tier Model Inventory (`free_tier_models.json`)](#free-tier-model-inventory-free_tier_modelsjson)
- [Probe Catalog (`provider_catalog.json`)](#probe-catalog-provider_catalogjson)
- [RAG Suitability Gate](#rag-suitability-gate)
- [Ranking: Fastest OK per Provider](#ranking-fastest-ok-per-provider)
- [Auto-Order Wiring (Factory)](#auto-order-wiring-factory)
- [Settings](#settings)
- [CLI: `dec probe-catalog`](#cli-dec-probe-catalog)
- [Live Probe Results (2026-08-23)](#live-probe-results-2026-08-23)
- [Fallback Behaviour & Stale Handling](#fallback-behaviour--stale-handling)
- [Verification](#verification)
- [Research Methodology (Free-Forever)](#research-methodology-free-forever)
- [Adding / Updating a Model or Provider](#adding--updating-a-model-or-provider)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)

---

## Why

The project supports **21 LLM providers** (`data_engineering_copilot/config/settings.py:652` + `factory.py:258` `_build_purpose_llm_client`) behind a unified `ProviderFallbackChain` (`infrastructure/provider_fallback.py:162`). Static `LLM_FALLBACK_ORDER` in `.env` is brittle: free-model names rotate (`openrouter` `:free` suffix), quotas expire (`opencodezen` `FreeUsageLimitError`), and a stale list masks outages until runtime.

The catalog solves three problems:
1. **Discovery** — curated `free_tier_models.json` is the single source of truth for `$0-forever` models (not `$1 credit`).
2. **Validation** — `dec probe-catalog` live-probes each `(provider, model)` via the real `LLMClient` path (`infrastructure/llm_client.py:258`, same as `cli_llm_probe.py:144`), capturing `status/latency/category`.
3. **Smart fallback** — `services/provider_catalog.py:114` `compute_recommended_order()` keeps the **fastest OK model per provider** (dedup) sorted by latency, per purpose, and `factory.py:680` `get_catalog_fallback_order()` feeds it to `build_llm_fallback_chain()` when `CATALOG_AUTO_ORDER=true` (fail-open to `LLM_FALLBACK_ORDER` when missing/stale).

> Scope is `free_forever` only. `$1-credit` providers (`together`, `fireworks`, `mistral` $10/mo) are excluded by construction — the loader enforces `tier == "free_forever"` (`services/provider_catalog.py:45`).

## Architecture

```
config/free_tier_models.json          curated, committed (11 models 2026-08-27)
        │ load_free_tier_models()
        ▼
services/provider_catalog.py          CatalogModel, ProbeEntry, filter/rank, stale check
        │                              is_rag_suitable(), compute_recommended_order(), load_provider_catalog()
        │
cli_catalog.py  ──probe_one()──► LLMClient POST /chat/completions  ──► ProbeEntry{status,latency,category}
        │   (reuses cli_llm_probe._probe_llm_target, provider_capabilities, SlidingWindowRateLimiter, ProviderHealthRegistry)
        ▼
data/provider_catalog.json            gitignored, live output {generated_at, probes[], recommended_fallback_order{global,answer,code,…}}
        │ load_provider_catalog() / get_catalog_fallback_order()
        ▼
factory.py  _build_llm_chain_config()  uses catalog order when CATALOG_AUTO_ORDER=true else LLM_FALLBACK_ORDER
        │                              Ollama stays degraded_fallback
        ▼
ProviderFallbackChain.execute() / generate_stream()  per-purpose chains (answer, rewrite, groundedness, intent, enrichment, evaluation, code)
```

- No SDKs — single `LLMClient` (`infrastructure/llm_client.py`) speaks OpenAI-compatible `/v1/chat/completions`.
- No new deps — uses `httpx`, existing rate limiters, health registry, and `provider_capabilities.py:13`.
- Fail-open: missing/stale/empty catalog never crashes — falls back to `settings.llm_fallback_order`.

## Files & Data Flow

| File | Role | Committed |
|---|---|---|
| `config/free_tier_models.json` | Curated inventory. 14 entries v1.1 (2026-08-23). Each entry is a free_forever model that *should* exist. | Yes |
| `services/provider_catalog.py` | Loader, `is_rag_suitable()`, `compute_recommended_order()`, stale check, serializers. | Yes |
| `cli_catalog.py` | `dec probe-catalog` implementation. | Yes |
| `data/provider_catalog.json` | **Live output** from `dec probe-catalog`. Written atomically `*.tmp→rename`. Contains per-model probe results + `recommended_fallback_order` per purpose. | No (`data/**` in `.gitignore:8`, only `!data/provider_catalog.example.json` whitelisted) |
| `data/provider_catalog.example.json` | Committed example of the live output (real probe 2026-08-23). | Yes |
| `config/settings.py:565` | `free_tier_models_path`, `provider_catalog_path`, `catalog_auto_order`, `catalog_stale_days`. | Yes |
| `factory.py:680` | `get_catalog_fallback_order()` + hook in `_build_llm_chain_config()`. | Yes |
| `cli.py:14` | Wires `probe-catalog` subcommand. | Yes |
| `.env` / `.env.example` | `LLM_FALLBACK_ORDER` (now ranked to match catalog) + `CATALOG_AUTO_ORDER` docs. | Yes |

## Free-Tier Model Inventory (`free_tier_models.json`)

**Schema** (`services/provider_catalog.py:18` `CatalogModel`):

```json
{
  "version": "1.1",
  "updated": "2026-08-23",
  "models": [
    {
      "provider": "groq",
      "model": "openai/gpt-oss-20b",
      "tier": "free_forever",
      "context_window": 131072,
      "max_tokens_field": "max_tokens",
      "supports_structured_output": true,
      "rag_suitable": true,
      "notes": "OK 521ms 2026-08-23"
    }
  ]
}
```

Fields:
- `provider` — lowercased key matching `settings.*_api_key` and `factory._build_purpose_llm_client()` branches (`groq`, `openrouter`, `cerebras`, `gemini`, `zai`, `nvidia`, `anyapi`, `cloudflare`, `opencodezen`, `ollama`, `llm7`, `agnes`, …). Must be `free_forever` — other tiers rejected at load.
- `model` — exact model string sent as `POST {model}`. Must match provider’s hosted name (e.g. `openrouter/nvidia/nemotron-3-nano-30b-a3b:free` needs `:free` suffix, Cloudflare needs `@cf/...`).
- `context_window` — advertised input limit (used for RAG gate `>=8192`).
- `max_tokens_field` — `max_tokens` (NVIDIA/Cloudflare/Ollama/DeepSeek/ZAI/SiliconFlow/…) vs `max_completion_tokens` (Groq/Cerebras/OpenRouter/Gemini/Mistral). Wrong field → HTTP 422.
- `supports_structured_output` — whether provider honors `response_format=json_schema` (`provider_capabilities.py:50` `SUPPORTS_STRUCTURED_OUTPUT`). Gut for `answer`/`code` intents (schema-enforced JSON).
- `rag_suitable` — static flag; combined with gates at rank time.
- `notes` — human note, last probe latency.

**Current inventory (14, v1.1, `free_tier_models.json:6`):** `ollama/phi4-mini:3.8b`, `ollama/qwen2.5-coder:7b`, `openrouter/openrouter/free`, `openrouter/nvidia/nemotron-3-nano-30b-a3b:free`, `zai/glm-4.7-flash`, `groq/openai/gpt-oss-20b`, `cerebras/gpt-oss-120b` (env `CEREBRAS_MODEL`), `gemini/gemma-4-31b-it` (was `gemini-2.5-flash` → 404, fixed), `nvidia/nvidia/nemotron-3-nano-30b-a3b`, `anyapi/nvidia/nemotron-3-nano-30b-a3b:free`, `llm7/default` (`rag_suitable=false` — no structured output), `agnes/agnes-2.5-flash`, `opencodezen/deepseek-v4-flash-free` (kept, currently 400), `cloudflare/@cf/meta/llama-3.1-8b-instruct-fast`.

Removed 2026-08-23 after live 404s: `openrouter/z-ai/glm-4.5-air:free`, `openrouter/google/gemma-3-27b-it:free`, `openrouter/qwen/qwen3-coder:free`, `zai/glm-4.5-flash` (timeout), `siliconflow/Qwen/Qwen3-8B` (402 balance), `huggingface/nvidia/Nemotron-3-Embed-1B-BF16` (not LLM, `helyx/deepseek-chat` missing key).

## Probe Catalog (`provider_catalog.json`)

**Schema** (`services/provider_catalog.py:32` `ProbeEntry` + `ProviderCatalog`):

```json
{
  "generated_at": "2026-08-23T05:38:16.329654+00:00",
  "probes": [
    {
      "provider": "cerebras",
      "model": "gpt-oss-120b",
      "status": "OK",
      "latency_ms": 447.6,
      "http_status": 200,
      "category": null,
      "retry_after": null,
      "message": "",
      "tier": "free_forever",
      "rag_suitable": true,
      "context_window": 8192,
      "supports_structured_output": true
    },
    {
      "provider": "groq",
      "model": "openai/gpt-oss-20b",
      "status": "FAIL",
      "latency_ms": 760.9,
      "http_status": 401,
      "category": "auth_error",
      "message": "HTTP 401: bad key",
      "tier": "free_forever",
      "rag_suitable": true,
      "context_window": 131072,
      "supports_structured_output": true
    },
    {
      "provider": "siliconflow",
      "model": "Qwen/Qwen3-8B",
      "status": "SKIP",
      "latency_ms": null,
      "http_status": null,
      "category": null,
      "message": "CONFIG: SILICONFLOW_API_KEY is required …",
      "tier": "free_forever",
      "rag_suitable": true,
      "context_window": 8192,
      "supports_structured_output": false
    }
  ],
  "recommended_fallback_order": {
    "global": ["cerebras","nvidia","groq","cloudflare","anyapi","openrouter","gemini","zai"],
    "answer": ["cerebras","nvidia","groq","anyapi","openrouter","gemini","zai"],
    "code":   ["cerebras","nvidia","groq","anyapi","openrouter","gemini","zai"]
  }
}
```

- `status`: `OK` (200), `FAIL` (network/HTTP error categorized via `provider_fallback._default_categorizer` + `factory._is_model_not_supported_text` handling 401 `ModelError`), `SKIP` (missing `*_API_KEY` or `offline` or embedding-only `local-hf`).
- `latency_ms`: `time.monotonic()` around the single `POST /chat/completions` with `prompt="Reply with exactly: pong"` (same as `cli_llm_probe.py:144`).
- `recommended_fallback_order`: per-purpose lists produced by `compute_recommended_order()` — see next section. Written atomically to `settings.provider_catalog_path` (default `data/provider_catalog.json`). Committed example at `data/provider_catalog.example.json`.

## RAG Suitability Gate

`services/provider_catalog.py:97` `is_rag_suitable(model, purpose)`:

- Hard: `rag_suitable == true` **and** `context_window >= 8192`. Models with 4k or 8k exactly at boundary pass; below fail.
- Purpose-specific: for `answer` and `code` intents, `supports_structured_output == true` is required — those intents use `services/structured_output.py:1` schema-enforced JSON (`response_format=json_schema` or Ollama `format`). Providers lacking it (SiliconFlow `Qwen/Qwen3-8B`, `llm7/default`, `agnes-2.5-flash`, Cloudflare `@cf/...`) are excluded from `answer`/`code` orders but remain in `global`/`rewrite`/`groundedness`/`intent`/`enrichment`/`evaluation` (which tolerate unstructured).
- Applied twice: statically when curating `free_tier_models.json` (author sets `rag_suitable`), and dynamically when ranking probes (the `ProbeEntry` mirrors the same fields, so a purpose that needs structured output never ranks a probe that lacks it).

Three-valued in spirit (`None` vs `[]` vs `list`) — ranking returns `[]` when no probe passes, which is a valid empty order (fail-open).

## Ranking: Fastest OK per Provider

`services/provider_catalog.py:114` `compute_recommended_order(probes, purpose)`:

1. Filter to `status == OK` **and** `is_rag_suitable` for the purpose (and `context_window >=8192`). Ollama is always excluded (it is the `degraded_fallback` in `factory.py:734`, never in ranked order).
2. **Dedup per provider:** keep only the **fastest** OK model for that provider (min `latency_ms`; `None` treated as `inf`). If a provider has two `:free` variants (e.g. `openrouter/openrouter/free` 2281ms vs `openrouter/nvidia/nemotron-3-nano-30b-a3b:free` 1019ms), only the 1019ms entry survives. This satisfies “fastest OK per provider to avoid duplication”.
3. **Sort** surviving providers by `latency_ms` ascending.

Result: a **provider-level** order (not model-level), stable and short (7–8 entries). Called for each purpose in `_PURPOSES = ["global","answer","rewrite","groundedness","intent","enrichment","evaluation","code"]` (`cli_catalog.py:18`), producing `recommended_fallback_order[purpose]`. `global` is the fallback for unknown purposes.

Non-OK probes (`FAIL`/`SKIP`) never enter ranking; they stay in `probes[]` for diagnostics but not in any order.

## Auto-Order Wiring (Factory)

`config/settings.py:565` adds:

```python
free_tier_models_path: Path = PROJECT_ROOT / "data_engineering_copilot" / "config" / "free_tier_models.json"
provider_catalog_path: Path = PROJECT_ROOT / "data" / "provider_catalog.json"
catalog_auto_order: bool = False
catalog_stale_days: int = 7
```

`factory.py:680` `get_catalog_fallback_order(purpose, app_settings)`:

- Returns `None` when `catalog_auto_order` is false, catalog missing, unreadable, stale (`is_catalog_stale()` > `catalog_stale_days`), or `recommended_fallback_order` empty — logged at `info`/`warning`.
- Else returns `recommended_fallback_order[purpose]` or `recommended_fallback_order["global"]`.

`factory.py:699` `_build_llm_chain_config(purpose, app_settings, ..., purpose_provider, purpose_model)`:

- **Explicit pin wins:** if `purpose_provider` (e.g. `ANSWER_LLM_PROVIDER=openrouter`) is set, it builds `ordered = [pin] + rest_of(LLM_FALLBACK_ORDER)` — catalog is ignored.
- **Else if catalog auto-order and fresh:** `ordered = catalog_order` (already provider-lowercased).
- **Else:** `ordered = settings.llm_fallback_order`.

Then each provider in `ordered` is built via `factory.py:249` `_build_purpose_llm_client()` (resolving model via priority `explicit > {provider}_{purpose}_llm_model > {provider}_model > llm_model`, same as `provider-onboarding` skill), wrapped in `ProviderConfig`, registered in `ProviderHealthRegistry`, split into `main` vs `degraded_fallback` (`ollama`). The chain is a `ProviderFallbackChain` (`provider_fallback.py:162`) with health-scored routing (`provider_selector.py:1`).

**Effect:** when `CATALOG_AUTO_ORDER=true` and a fresh `data/provider_catalog.json` exists, every purpose’s LLM chain automatically prefers the live-fastest free_forever providers in latency order, without touching `LLM_FALLBACK_ORDER`. Flip to `false` or delete the catalog → static `LLM_FALLBACK_ORDER` resumes.

Current live config (`.env:40`): `CATALOG_AUTO_ORDER=true`, `LLM_FALLBACK_ORDER='["cerebras","nvidia","groq","cloudflare","anyapi","openrouter","gemini","zai","sambanova","mistral","llm7","agnes","ollama_cloud","ollama"]'` — kept in sync with the catalog’s global order plus paid fallbacks (`sambanova`/`mistral`) for completeness.

## Settings

| Setting | Path / Field | Default | Env var | Notes |
|---|---|---|---|---|
| Free-tier inventory | `config/settings.py:565` `free_tier_models_path` | `PROJECT_ROOT/config/free_tier_models.json` | `FREE_TIER_MODELS_PATH` | Curated input to `dec probe-catalog`. |
| Probe output | `config/settings.py:565` `provider_catalog_path` | `PROJECT_ROOT/data/provider_catalog.json` | `PROVIDER_CATALOG_PATH` | Live output + factory input. |
| Auto-order toggle | `config/settings.py:565` `catalog_auto_order` | `false` | `CATALOG_AUTO_ORDER` | Set `true` to use catalog order. |
| Stale threshold | `config/settings.py:565` `catalog_stale_days` | `7` | `CATALOG_STALE_DAYS` | Days after `generated_at` before catalog is considered stale. |

All are read via `AppSettings` (`pydantic-settings`), so `.env` → `.env.secrets` → `.env.local` precedence applies (AGENTS.md env layering). Changes to `CATALOG_AUTO_ORDER` require no code change — just set the env and re-probe.

## CLI: `dec probe-catalog`

```bash
dec probe-catalog [-h] [--providers [PROVIDERS ...]] [--purpose PURPOSE]
                  [--prompt PROMPT] [--timeout TIMEOUT] [--json] [--offline] [--output OUTPUT]
```

| Flag | Default | Description |
|---|---|---|
| `--providers [p ...]` | all | Filter to these providers (e.g. `--providers openrouter groq`). |
| `--purpose PURPOSE` | — | Filter `recommended_fallback_order` to one purpose (`global`, `answer`, `code`, `rewrite`, `groundedness`, `intent`, `enrichment`, `evaluation`). Without it, all purposes are computed. |
| `--prompt PROMPT` | `Reply with exactly: pong` | Prompt sent to each `POST /v1/chat/completions`. Keep short to minimize cost. |
| `--timeout TIMEOUT` | `10.0` | Per-provider HTTP timeout (seconds). Fallback chains use `llm_fallback_call_timeout` (30) for non-primary providers, but probe uses this value. |
| `--json` | `false` | Print the catalog JSON to stdout in addition to writing the file. |
| `--offline` | `false` | **No network** — writes `SKIP` skeletons for every model (useful for CI/docs, `make_settings` hermetic tests). |
| `--output OUTPUT` | `settings.provider_catalog_path` | Alternate output path. |

**Behaviour:**
- Loads `free_tier_models.json` via `load_free_tier_models()` (validates `free_forever` only, dedup).
- For each `CatalogModel`, calls `cli_llm_probe._probe_llm_target()` with a real `LLMClient` built via `factory._build_purpose_llm_client(provider, model, timeout)` — same headers, `max_tokens_field`, `Authorization: Bearer …` redacted in logs, same `categorize_provider_error` path. `local-hf` is SKIPs as `embedding-only, not LLM-probed`; `huggingface` is unsupported LLM (SKIP `unsupported provider`). Missing `*_API_KEY` → `SKIP` `CONFIG: … is required` (no HTTP).
- Computes `recommended_fallback_order` per purpose via `compute_recommended_order()`.
- Writes `ProviderCatalog{generated_at, probes, recommended_fallback_order}` atomically to `catalog_path` (`*.tmp→rename`) and optionally to stdout.

**Exit codes:** `0` even if some probes `FAIL` (catalog still valid); `2` on bad config (missing `free_tier_models.json`, unknown `--purpose`).

**Examples:**

```bash
# Full live probe (makes ~14 real API calls — get approval, respects SlidingWindowRateLimiter)
dec_venv/bin/dec probe-catalog --json | tee /tmp/catalog.json

# Only two providers, offline skeleton
dec_venv/bin/dec probe-catalog --providers openrouter groq --offline --json

# Purpose-filtered, custom output
dec_venv/bin/dec probe-catalog --purpose answer --output /tmp/answer_catalog.json --json

# Verify factory sees it
dec_venv/bin/python -c "from data_engineering_copilot.config.settings import settings; from data_engineering_copilot.factory import get_catalog_fallback_order; print(get_catalog_fallback_order('answer', settings))"
```

**Cost & throttling:** each probe is one `POST /chat/completions` per model. Live probes spend free-tier quota (not paid, but rate-limited). `factory._build_provider_rate_limiters()` creates per-provider `SlidingWindowRateLimiter` (`openrouter 18/min 900/day`, `groq 27/min 13000/day`, `cerebras 4/min 2200/day`, `gemini 13/min 450/day`, `zai 60/min 1000/day`, `nvidia 36/min 1000/day`, `anyapi 20/min 500/day`, `llm7 120/min 1000/day`, `agnes 20/min 500/day`, `cloudflare 60/min 1000/day` — `settings.py:704`). Probes are sequential (no concurrency) to avoid burst 429s.

**Related command:** `dec probe-llm` (`cli_llm_probe.py:1`) probes *configured* providers (those referenced by `llm_provider`/`*_llm_provider`/`llm_fallback_order`) — it knows nothing about `free_tier_models.json`. Use `probe-llm` to check wiring, `probe-catalog` to check the *inventory*.

## Live Probe Results (2026-08-23)

Run: `dec_venv/bin/dec probe-catalog --json` (keys from `.env.secrets`, env as in `.env:40`).

| Provider | Model (catalog) | Status | Latency | HTTP | Category / Message | In answer/code order? |
|---|---|---|---|---|---|---|
| `cerebras` | `gpt-oss-120b` | **OK** | 447ms | 200 | — | Yes |
| `nvidia` | `nvidia/nemotron-3-nano-30b-a3b` | **OK** | 513ms | 200 | — | Yes |
| `groq` | `openai/gpt-oss-20b` | **OK** | 521ms | 200 | — | Yes |
| `cloudflare` | `@cf/meta/llama-3.1-8b-instruct-fast` | **OK** | 583ms | 200 | — | No (no structured output) |
| `anyapi` | `nvidia/nemotron-3-nano-30b-a3b:free` | **OK** | 665ms | 200 | — | Yes |
| `openrouter` | `nvidia/nemotron-3-nano-30b-a3b:free` (fastest) + `openrouter/free` 2281ms | **OK** | 1019ms | 200 | — | Yes (keep 1019ms) |
| `llm7` | `default` | **OK** | 1370ms | 200 | — | No (`rag_suitable=false`) |
| `gemini` | `gemma-4-31b-it` | **OK** | 1935ms | 200 | — | Yes |
| `zai` | `glm-4.7-flash` | **OK** → **FAIL** | 3633ms → 654ms | 200 → 429 | OK 2026-08-23 05:38, **429 `code 1305` rate_limited** 05:55 — dropped from order until quota resets | Yes when OK |
| `agnes` | `agnes-2.5-flash` | **FAIL** | — | — | `ReadTimeout` (intermittent; was OK 2111ms 05:38) | No this run |
| `ollama` | `phi4-mini:3.8b` / `qwen2.5-coder:7b` | **FAIL** | — | — | `ReadTimeout` (no local Ollama on this host) | Never (degraded) |
| `opencodezen` | `deepseek-v4-flash-free` | **FAIL** | 811ms | 400 | `invalid_request` `Model is unavailable` | No |
| `openrouter` | `z-ai/glm-4.5-air:free` | **FAIL** | 882ms | 404 | `permanent_error` `unavailable for free` | No |
| `openrouter` | `google/gemma-3-27b-it:free` | **FAIL** | 432ms | 404 | `permanent_error` | No |
| `openrouter` | `qwen/qwen3-coder:free` | **FAIL** | 721ms | 404 | `permanent_error` | No |
| `zai` | `glm-4.5-flash` | **FAIL** | — | — | `ReadTimeout` | No |
| `siliconflow` | `Qwen/Qwen3-8B` | **FAIL** | 1214ms | 402 | `402 balance insufficient` | No |
| `gemini` | `gemini-2.5-flash` | **FAIL** | 281ms | 404 | `no longer available, use gemini-3.6-flash` | No (fixed to `gemma-4-31b-it`) |
| `huggingface` | `nvidia/Nemotron-3-Embed-1B-BF16` | **SKIP** | — | — | `unsupported LLM provider` (embedding) | No |
| `helyx` | `deepseek-chat` | **SKIP** | — | — | `HELYX_API_KEY missing` | No |

**Resulting orders (fastest per provider, `data/provider_catalog.json:42`, live 05:38 with `zai` OK):**
- `global`/`rewrite`/`groundedness`/`intent`/`enrichment`/`evaluation`: `["cerebras","nvidia","groq","cloudflare","anyapi","openrouter","gemini","zai"]` (adds `cloudflare`; `llm7` would be last if `rag_suitable` true, currently excluded from answer/code)
- `answer`/`code`: `["cerebras","nvidia","groq","anyapi","openrouter","gemini","zai"]` (drops `cloudflare`/`llm7`/`agnes` for missing structured output)

**Current live catalog (05:55, `zai` 429 rate_limited → excluded):** `global ["cerebras","groq","anyapi","cloudflare","nvidia","openrouter","gemini"]`, `answer ["cerebras","groq","anyapi","nvidia","openrouter","gemini"]` — `zai` returns when quota resets (re-probe after ~1h).

The probe output at `data/provider_catalog.json` (ignored, 05:55) and the committed example at `data/provider_catalog.example.json` (05:38 snapshot with `zai` OK) reflect live variability.

## Fallback Behaviour & Stale Handling

- **Missing catalog** → `factory.get_catalog_fallback_order()` returns `None` (`services/provider_catalog.py:191` `load_provider_catalog()` `None` → `info catalog_auto_order_no_catalog`), `_build_llm_chain_config()` uses `settings.llm_fallback_order`.
- **Stale catalog** → `is_catalog_stale(generated_at, stale_days)` parses `generated_at` (ISO8601 `+00:00`) vs `now(timezone.utc)`; if `age > stale_days` (`config/settings.py:565` `catalog_stale_days` default 7), returns `None` with `warning catalog_auto_order_stale`, falls back to `llm_fallback_order`. No crash.
- **Corrupt catalog** → `load_provider_catalog()` swallows `OSError`/`JSONDecodeError`, returns `None`.
- **Empty order** → `get_catalog_fallback_order()` returns `None` when `recommended_fallback_order` lacks the purpose and `global`.
- **Ollama degraded** → never in ranked order (`compute_recommended_order` skips `ollama`); `factory.py:734` splits it to `degraded_fallback` (tried only after all external providers fail, with `max_degraded_consecutive_failures` 2).
- **Provider without key** → probe `SKIP` `CONFIG`, not in order, but `factory._build_purpose_llm_client()` would also `raise ValueError` and be skipped at chain build with `warning Skipping provider in LLM fallback chain`.
- **Rate-limit / 429** → probe records `FAIL` `rate_limited` with `retry_after`, `ProviderHealthRegistry` cools down the provider for `retry_after` seconds; next `dec ask` skips it via `_provider_gate()`.

## Verification

**Tier 1 (after every edit, ~5–10s, only touched files):**
```bash
dec_venv/bin/python -m ruff check data_engineering_copilot/services/provider_catalog.py data_engineering_copilot/cli_catalog.py data_engineering_copilot/config/settings.py data_engineering_copilot/factory.py data_engineering_copilot/cli.py --fix
dec_venv/bin/python -m ruff format <files>
dec_venv/bin/python -m pyright <files>
dec_venv/bin/python -m pytest tests/unit/test_provider_catalog.py -v -n 0
```

**Tier 2 (milestone, before commit):**
```bash
dec_venv/bin/python -m ruff check data_engineering_copilot/ tests/ --fix
dec_venv/bin/python -m ruff format data_engineering_copilot/ tests/
dec_venv/bin/python -m pyright data_engineering_copilot/ tests/
dec_venv/bin/python -m pytest tests/unit/ -n 6
```

**Catalog-specific checks:**
```bash
dec_venv/bin/python -m pytest tests/unit/test_provider_catalog.py tests/unit/test_provider_factory.py -n 0
dec_venv/bin/dec probe-catalog --offline --json   # no network, writes SKIP skeleton
dec_venv/bin/dec probe-catalog --providers groq --json  # subset live probe
dec_venv/bin/python -c "from data_engineering_copilot.config.settings import settings; from data_engineering_copilot.factory import get_catalog_fallback_order; print(get_catalog_fallback_order('answer', settings))"
dec_venv/bin/dec probe-llm --json  # probes *configured* chain (not catalog) for wiring check
```

CI (`.github/workflows/test.yml`) is hermetic — it never runs `probe-catalog` live and never needs `data/provider_catalog.json`. `tests/conftest.py:350` `make_settings()` defaults `*_api_key=""` and `catalog_auto_order` is `False` there; the `test_catalog_integration_with_factory` test opts in with `_test_allow_non_ollama=True` and a temp catalog path.

## Research Methodology (Free-Forever)

Free-forever was defined as **$0 prompt + $0 completion, no credit-card, no expiry** — not `$1 starter credit` or `$10 promo`. Sources per provider (2026-08-23):

- **OpenRouter** — `openrouter.ai/api/v1/models` filtered `pricing.prompt==0 && pricing.completion==0` (snapshot 25 models 2026-07-01) + `openrouter.ai/collections/free-models`. Confirmed `openrouter/free` router + `:free` suffix models; `qwen/qwen3-coder:free` etc. have since rotated to paid (404 `unavailable for free` on probe — removed).
- **ZAI (bigmodel.cn)** — `glm-4.7-flash` / `glm-4.5-flash` are `$0 forever` per ZAI docs; `glm-4.7-flash` OK, `glm-4.5-flash` timed out (kept but not ranked this run).
- **NVIDIA NIM** — `nvidia/nemotron-3-nano-30b-a3b` on `integrate.api.nvidia.com/v1`, Developer tier `40 RPM / 1000 RPD` free.
- **Groq** — `openai/gpt-oss-20b`, `27 RPM / 13000 RPD` free.
- **Cerebras** — `gpt-oss-120b` (env `CEREBRAS_MODEL`), `4 RPM / 2200 RPD` free; previous `gemma-4-31b` also OK but env pins `gpt-oss-120b`.
- **Gemini** — `gemma-4-31b-it` via `generativelanguage.googleapis.com/v1beta/openai/`, `13 RPM / 450 RPD` free; `gemini-2.5-flash` is deprecated (404 `use gemini-3.6-flash`).
- **AnyAPI** — `nvidia/nemotron-3-nano-30b-a3b:free` (`:free` suffix required, else 403).
- **Cloudflare Workers AI** — `@cf/meta/llama-3.1-8b-instruct-fast` on `api.cloudflare.com/client/v4/accounts/{id}/ai/v1`, `60 RPM`.
- **SiliconFlow** — `Qwen/Qwen3-8B` (`api.siliconflow.com/v1`, global `.com` not `.cn`) — confirmed permanently free in docs but account balance exhausted in probe (402) → kept but deprioritized.
- **OpenCode Zen** — `deepseek-v4-flash-free` on `/zen/v1` (free on Zen, not on `/zen/go/v1`) — currently 400 `Model is unavailable`.
- **LLM7** — `default` on `api.llm7.io/v1`, `120 RPM` free (aggregator, no structured output).
- **Agnes** — `agnes-2.5-flash` on `apihub.agnes-ai.com/v1`, `20 RPM` free, 512K context (intermittent timeout).
- **Ollama** — `phi4-mini:3.8b` / `qwen2.5-coder:7b` local — degraded fallback, never probed as cloud.

Credit-only providers **excluded** after research: `together` (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, $1), `fireworks` (`llama-v3p3-70b-instruct`, $1), `mistral` (`mistral-small-latest`, $10), `deepseek` (`deepseek-chat`, $0.14/M), `sambanova` (`Meta-Llama-3.3-70B-Instruct`, paid).

Each candidate was checked against the provider’s pricing page and then **live-proved** via `dec probe-catalog` (single `POST /chat/completions` with `pong`).

## Adding / Updating a Model or Provider

1. **Add to `config/free_tier_models.json`** — one object per `(provider, model)` with `tier: "free_forever"`. Copy an existing block, set `context_window`, `max_tokens_field` (`max_completion_tokens` for Groq/Cerebras/OpenRouter/Gemini/Mistral/Opencode*/…; `max_tokens` for NVIDIA/Cloudflare/Ollama/DeepSeek/ZAI/SiliconFlow/Together/Fireworks/LLM7/Agnes/Ollama Cloud), `supports_structured_output` from `provider_capabilities.py:50`.
2. **Set the API key** in `.env.secrets` (`GROQ_API_KEY`, etc.) — without it the probe is `SKIP`.
3. **Probe live:** `dec_venv/bin/dec probe-catalog --providers <new> --json` — must be `OK`.
4. **Verify RAG gate:** if `answer`/`code` need it, ensure `supports_structured_output=true` and `context_window>=8192`.
5. **Re-probe all:** `dec_venv/bin/dec probe-catalog --json` — check `recommended_fallback_order` includes the new provider at the expected latency rank.
6. **Tests:** `dec_venv/bin/python -m pytest tests/unit/test_provider_catalog.py -n 0` — add a case in `test_load_real_free_tier_file` if needed.
7. **Onboard as full provider** (if the provider itself is new, not just a model): follow `.agents/skills/provider-onboarding/SKILL.md` — add settings block in `config/settings.py:704`, `factory._build_provider_rate_limiters()` + `_build_purpose_llm_client()` branch, `provider_api_key_map`, `tests/conftest.py:380` `make_settings` + `tests/conftest.py:283` `_AMBIENT_PROVIDER_VARS`, and update `LLM_FALLBACK_ORDER`.

**Pruning:** when a model starts returning 404 `unavailable for free` (as 3 OpenRouter models did), delete it from `free_tier_models.json` and re-probe.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `probe-catalog` all `SKIP` `CONFIG: …_API_KEY is required` | No keys in `.env.secrets` | Add the provider’s `*_API_KEY` (see `.env.example` per-provider sections). |
| `FAIL` `HTTP 404 … unavailable for free` | Model rotated to paid (OpenRouter) | Remove it from `free_tier_models.json`, keep the router `openrouter/free` or the surviving `:free` model. |
| `FAIL` `HTTP 402 balance insufficient` | SiliconFlow/Together quota exhausted | Top up or remove; catalog will rank without it. |
| `FAIL` `HTTP 401 ModelError: not supported` | Model not hosted by that provider (e.g. Ollama local `llama3.2:3b` on Ollama Cloud, or opencodego free model) | Use the provider’s hosted name (`gpt-oss:20b` for Ollama Cloud); check `provider_capabilities`. |
| `FAIL` `422 unknown field max_tokens` | Wrong `max_tokens_field` | Flip to `max_completion_tokens` (Groq/Cerebras/OpenRouter/…) or `max_tokens` (NVIDIA/Cloudflare/…). |
| `FAIL` `ReadTimeout` | Provider slow / host Ollama down | Re-probe; increase `--timeout 20`; for Ollama ensure `ollama serve` and `ollama pull phi4-mini:3.8b`. |
| `factory` still uses old order despite fresh catalog | `CATALOG_AUTO_ORDER` not `true` or catalog stale | `grep CATALOG_AUTO_ORDER .env` → set `true`; `cat data/provider_catalog.json | grep generated_at` → re-run `dec probe-catalog` if > `CATALOG_STALE_DAYS`. |
| `answer` order missing a provider that is in `global` | Provider lacks structured output | Expected — `answer`/`code` filter. Use `global` order for `rewrite`/`intent`. |
| `tests` fail `Ambient provider env var present` | Exported `*_API_KEY` in shell | `unset` it; tests enforce hermetic `make_settings()` (`tests/conftest.py:252`). |

**Diagnostics:**
```bash
dec_venv/bin/dec probe-llm --json | jq '.[] | select(.status=="FAIL")'   # wiring
dec_venv/bin/dec probe-catalog --offline --json | jq .recommended_fallback_order
dec_venv/bin/python -c "from data_engineering_copilot.config.settings import settings; print(settings.catalog_auto_order, settings.provider_catalog_path)"
cat data/provider_catalog.json | jq .generated_at, .recommended_fallback_order
```

## Future Work

- **Live `/models` discovery** — optional `--refresh` that calls `GET {base_url}/models` per provider (where supported: OpenRouter, Groq, Together) to auto-discover new `:free` models before probing, reducing manual curation.
- **Embedding catalog** — same pattern for `huggingface`/`nvidia`/`openrouter`/`local-hf` embeddings (dimension-gated via `settings.validate_all()` `EMBEDDING_FALLBACK_ORDER mixes dimensions`).
- **Langfuse telemetry** — record per-probe latency + per-purpose order to `langfuse` traces for drift alerts.
- **CI gate** — hermetic `probe-catalog --offline` in CI to ensure `free_tier_models.json` stays valid without live calls.
