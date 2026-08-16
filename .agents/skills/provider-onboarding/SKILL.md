---
name: provider-onboarding
description: Use when adding, renaming, or debugging an LLM / embedding / rerank provider in DataEngineeringCopilot — any task touching factory.py build_* provider branches, settings.py provider blocks, provider_api_key_map, LLM_FALLBACK_ORDER, per-purpose LLM chains, rate limiters, cli.py health output, or dec probe-llm. Triggers: new provider, onboard, provider, llm_provider, embedding_provider, api key missing, fallback order, opencodezen, opencodego, cloudflare, groq, cerebras, gemini.
---

# DataEngineeringCopilot Provider Onboarding

Every provider is a **constructor-parameter difference**, not a new SDK.
All LLM calls go through the single httpx-based `LLMClient`
(`infrastructure/llm_client.py`) which speaks the OpenAI-compatible
`/v1/chat/completions` protocol. **Never import provider SDKs** (`openai`,
`anthropic`, ...). Embeddings use `OpenAICompatibleEmbeddings` or a native
route; rerank uses `infrastructure/rerank_clients.py`.

## Provider classes in this project

| Class | Surface | Endpoint/protocol |
|---|---|---|
| `LLM` — LLMClient | chat completions | `POST /chat/completions` (OpenAI-compatible) |
| `Embedding` — OpenAICompatibleEmbeddings | embeddings | `POST /embeddings`; `include_provider_param=True` adds `{"provider":{"truncate":"END"}}` (OpenRouter) |
| `Embedding` — native | feature extraction | `HuggingFaceServerlessEmbeddings` (`feature-extraction` route), `AsyncOllamaEmbeddings` (`/api/embed`) |
| `Rerank` — LLMClient | rerank endpoints | OpenRouter `/rerank`, NVIDIA retrieval reranking, HF, or local cross-encoder |

LLM-only providers (groq, cerebras, cloudflare, opencodezen, opencodego)
have **no** embedding/rerank branches — do not add them there.

## Onboarding checklist (LLM provider, e.g. `acme`)

1. **`config/settings.py`**
   - Add a settings block (copy the `opencodego` block): `acme_api_key`
     (`SecretStr`), `acme_model`, `acme_base_url`, `acme_rpm_limit`,
     `acme_rpd_limit`. Use `Field(validation_alias=AliasChoices(...))` ONLY for
     genuine external/legacy aliases (e.g. `HF_TOKEN`, `NVIDIA_NIM_*`) — not
     for your own old names.
   - Add `acme_{purpose}_llm_model` per-purpose overrides for all 7 purposes
     (answer/rewrite/groundedness/intent/enrichment/evaluation/code).
   - Optionally add `acme` to the `llm_fallback_order` default.
   - Register in `_validate_provider_api_keys`'s `provider_api_key_map`
     (sets `field_name` + the env-var name used in the error message).
2. **`factory.py`**
   - `_build_provider_rate_limiters()` — add `elif p == "acme"` using
     `acme_rpm_limit` / `acme_rpd_limit`.
   - `_build_purpose_llm_client()` — add `if eff_provider == "acme":` branch:
     read key (raise `ValueError("ACME_API_KEY is required ...")` if empty),
     return `LLMClient(base_url=..., model=eff_model, api_key=..., max_tokens=purpose_max_tokens,
     rate_limiter=rate_limiter)`. Set `max_tokens_field`:
     - `"max_completion_tokens"` → groq/cerebras/opencode*/gemini-style OpenAI-compatible services
     - `"max_tokens"` → nvidia/cloudflare/ollama
     Wrong field = 422 (`unknown field`) on some providers.
   - Append the name to the `Unsupported LLM provider` message list.
3. **`cli.py`** — `dec health` branch: `elif llm_provider == "acme": print(...)`.
4. **`.env.example`** — document `ACME_API_KEY`, model, base URL, RPM/RPD, and
   the fallback-order example. Secrets belong in `.env.secrets` (gitignored).
5. **Tests** — `tests/unit/test_provider_factory.py`:
   - `test_{provider}_uses_base_url_model_and_max_completion_tokens_field`
     (mirror `test_opencodego_*`): assert model/base_url/`_max_tokens_field`/`_max_tokens`.
   - `test_{provider}_missing_api_key_raises` (build settings, blank the key
     via `object.__setattr__`, expect `ValueError`).
   - If you changed the `llm_fallback_order` default, update
     `tests/unit/test_ragas_evaluation.py::test_build_runtime_adaptive_judge_has_no_pinned_primary`
     to pass a placeholder key for the new provider.
6. **Probe** — `dec_venv/bin/dec probe-llm --providers acme --json` (live paid
   call — only with explicit user approval). The probe auto-discovers targets
   from `llm_provider`, per-purpose providers, and `llm_fallback_order`; a
   provider not referenced anywhere yields `[]`.

## Embedding / rerank providers

Embedding branch lives in `_build_embedding_chain_config()` (nvidia/openrouter/
gemini use `OpenAICompatibleEmbeddings`; huggingface uses the native route).
Rerank branches in `build_rerank_fallback_chain()` read
`rerank_fallback_order` + `{provider}_rerank_model`/`_rerank_url`. Add the
provider to the matching fallback-order default and `validate_all()`'s known
provider sets only when the surface exists.

## Model-resolution priority (LLM)

1. explicit `model` arg → 2. `{provider}_{purpose}_llm_model` → 3.
`{provider}_model` → 4. global `llm_model`. Empty provider/model reuses the
global client.

## Gotchas

- **401 is not always an auth failure.** opencodego (and other OpenAI-compatible
  gateways) return HTTP 401 with `ModelError: "Model X is not supported"` for
  an unsupported model — not a bad key. `factory.py:198` /
  `adaptive_llm_router.py:87` map any 401 → `AUTHENTICATION_ERROR`, so logs show
  `provider_cooldown_set ... category=authentication_error` misleadingly. Verify
  by probing the same key with the provider's own default model (200) vs the
  offending model (401 `ModelError`).
- **Pinning a provider via env must pin its model too.** Model resolution
  priority (above) puts an explicit `purpose_model` before `{provider}_model`,
  so `*_LLM_PROVIDER=opencodego` without `*_LLM_MODEL=deepseek-v4-flash` sends
  the old purpose model to opencodego → 401 `ModelError`.
- OpenAI-compatible vs native endpoints: not every surface is chat-completions
  (HF embeddings use `feature-extraction`; Ollama native uses `/api/chat`).
- Two OpenCode surfaces serve **disjoint model lists**: Zen `/zen/v1`
  (pay-as-you-go, hosts `*-free` models) vs Go `/zen/go/v1` (subscription, no
  `-free`). Free models return `FreeUsageLimitError` (429) when quota is exhausted.
- `provider_api_key_map` validation only fires for providers referenced by an
  LLM purpose or `embedding_provider` — a key listed only in fallback order is
  skipped lazily (chain build logs a warning).
- After changing `opencode.json`/skills: opencode config is not hot-reloaded —
  restart opencode.
