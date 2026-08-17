---
name: provider-onboarding
description: Use when adding, renaming, or debugging an LLM / embedding / rerank provider in DataEngineeringCopilot — any task touching factory.py build_* provider branches, settings.py provider blocks, provider_api_key_map, LLM_FALLBACK_ORDER, per-purpose LLM chains, rate limiters, cli.py health output, or dec probe-llm. Triggers: new provider, onboard, provider, llm_provider, embedding_provider, api key missing, fallback order, opencodezen, opencodego, cloudflare, groq, cerebras, gemini, deepseek, zai, siliconflow, together, fireworks, llm7, agnes, ollama_cloud.
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

LLM-only providers (groq, cerebras, cloudflare, opencodezen, opencodego,
sambanova, mistral, deepseek, zai, siliconflow, together, fireworks,
llm7, agnes, ollama_cloud)
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
   - Optionally add `acme` to the `llm_fallback_order` default. **Remember:**
     the class default is only half the battle — `.env` overrides it (see
     step 4).
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
4. **`.env` + `.env.example`** — document `ACME_API_KEY`, model, base URL,
   RPM/RPD. **Critical:** if `.env` has a hardcoded `LLM_FALLBACK_ORDER`, it
   overrides the class-level default in `settings.py` — update it to include
   the new provider. Also update `.env.example` to match. Secrets belong in
   `.env.secrets` (gitignored).
   - Verify after editing: `python -c "from data_engineering_copilot.config.settings import settings; print('acme' in settings.llm_fallback_order)"`
5. **`tests/conftest.py`** — **required for hermetic isolation**:
   - Add `"acme_api_key": ""` to the `defaults` dict in `make_settings()`.
     Without this, pydantic-settings falls back to `os.environ` and ambient
     API keys silently leak into tests, causing the factory to build a
     `ProviderFallbackChain` instead of the expected bare `LLMClient`.
   - Add `"ACME_API_KEY"` to the `_AMBIENT_PROVIDER_VARS` list in
     `pytest_configure()`. This catches any env vars that `make_settings`
     doesn't neutralize, raising `RuntimeError` instead of silently passing.
6. **Tests** — `tests/unit/test_provider_factory.py`:
   - `test_{provider}_uses_base_url_model_and_max_completion_tokens_field`
     (mirror `test_opencodego_*`): assert model/base_url/`_max_tokens_field`/`_max_tokens`.
   - `test_{provider}_missing_api_key_raises` (build settings, blank the key
     via `object.__setattr__`, expect `ValueError`).
   - If you changed the `llm_fallback_order` default, update
     `tests/unit/test_ragas_evaluation.py::test_build_runtime_adaptive_judge_has_no_pinned_primary`
     to pass a placeholder key for the new provider.
7. **Probe** — `dec_venv/bin/dec probe-llm --providers acme --json` (live paid
   call — only with explicit user approval). The probe auto-discovers targets
   from `llm_provider`, per-purpose providers, and `llm_fallback_order`; a
   provider not referenced anywhere yields `[]`.
   - **If probe returns `[]`:** the provider isn't in the *actual* fallback
     order (`.env` may override the class default — see step 4), isn't set as
     `llm_provider`, and has no per-purpose provider referencing it. Check:
     `python -c "from data_engineering_copilot.config.settings import settings; print(settings.llm_fallback_order)"`.
   - The probe uses the global `settings` singleton (created at import time
     from `.env`/`.env.secrets`), not a fresh `AppSettings()`. A provider
     added to `settings.py` defaults but missing from `.env` will not appear.
   - Manual verification without the probe: build a client directly via
     `_build_purpose_llm_client(provider='acme', model='', app_settings=settings,
     purpose='answer')` and call `await client.generate(prompt='pong', max_tokens=10)`.
     `LLMClient.generate()` takes a `prompt` string (not `messages`); the
     client wraps it via `build_chat_messages()`. The `api_key` attribute is
     public (not `_api_key`).

## Verification checklist (after onboarding)

1. **Settings load:** `python -c "from ...settings import settings; print(settings.acme_api_key.get_secret_value())"` — key must be non-empty.
2. **Fallback order:** `python -c "from ...settings import settings; print('acme' in settings.llm_fallback_order)"` — must be `True` if added to fallback chain.
3. **Client builds:** `_build_purpose_llm_client(provider='acme', model='', app_settings=settings, purpose='answer')` — must return an `LLMClient` (not `None`).
4. **Live probe:** `dec probe-llm --providers acme --json` — must return `status: "OK"`.
5. **Tests pass:** `pytest tests/unit/test_provider_factory.py -v -n 0` — all pass, including new provider tests.
6. **Hermetic isolation:** `pytest tests/unit/test_ragas_evaluation.py::TestRagasEvaluator::test_build_runtime_adaptive_judge_has_no_pinned_primary -v -n 0` — passes with new provider's placeholder key.

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

- **opencode.json ≠ project LLM provider.** `opencode.json` configures the
  OpenCode assistant's model provider (what powers *this* coding agent).
  The project's LLM provider configures DataEngineeringCopilot's own RAG
  pipeline routing. They are independent — adding a provider to one does NOT
  add it to the other. Both need separate entries.
- **Hermetic test isolation is mandatory.** When adding a new provider, you
  MUST update `tests/conftest.py` (step 5 above). Skipping this causes
  ambient API keys to silently leak into tests — the test appears to pass
  but is exercising the wrong provider. The symptom: `build_global_llm_client`
  returns a `ProviderFallbackChain` instead of a bare `LLMClient`, or the
  fallback chain includes unexpected providers.
- **401 is not always an auth failure.** opencodego (and other OpenAI-compatible
  gateways) return HTTP 401 with `ModelError: "Model X is not supported"` for
  an unsupported model — not a bad key. `_default_categorizer` in
  `provider_fallback.py` inspects `LLMClientError.response_body` and
  `httpx.HTTPStatusError.response.text` for model-not-supported patterns,
  classifying them as `INVALID_REQUEST` (short cooldown) instead of
  `AUTHENTICATION_ERROR` (60s cooldown). Verify by probing the same key with
  the provider's own default model (200) vs the offending model (401
  `ModelError`).
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
- **`.env` overrides class defaults.** `pydantic-settings` reads `LLM_FALLBACK_ORDER`
  from `os.environ` first; if `.env` has it hardcoded, the class-level default
  in `settings.py` is ignored. After adding a provider to the fallback chain,
  always verify `settings.llm_fallback_order` includes it, and update both
  `.env` and `.env.example`. The same applies to any `Field(default_factory=...)`
  — pydantic-settings resolves env vars before falling back to defaults.
- **`max_tokens_field` choice is provider-specific.** Wrong field = HTTP 422
  (`unknown field`). Use `"max_completion_tokens"` for groq/cerebras/opencodezen/
  opencodego/gemini/mistral; use `"max_tokens"` for nvidia/cloudflare/ollama/
  deepseek/zai/siliconflow/together/fireworks/llm7/agnes/ollama_cloud.
  If a provider returns 422, check this field first.
- **Rate limiter field naming is `{provider}_rpm_limit` / `{provider}_rpd_limit`.**
  These must match what `_build_provider_rate_limiters()` reads. A mismatch
  means the rate limiter silently uses defaults (or `None`).
- After changing `opencode.json`/skills: opencode config is not hot-reloaded —
  restart opencode.
- **SiliconFlow base URL is `.com`, not `.cn`.** `api.siliconflow.cn` rejects
  valid keys (401). Use `https://api.siliconflow.com/v1`. No free tier —
  requires paid credits.
- **AnyAPI.ai free tier uses `:free` suffix models.** Paid models (e.g.
  `gpt-4o-mini`) return 403 `key_model_access_denied`. Use
  `nvidia/nemotron-3-nano-30b-a3b:free` or similar. Some `:free` models
  return 404 (no provider routing) — test with curl first.
- **Ollama Cloud uses cloud-only models.** `llama3.2:3b` is a local-only model.
  Use cloud-available models like `gpt-oss:20b`. Check available models via
  `curl https://ollama.com/api/tags`.
