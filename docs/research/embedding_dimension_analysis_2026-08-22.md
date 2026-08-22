# Embedding Dimension Analysis — where 768 lives, and what `local-hf` actually is

**Date:** 2026-08-22 · **Scope:** dimension hygiene + provider taxonomy for embeddings
**Trigger:** confusion around `OPENROUTER_EMBEDDING_DIMENSION=2048` vs `LOCAL_EMBEDDING_DIMENSION=768` after switching live production to `nvidia/Nemotron-3-Embed-1B-BF16`.

---

## 1. Dimension audit — where 768 exists today

### 1.1 Phantom (the confusion source)
- `.env:111` → `LOCAL_EMBEDDING_DIMENSION=768`
- **No such setting exists in `AppSettings`.** pydantic-settings `extra="ignore"` silently discards it. It does nothing, but reads like a live knob.

### 1.2 Settings map entries
| Entry | Purpose | Live? |
|---|---|---|
| `embedding_model_dimensions["nomic-embed-text"] = 768` | Ollama model | local integration/e2e only |
| `embedding_model_dimensions["text-embedding-004"] = 768` | Gemini entry | unused provider |
| `default_embedding_dimension = 768` | fallback for models *not found* in the map | ⚠️ latent: unknown model in live chain would silently resolve to legacy geometry |

(`settings.py` `"rewrite"/"intent": 768` values are `purpose_max_tokens` — **token budgets, not dimensions**; false positive.)

### 1.3 Test fixtures
~145 literals across tests: stub vectors `[0.1] * 768`, `StubEmbedder(dimension=768)` — all anchored to the hermetic convention `make_settings()` defaulting to `embedding_provider="ollama"`, `embedding_model_name="nomic-embed-text"` (`tests/conftest.py`). Hermetic tests never hit a real embedder; the number is fixture-consistency only.

### 1.4 Local infra
Makefile / docker-compose pull `nomic-embed-text` into Ollama for manual `make test-integration` / `test-e2e` runs.

### 1.5 Production reality
Live `.env`: `EMBEDDING_PROVIDER=nvidia`; `EMBEDDING_FALLBACK_ORDER='["nvidia","openrouter","huggingface","local-hf"]'` — **every leg resolves to 2048-dim Nemotron**. Ollama is not in the live embedding chain.

### 1.6 Verdict & recommended path — ✅ EXECUTED 2026-08-22 (commit series nomic-A…D)
- Production paths are already 768-free.
- Full repo purge is blocked by test-fixture convention + local Ollama e2e stack (~150 sites).
- **Adopted steps (pending implementation):**
  1. delete dead `.env LOCAL_EMBEDDING_DIMENSION`;
  2. flip `default_embedding_dimension` 768 → 2048;
  3. add `validate_all()` guard: every model in `EMBEDDING_FALLBACK_ORDER` must resolve to the SAME dimension (prevents mid-build index corruption forever);
  4. defer full purge until the Ollama embedder is formally dropped.

---

## 2. Provider taxonomy — what `local-hf` is

**Short answer: `local-hf` has nothing to do with Ollama. It is an in-process HuggingFace `sentence-transformers` encoder running inside the Python app on CPU.** "Local" means *your machine*, not *Ollama*.

### 2.1 The four embedding providers side by side

| provider id | Class | Runtime | Network | Model | Dim |
|---|---|---|---|---|---|
| `nvidia` | `OpenAICompatibleEmbeddings` | NVIDIA NIM REST API | yes (paid/free tier) | `nvidia/nemotron-3-embed-1b` | 2048 |
| `openrouter` | `OpenAICompatibleEmbeddings` | OpenRouter REST | yes | `nemotron-3-embed-1b:free` | 2048 |
| `huggingface` | `HuggingFaceServerlessEmbeddings` | HF **serverless inference API** (remote) | yes (`HF_TOKEN`) | same Nemotron model hosted by HF | 2048 |
| `local-hf` | `LocalSentenceTransformerEmbeddings` | **in-process sentence-transformers, CPU, app's own venv** | no (one-time HF hub download ~1.14 GB, then disk cache) | `nvidia/Nemotron-3-Embed-1B-BF16` loaded locally | 2048 |
| `ollama` / `local` | `AsyncOllamaEmbeddings` | Ollama HTTP daemon | localhost only | `nomic-embed-text` | 768 |

Factory wiring: `factory.build_embedder()` and the per-provider chain builder both branch explicitly — `local-hf → LocalSentenceTransformerEmbeddings(model_name=settings.local_hf_embedding_model)`, while `ollama/local → AsyncOllamaEmbeddings(...)` against `embedding_ollama_base_url`.

### 2.2 How `local-hf` runs
- Loads via a module-level singleton (`_load_model`) using `sentence_transformers.SentenceTransformer(model_name, device="cpu")`; model cached in-process like the cross-encoder reranker; `clear_model_cache()` is the test seam.
- CPU-bound batch inference is offloaded with `asyncio.to_thread`.
- Dual-mode parity: applies the same `query:`/`passage:` prefixes as the NVIDIA endpoint, so local vectors land in the same subspace as the API's (verified cos ≈ 1.0 — module docstring).
- Same `ProviderClient` shape as remote clients (model/call/close/last_usage), so it slots into `ProviderFallbackChain` identically — that's why it can be the last leg of the live order.

### 2.3 Why Ollama cannot serve this model
Ollama serves GGUF_quantized models from its own registry/runtimes. `Nemotron-3-Embed-1B-BF16` is consumed here straight from HuggingFace weights through transformers/sentence-transformers. There is no Ollama route for it in this repo — hence two distinct "local" concepts:
- **local-hf** = local hardware, HF weights, in-process encoder (production-grade, 2048).
- **ollama embedding** = Ollama daemon serving its own nomic model (legacy/test, 768).

### 2.4 Practical consequences
- Switching `EMBEDDING_PROVIDER=local-hf` gives fully offline production at identical geometry to the NVIDIA API (same dim, same prefix modes) — but CPU latency per batch is far higher than the API; that's why it sits LAST in the fallback order rather than first.
- Any provider switch between 2048-legs needs NO reindex (same geometry); switching to/from any 768 model requires `dec reset-index` + reingest (README rule).
- After step 2.1 cleanup above, the only remaining 768 references will be the nomic map entry + tests + local stack — all intentionally quarantined.

---

## 3. Migration closeout (2026-08-22)

All four steps executed via `plans/2026-08-22_12-18_nomic_to_localhf_migration_plan.md`:
- nomic map entry deleted; `default_embedding_dimension` → 2048;
- `AsyncOllamaEmbeddings` + factory branches deleted; embedding fields removed;
- `active_embedding_model_name()` helper; mixed-dim fallback guard in `validate_all()`;
- conftest hermetic defaults → local-hf; e2e/integration fixtures migrated;
- eval-fast hardwired offline; Makefile/ci pulls dropped; `.env` dead var scrubbed;
- remaining 768s are true properties only (Gemini text-embedding-004 map entry,
  purpose_max_tokens budgets) — verified by milestone grep gates.
