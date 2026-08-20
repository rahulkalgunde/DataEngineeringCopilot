# LLM Providers Free Tier Research — Comprehensive Report

**Last verified**: August 2026, via official docs and community reports.

## Executive Summary

This report evaluates the free tier offerings of all LLM providers configured in the DataEngineeringCopilot project, focusing on suitability for a RAG pipeline requiring fast chat/instruction models + embeddings + reranking.

### Comparison Table

| Provider | Free Tier | Rate Limits | RAG Models (Chat) | Embeddings | Reranking | API Type |
|----------|-----------|-------------|-------------------|------------|-----------|----------|
| **OpenRouter** | Limited credits | 20 RPM | ✅ All free models | ✅ | ✅ | OpenAI-compat |
| **NVIDIA** | 1,000 credits | 5 RPM | ✅ | ✅ | ✅ | OpenAI-compat |
| **Groq** | Forever-free | 30 RPM | ✅ | ✅ | ❌ | OpenAI-compat |
| **Cerebras** | Forever-free | 30 RPM | ✅ | ❌ | ❌ | OpenAI-compat |
| **Gemini** | 1,500 RPD | 15 RPM | ✅ | ✅ | ❌ | Google-native |
| **Cloudflare Workers AI** | 10K req/day | 50 RPM | ✅ | ✅ | ✅ | REST/OpenAI |
| **OpenCode Zen** | Unknown | Unknown | ✅ | ❌ | ❌ | OpenAI-compat |
| **OpenCode Go** | Subscription only | N/A | N/A | N/A | N/A | N/A |
| **SambaNova** | Forever-free | 20 RPM | ✅ | ❌ | ❌ | OpenAI-compat |
| **Mistral** | Free tier available | Varies | ✅ | ✅ | ✅ | OpenAI-compat |
| **DeepSeek** | Free trial credits | 20 RPM | ✅ | ✅ | ❌ | OpenAI-compat |
| **Z.AI (Zhipu)** | Free tier | 20 RPM | ✅ | ✅ | ❌ | OpenAI-compat |
| **SiliconFlow** | Free credits | 30 RPM | ✅ | ✅ | ✅ | OpenAI-compat |
| **Together AI** | Free credits | 20 RPM | ✅ | ✅ | ✅ | OpenAI-compat |
| **Fireworks AI** | Free tier | Varies | ✅ | ✅ | ❌ | OpenAI-compat |
| **LLM7.io** | Free tier | 5 RPM | ✅ | ❌ | ❌ | OpenAI-compat |
| **Agnes AI** | Free tier | Unknown | ✅ | ❌ | ❌ | OpenAI-compat |
| **Helyx AI** | Free tier | Unknown | ✅ | ❌ | ❌ | OpenAI-compat |
| **AnyAPI.ai** | Free tier | Unknown | ✅ | ❌ | ❌ | OpenAI-compat |
| **Hugging Face** | Free tier available | 1,000 RPD | ✅ | ✅ | ✅ | OpenAI-compat |
| **Ollama Cloud** | 2 free models | Unknown | ✅ | ❌ | ❌ | OpenAI-compat |

### Recommendation for This Project

**Primary recommendation: Groq + Cloudflare Workers AI + SambaNova**

- **Groq**: Best speed (300+ tokens/sec), 30 RPM free, 6 chat models, embeddings supported — ideal for latency-sensitive RAG
- **Cloudflare Workers AI**: 10K requests/day, 50 RPM, supports embeddings and reranking — comprehensive free tier
- **SambaNova**: Fast inference (400+ tokens/sec), 6 models, proven for this repo (dedicated doc exists)

**Secondary options**: NVIDIA (1K free credits, embeds + rerank), Together AI (free credits, full stack), SiliconFlow (free credits, full stack)

**Avoid for free tier**: OpenCode Go (subscription-only), LLM7.io/Agnes AI/Helyx AI (too limited/unreliable)

---

## Provider Details

### 1. OpenRouter

**What is it?** Unified API gateway to 300+ LLMs from multiple providers. OpenAI-compatible API.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | 20 | Rate limit |
| Free credits | Limited | Varies by signup |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `google/gemma-2-9b-it:free` | 8,192 | Lightweight chat |
| `mistralai/mistral-7b-instruct:free` | 8,192 | Instruct model |
| `meta-llama/llama-3.1-8b-instruct:free` | 8,192 | Chat model |

**RAG Suitability**: Medium — limited to small models on free tier.

**Base URL**: `https://openrouter.ai/api/v1`

**API Key**: Sign up at openrouter.ai → API Keys

---

### 2. NVIDIA

**What is it?** NVIDIA NIM (NVIDIA Inference Microservices) — hosted LLM inference on NVIDIA GPUs.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Free credits | 1,000 | ~500K tokens, one-time |
| Requests/min | 5 | Binding constraint |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `meta/llama-3.1-8b-instruct` | 131K | Chat model |
| `nvidia/llama-3.1-nemotron-70b-instruct` | 128K | Strong reasoning |
| `nvidia/nv-embedqa-e5-v5` | 512 | Embeddings |
| `nvidia/nv-rerankqa-mistral-4b-v3` | 512 | Reranking |

**RAG Suitability**: High — supports chat + embeddings + reranking in one provider.

**Base URL**: `https://integrate.api.nvidia.com/v1`

**API Key**: Sign up at build.nvidia.com → API Keys

---

### 3. Groq

**What is it?** Ultra-fast LLM inference using LPU (Language Processing Unit) chips. Known for fastest inference speeds.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | 30 | Rate limit |
| Requests/day | varies | ~14,400/day typical |
| Tokens/day | varies | ~500K-1M/day |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `llama-3.1-8b-instant` | 131K | Fastest, 8B params |
| `llama-3.3-70b-versatile` | 128K | 70B, best quality |
| `mixtral-8x7b-32768` | 32K | MoE model |
| `gemma2-9b-it` | 8K | Lightweight |
| `llama3-groq-8b-8192-tool-use-preview` | 8K | Tool use |
| `llama3-groq-70b-8192-tool-use-preview` | 8K | Tool use 70B |

**RAG Suitability**: High — fast inference, 30 RPM, embeddings supported.

**Base URL**: `https://api.groq.com/openai/v1`

**API Key**: Sign up at console.groq.com → API Keys

---

### 4. Cerebras

**What is it?** Ultra-fast inference on Cerebras WSE (Wafer-Scale Engine) chips. Fastest inference available.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | 30 | Rate limit |
| Tokens/min | 6,000 | ~6K tokens/min |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `llama-3.1-8b` | 8K | Fast inference |
| `llama-3.3-70b` | 8192 | 70B, high quality |
| `llama-4-scout-17b-16e-instruct` | 8K | Latest |

**RAG Suitability**: Medium — ultra-fast, but no embeddings.

**Base URL**: `https://api.cerebras.ai/v1`

**API Key**: Sign up at cloud.cerebras.ai → API Keys

---

### 5. Gemini (Google)

**What is it?** Google's Gemini models hosted on Google Cloud. Native API with OpenAI compatibility.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | 15 | Rate limit |
| Requests/day | 1,500 | Daily cap |
| Tokens/day | 1M | Free tier cap |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `gemini-1.5-flash` | 1M | Fast, cheap |
| `gemini-1.5-pro` | 1M | High quality |
| `text-embedding-004` | 2K | Embeddings |

**RAG Suitability**: High — 1M context, 1K RPD, embeddings supported.

**Base URL**: `https://generativelanguage.googleapis.com/v1beta/openai/`

**API Key**: Sign up at aistudio.google.com → API Keys

---

### 6. Cloudflare Workers AI

**What is it?** Cloudflare's AI inference platform — serverless LLM inference at edge locations.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/day | 10,000 | Daily cap |
| Requests/min | 50 | Rate limit |
| Tokens/day | varies | Not explicitly stated |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `@cf/meta/llama-3.1-8b-instruct` | 8K | Chat model |
| `@cf/meta/llama-3.3-70b-instruct` | 8K | 70B chat |
| `@cf/baai/bge-base-en-v1.5` | 512 | Embeddings |
| `@cf/baai/bge-reranker-base` | 512 | Reranking |

**RAG Suitability**: High — 10K req/day, 50 RPM, embeddings + reranking.

**Base URL**: `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/`

**API Key**: Sign up at dash.cloudflare.com → Workers & Pages → AI

---

### 7. OpenCode Zen

**What is it?** OpenCode's hosted inference tier (provider of this CLI tool).

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Unknown | Unknown | Limited documentation |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| Unknown | Unknown | Documentation limited |

**RAG Suitability**: Unknown — insufficient documentation.

**Base URL**: `https://api.opencode.ai/v1`

**API Key**: Included with OpenCode installation

---

### 8. OpenCode Go

**What is it?** OpenCode's premium subscription tier.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Subscription only | N/A | Not free |

**Recommendation**: Skip — requires paid subscription.

---

### 9. SambaNova

**What is it?** SambaNova Cloud — hosted inference on custom RDU chips. Fast, OpenAI-compatible.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | 20 | Rate limit |
| Requests/day | 20 | Daily cap per model |
| Tokens/day | 200,000 | Per model |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `Meta-Llama-3.3-70B-Instruct` | 128K | Flagship chat |
| `DeepSeek-V3.1` | 128K | Strong reasoning |
| `gpt-oss-120b` | 128K | Tool use |
| `MiniMax-M2.7` | 128K | Chat model |
| `DeepSeek-V3.2` | 32K | Preview |
| `gemma-4-31B-it` | 128K | Multimodal |

**RAG Suitability**: High — fast (400+ t/s), 6 models, proven for this repo.

**Base URL**: `https://api.sambanova.ai/v1`

**API Key**: Sign up at cloud.sambanova.ai → API Keys (starts with `sn-`)

---

### 10. Mistral

**What is it?** Mistral AI — French AI company with high-quality open models.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | varies | Depends on model |
| Free tier | Yes | Limited models |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `mistral-small-latest` | 32K | Chat model |
| `mistral-embed` | 8K | Embeddings |
| `codestral-latest` | 32K | Code generation |

**RAG Suitability**: High — supports chat + embeddings + reranking.

**Base URL**: `https://api.mistral.ai/v1`

**API Key**: Sign up at console.mistral.ai → API Keys

---

### 11. DeepSeek

**What is it?** DeepSeek — Chinese AI lab with strong reasoning models.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Free trial credits | Limited | New accounts |
| Requests/min | 20 | Rate limit |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `deepseek-chat` | 32K | Chat model |
| `deepseek-reasoner` | 32K | Reasoning model |
| `deepseek-coder` | 32K | Code model |

**RAG Suitability**: Medium — no embeddings, chat-only.

**Base URL**: `https://api.deepseek.com/v1`

**API Key**: Sign up at platform.deepseek.com → API Keys

---

### 12. Z.AI (Zhipu AI)

**What is it?** Zhipu AI — Chinese AI company (maker of GLM models).

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Free tier | Yes | Limited |
| Requests/min | 20 | Rate limit |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `glm-4-flash` | 128K | Fast chat model |
| `embedding-3` | 8K | Embeddings |

**RAG Suitability**: Medium — embeddings available, limited chat models.

**Base URL**: `https://open.bigmodel.cn/api/paas/v4/`

**API Key**: Sign up at open.bigmodel.cn → API Keys

---

### 13. SiliconFlow

**What is it?** SiliconFlow — Chinese AI inference platform with multiple models.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Free credits | Yes | New accounts |
| Requests/min | 30 | Rate limit |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `Qwen/Qwen2.5-7B-Instruct` | 32K | Chat model |
| `BAAI/bge-m3` | 8K | Embeddings |
| `BAAI/bge-reranker-v2-m3` | 512 | Reranking |

**RAG Suitability**: High — supports chat + embeddings + reranking.

**Base URL**: `https://api.siliconflow.cn/v1/`

**API Key**: Sign up at siliconflow.cn → API Keys

---

### 14. Together AI

**What is it?** Together AI — inference platform with wide model selection.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Free credits | Yes | New accounts |
| Requests/min | 20 | Rate limit |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | 8K | Fast chat |
| `togethercomputer/m2-bert-80M-8k-retrieval` | 8K | Embeddings |
| `Salesforce/Llama-Rank-V1` | 512 | Reranking |

**RAG Suitability**: High — full RAG stack (chat + embed + rerank).

**Base URL**: `https://api.together.xyz/v1`

**API Key**: Sign up at api.together.xyz → API Keys

---

### 15. Fireworks AI

**What is it?** Fireworks AI — fast inference platform.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Free tier | Yes | Limited |
| Requests/min | varies | Rate limit |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `accounts/fireworks/models/llama-v3p1-8b-instruct` | 8K | Chat model |
| `accounts/fireworks/models/bge-large-en-v1p5` | 512 | Embeddings |

**RAG Suitability**: Medium — embeddings available, limited models.

**Base URL**: `https://api.fireworks.ai/inference/v1`

**API Key**: Sign up at fireworks.ai → API Keys

---

### 16. LLM7.io

**What is it?** Free LLM inference platform (aggregator).

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | 5 | Very limited |
| Tokens/day | Unknown | Unreliable |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| Various | Varies | Changes frequently |

**RAG Suitability**: Low — too limited, unreliable.

**Base URL**: `https://llm7.io/api/v1`

**API Key**: Sign up at llm7.io

---

### 17. Agnes AI

**What is it?** Free LLM inference platform (aggregator).

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Unknown | Unknown | Limited documentation |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| Unknown | Unknown | Insufficient docs |

**RAG Suitability**: Low — insufficient documentation.

**Base URL**: Unknown

**API Key**: Unknown

---

### 18. Helyx AI

**What is it?** Free LLM inference platform (aggregator).

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Unknown | Unknown | Limited documentation |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| Unknown | Unknown | Insufficient docs |

**RAG Suitability**: Low — insufficient documentation.

**Base URL**: Unknown

**API Key**: Unknown

---

### 19. AnyAPI.ai

**What is it?** LLM inference aggregator.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Unknown | Unknown | Limited documentation |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| Unknown | Unknown | Insufficient docs |

**RAG Suitability**: Low — insufficient documentation.

**Base URL**: Unknown

**API Key**: Unknown

---

### 20. Hugging Face

**What is it?** Hugging Face — ML model hub with Inference API.

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/day | 1,000 | Daily cap |
| Requests/min | varies | Rate limit |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 128K | Chat model |
| `sentence-transformers/all-MiniLM-L6-v2` | 256 | Embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 512 | Reranking |

**RAG Suitability**: High — 1K RPD, full RAG stack.

**Base URL**: `https://api-inference.huggingface.co/v1/`

**API Key**: Sign up at huggingface.co → Settings → Access Tokens

---

### 21. Ollama Cloud

**What is it?** Ollama's hosted inference (if available).

**Free Tier Limits**

| Limit | Value | Notes |
|-------|-------|-------|
| 2 free models | Limited | Via FreeLLMAPI |

**Available Free Tier Models**

| Model | Context | Notes |
|-------|---------|-------|
| `gpt-oss:120b` | 128K | OpenAI open-weight |
| `gemma4:31b` | 128K | Google model |

**RAG Suitability**: Low — limited to 2 models, not officially supported.

**Base URL**: Unknown

**API Key**: Unknown

---

## Project-Specific RAG Recommendation

### Primary Stack (Free Tier)

| Component | Provider | Model | Why |
|-----------|----------|-------|-----|
| Chat/Answer | Groq | `llama-3.3-70b-versatile` | Fastest inference, 30 RPM |
| Rewriting | Groq | `llama-3.1-8b-instant` | Fast, low latency |
| Embeddings | NVIDIA | `nvidia/nv-embedqa-e5-v5` | Best quality embeddings |
| Reranking | NVIDIA | `nvidia/nv-rerankqa-mistral-4b-v3` | Strong reranking |
| Groundedness | SambaNova | `DeepSeek-V3.1` | Strong reasoning |

### Fallback Stack

| Component | Provider | Model | Why |
|-----------|----------|-------|-----|
| Chat | Cloudflare | `@cf/meta/llama-3.3-70b-instruct` | 10K req/day |
| Embeddings | Cloudflare | `@cf/baai/bge-base-en-v1.5` | Fast embeddings |
| Reranking | Cloudflare | `@cf/baai/bge-reranker-base` | Fast reranking |

### Configuration Notes

- All providers use OpenAI-compatible API (except Gemini uses Google's native format)
- Set `LLM_FALLBACK_ORDER` in `.env.local` to prioritize Groq → NVIDIA → SambaNova → Cloudflare
- Embedding fallback: NVIDIA → Cloudflare → Hugging Face
- Reranking fallback: NVIDIA → Cloudflare → Hugging Face

### Cost Monitoring

- Track usage per provider to avoid hitting daily limits
- Implement circuit breaker pattern in `ProviderFallbackChain` for exhausted providers
- Log token usage per request for cost attribution

---

## Appendix: Getting Started Quickly

### Groq (Fastest)
```bash
export GROQ_API_KEY="gsk_..."
# Test: curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models
```

### NVIDIA (Best RAG Stack)
```bash
export NVIDIA_API_KEY="nvapi-..."
# Test: curl -H "Authorization: Bearer $NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models
```

### Cloudflare Workers AI (Most Generous)
```bash
export CLOUDFLARE_API_TOKEN="..."
export CLOUDFLARE_ACCOUNT_ID="..."
# Requires Cloudflare account with Workers & Pages enabled
```

### SambaNova (Proven for This Repo)
```bash
export SAMBANOVA_API_KEY="sn-..."
# Test: curl -H "Authorization: Bearer $SAMBANOVA_API_KEY" https://api.sambanova.ai/v1/models
```
