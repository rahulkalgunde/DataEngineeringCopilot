# SambaNova Free Tier — Research Summary

**Last verified**: August 2026, via official docs (docs.sambanova.ai, cloud.sambanova.ai) and community reports.

## What is SambaNova?

SambaNova Cloud is a hosted inference API running open LLMs on SambaNova's custom RDU (Reconfigurable Dataflow Unit) chips. API is **OpenAI-compatible** — use the standard OpenAI SDK or raw HTTP with `Authorization: Bearer <key>`.

**Base URL**: `https://api.sambanova.ai/v1`

## Free Tier Limits

No credit card required. Standing forever-free tier (not a trial):

| Limit | Per model | Notes |
|-------|-----------|-------|
| Requests/min | 20 | Binding constraint for most use cases |
| Requests/day | 20 | Resets on fixed daily window (not 24h from last request) |
| Tokens/day | 200,000 | Per model, not shared across models |

Limits are **per model, per account** (not per API key). Switching keys on the same account doesn't reset limits.

New accounts get a one-time **$5 starter credit** (Developer tier) that expires after ~3 months.

## Available Free Tier Models

| Model ID | Context | Notes |
|----------|---------|-------|
| `Meta-Llama-3.3-70B-Instruct` | 128K | Fast general chat; flagship |
| `DeepSeek-V3.1` | 128K | Strong reasoning, MoE |
| `gpt-oss-120b` | 128K | OpenAI open-weight, tool use |
| `MiniMax-M2.7` | 128K | Chat model |
| `DeepSeek-V3.2` | 32K | Preview — may change/be removed |
| `gemma-4-31B-it` | 128K | Preview, multimodal input (text/image/video) |

Speed: ~400+ tokens/sec on flagship models (varies with load and context).

## Developer Tier (paid, pay-as-you-go)

Requires a linked payment method. Higher limits + $5 starter credit:

| Model ID | RPM | RPD |
|----------|-----|-----|
| Meta-Llama-3.3-70B-Instruct | 240 | 48,000 |
| DeepSeek-V3.1 | 60 | 12,000 |
| gpt-oss-120b | 60 | 12,000 |
| MiniMax-M2.7 | 60 | 12,000 |
| DeepSeek-V3.2 | 60 | 12,000 |
| gemma-4-31B-it | 60 | 12,000 |

Dev tier daily cap: **20M tokens/day** across all models.

## Getting an API Key

1. Sign up at [cloud.sambanova.ai](https://cloud.sambanova.ai) (no card needed for free tier)
2. Go to **API Keys** section → generate a key
3. Export it: `export SAMBANOVA_API_KEY="sn-..."`
4. Keys start with `sn-`; up to 25 keys per account

## Using with OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=os.environ["SAMBANOVA_API_KEY"],
)

response = client.chat.completions.create(
    model="DeepSeek-V3.1",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Using with curl

```bash
curl -H "Authorization: Bearer $SAMBANOVA_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
        "model": "Meta-Llama-3.3-70B-Instruct",
        "messages": [{"role":"user","content":"Hello!"}]
      }' \
     -X POST https://api.sambanova.ai/v1/chat/completions
```

## Using with OpenCode (this repo)

OpenCode config uses `@ai-sdk/openai-compatible` npm package. The provider reads `SAMBANOVA_API_KEY` from the environment.

**Fix for "API key not provided" error**: the key must be in the shell environment, not just in `.env.secrets`. Run:

```bash
export SAMBANOVA_API_KEY="sn-...your-key-here"
```

Or add it to `.env.local` (OpenCode may auto-load `.env` files depending on version).

## Known Gotchas

- **Daily limit resets on a fixed window**, not 24h from last request. If you exhaust the limit late in the day, the reset may feel delayed.
- **Free tier is evaluation-grade** — 20 req/day per model is enough for testing/prototyping, not for serving users.
- **Limits are per-account, not per-key** — generating a new key doesn't bypass daily caps.
- **Speed varies** — 400+ t/s is best-case; real-world depends on context length and load.
