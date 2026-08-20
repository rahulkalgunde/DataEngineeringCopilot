# OpenRouter Free Tier — Research Summary

**Last verified**: August 2026, via openrouter.ai/docs.

## What is OpenRouter?

OpenRouter provides a unified API to access hundreds of AI models. It handles fallbacks automatically and picks the most cost-effective option.

**Base URL**: `https://openrouter.ai/api/v1`

## Free Tier Limits

| Limit | Without Credits | With $10+ Credits |
|-------|-----------------|-------------------|
| Free model requests/day | 50 | 1,000 |

Free models are marked with `:free` suffix or listed at [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0).

## Free Models Router

Use `openrouter/free` to automatically select a free model for your requests.

## Getting an API Key

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Go to **Credits** page (optional: add $10+ for higher free model limits)
3. Create an API key
4. Export: `export OPENROUTER_API_KEY="sk-or-..."`

## Known Gotchas

- Free model limits are 50 RPD without credits, 1000 RPD with $10+ credits
- Credits expire after 1 year of non-use
- Platform fee: 5.5% ($0.80 minimum) when purchasing credits
- BYOK (Bring Your Own Key) has a 5% fee above $25K/month allowance
