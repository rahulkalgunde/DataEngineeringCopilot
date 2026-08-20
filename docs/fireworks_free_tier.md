# Fireworks Free Tier — Research Summary

**Last verified**: August 2026, via docs.fireworks.ai.

## What is Fireworks?

Fireworks provides serverless inference, dedicated endpoints, fine-tuning, and GPU clusters for open-source models.

**Base URL**: `https://api.fireworks.ai/inference/v1`

## Free Tier Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Free credits | None explicit | Pay-as-you-go only |

Fireworks does not appear to have a standing free tier.

## Available Models

Extensive catalog including:
- DeepSeek V3.1
- Kimi models
- GLM models
- Llama models
- And many more

## Getting an API Key

1. Sign up at [app.fireworks.ai](https://app.fireworks.ai)
2. Go to **Settings** → **API Keys** → create a key
3. Export: `export FIREWORKS_API_KEY="..."`

## Known Gotchas

- No explicit free tier documented
- Pricing varies by model
- Supports OpenAI, Anthropic, and native SDKs
- Batch inference available for cost savings
