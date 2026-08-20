# Cerebras Free Tier — Research Summary

**Last verified**: August 2026, via cerebras.ai/pricing.

## What is Cerebras?

Cerebras provides fast LLM inference on their custom wafer-scale chips. Claims 20x faster than OpenAI and Anthropic.

**Base URL**: `https://api.cerebras.ai/v1`

## Free Tier Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Credits | $5 | One-time on signup |
| Rate limits | Lower than Developer | 10x lower than paid tier |

## Available Models

Access to all Cerebras-powered models during free trial.

## Getting an API Key

1. Sign up at [cloud.cerebras.ai](https://cloud.cerebras.ai)
2. Get $5 in free credits automatically
3. Export: `export CEREBRAS_API_KEY="..."`

## Developer Tier

- Self-serve starting at $10
- 10x higher rate limits than free tier
- Higher priority processing

## Known Gotchas

- $5 credits expire (typically after ~3 months)
- Free tier has community support only (Discord)
- Enterprise required for custom model weights and fine-tuning
