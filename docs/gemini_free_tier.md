# Gemini Free Tier — Research Summary

**Last verified**: August 2026, via cloud.google.com/vertex-ai/generative-ai/pricing.

## What is Gemini?

Google's Gemini models available via Vertex AI or AI Studio.

**Base URL**: `https://generativelanguage.googleapis.com/v1beta`

## Free Tier Limits

Google AI Studio offers a free tier:

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | 15 (Flash) | Varies by model |
| Requests/day | 1,500 (Flash) | Varies by model |
| Tokens/min | 1M (Flash) | Varies by model |

## Available Free Tier Models

| Model | Free RPM | Free RPD |
|-------|----------|----------|
| Gemini 2.5 Flash | 15 | 1,500 |
| Gemini 2.5 Pro | 5 | 250 |
| Gemini 2.0 Flash | 15 | 1,500 |

## Getting an API Key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Create an API key
3. Export: `export GEMINI_API_KEY="AIza..."`

## Known Gotchas

- Free tier has lower rate limits than paid
- Paid pricing varies by model and context length
- Context caching available for cost savings
- Grounding with Google Search has free quota (5,000 queries/month)
