# Groq Free Tier — Research Summary

**Last verified**: August 2026, via console.groq.com/docs.

## What is Groq?

Groq provides fast LLM inference on their custom LPU (Language Processing Unit) chips. API is **OpenAI-compatible**.

**Base URL**: `https://api.groq.com/openai/v1`

## Free Tier Limits

No credit card required. Free plan:

| Limit | Value | Notes |
|-------|-------|-------|
| Requests/min | Varies by model | See table below |
| Requests/day | Varies by model | See table below |
| Tokens/min | Varies by model | See table below |

## Available Free Tier Models (August 2026)

| Model ID | RPM | RPD | TPM | TPD |
|----------|-----|-----|-----|-----|
| meta-llama/llama-prompt-guard-2-22m | 30 | 14,400 | 15K | 500K |
| meta-llama/llama-prompt-guard-2-86m | 30 | 14,400 | 15K | 500K |
| openai/gpt-oss-120b | 30 | 1,000 | 8K | 200K |
| openai/gpt-oss-20b | 30 | 1,000 | 8K | 200K |
| openai/gpt-oss-safeguard-20b | 30 | 1,000 | 8K | 200K |
| qwen/qwen3.6-27b | 30 | 1,000 | 8K | 200K |
| groq/compound | 30 | 250 | 70K | - |
| groq/compound-mini | 30 | 250 | 70K | - |
| canopylabs/orpheus-arabic-saudi | 10 | 100 | 1.2K | 3.6K |
| canopylabs/orpheus-v1-english | 10 | 100 | 1.2K | 3.6K |
| whisper-large-v3 | 20 | 2K | - | - |
| whisper-large-v3-turbo | 20 | 2K | - | - |

## Developer Tier (paid)

Higher limits + Batch and Flex processing. Base limits shown; higher available for enterprise.

## Getting an API Key

1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** → create a key
3. Export: `export GROQ_API_KEY="gsk_..."`

## Known Gotchas

- Rate limits apply at organization level, not individual users
- Cached tokens do not count towards rate limits
- Free tier is evaluation-grade
