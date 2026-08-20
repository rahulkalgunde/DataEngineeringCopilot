# DeepSeek Free Tier — Research Summary

**Last verified**: August 2026, via api-docs.deepseek.com.

## What is DeepSeek?

DeepSeek provides LLM inference with competitive pricing and strong reasoning capabilities.

**Base URL (OpenAI Format)**: `https://api.deepseek.com`
**Base URL (Anthropic Format)**: `https://api.deepseek.com/anthropic`

## Free Tier Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Free credits | New accounts get initial balance | Top-up required after |

DeepSeek uses a balance-based system. New accounts may receive initial credits.

## Available Models

| Model | Context | Features |
|-------|---------|----------|
| deepseek-v4-flash | 1M | JSON output, tool calls, Responses API, Anthropic API |
| deepseek-v4-pro | 1M | JSON output, tool calls, Responses API, Anthropic API |

## Pricing (per 1M tokens)

| Model | Input (cache miss) | Input (cache hit) | Output |
|-------|-------------------|-------------------|--------|
| deepseek-v4-flash (off-peak) | $0.22 | $0.007 | $0.66 |
| deepseek-v4-flash (peak) | $0.44 | $0.014 | $1.32 |
| deepseek-v4-pro (off-peak) | $0.66 | $0.022 | $1.98 |
| deepseek-v4-pro (peak) | $1.32 | $0.044 | $3.96 |

Peak hours: 01:00-04:00 and 06:00-10:00 UTC (off-peak is half price).

## Getting an API Key

1. Sign up at [platform.deepseek.com](https://platform.deepseek.com)
2. Top up balance
3. Export: `export DEEPSEEK_API_KEY="..."`

## Known Gotchas

- Pricing varies by peak/off-peak hours
- Cache hits significantly reduce input costs
- Concurrency limits: 2500 (flash), 500 (pro)
- Supports both OpenAI and Anthropic API formats
