# Cloudflare Workers AI Free Tier — Research Summary

**Last verified**: August 2026, via developers.cloudflare.com/workers-ai.

## What is Cloudflare Workers AI?

Cloudflare's AI inference platform running on their edge network.

**Base URL**: `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run`

## Free Tier Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Neurons/month | 10,000 | Free tier allowance |
| Requests/min | 300 | Per account |

1 Neuron = 1 unit of compute (varies by model).

## Available Models

Access to various models including:
- Llama models
- Mistral models
- And more

## Getting an API Key

1. Sign up at [dash.cloudflare.com](https://dash.cloudflare.com)
2. Go to **AI** → **Workers AI**
3. Create an API token
4. Export: `export CLOUDFLARE_API_KEY="..."`

## Known Gotchas

- Free tier has limited compute units
- Rate limits apply per account
- Production use requires paid plan
