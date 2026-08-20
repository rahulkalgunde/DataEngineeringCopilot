# Mistral Free Tier — Research Summary

**Last verified**: August 2026, via mistral.ai/pricing.

## What is Mistral?

Mistral AI provides LLM inference and the Vibe coding agent. API is available via Mistral Studio.

**Base URL**: `https://api.mistral.ai/v1`

## Free Tier Limits

| Limit | Value | Notes |
|-------|-------|-------|
| API credits | $10/mo | Included with Free plan |
| Messages | Limited | Up to 6x free vs paid |
| Web searches | Limited | Up to 5x free |
| Image generation | Limited | Up to 40x free |

## Free Plan Includes

- Access to Mistral models in Studio
- Limited Vibe (chat) access
- $10/mo in API credits
- 100+ connectors

## Getting an API Key

1. Sign up at [chat.mistral.ai](https://chat.mistral.ai) or [console.mistral.ai](https://console.mistral.ai)
2. Free plan gives $10/mo API credits
3. Export: `export MISTRAL_API_KEY="..."`

## Known Gotchas

- Free plan is limited compared to Pro ($14.99/mo)
- API pricing is per million tokens (input/output separate)
- Batch processing reduces price by 50%
- Cached input tokens reduce cost by up to 90%
