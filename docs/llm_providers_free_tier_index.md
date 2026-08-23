# LLM Provider Free Tier Research — Index

**Last updated**: August 2026

> **Live system:** the curated `free_forever` inventory + live probes + ranked fallback are now codified in `docs/provider_catalog.md` (and `config/free_tier_models.json` → `data/provider_catalog.json` via `dec probe-catalog`). This index is historical research; see `provider_catalog.md` for the current `CATALOG_AUTO_ORDER` system and 2026-08-23 probe results.

## Summary Table

| Provider | Free Tier | Key Limits | File |
|----------|-----------|------------|------|
| SambaNova | Yes (permanent) | 20 req/day, 200K tokens/day per model | [sambanova_free_tier.md](sambanova_free_tier.md) |
| Groq | Yes | Varies by model (see file) | [groq_free_tier.md](groq_free_tier.md) |
| OpenRouter | Yes (limited) | 50 RPD without credits, 1000 RPD with $10+ | [openrouter_free_tier.md](openrouter_free_tier.md) |
| Cerebras | Yes ($5 credit) | $5 on signup, expires ~3 months | [cerebras_free_tier.md](cerebras_free_tier.md) |
| Mistral | Yes ($10/mo) | $10/mo API credits on Free plan | [mistral_free_tier.md](mistral_free_tier.md) |
| Together AI | No explicit free tier | Pay-as-you-go | [together_free_tier.md](together_free_tier.md) |
| Fireworks | No explicit free tier | Pay-as-you-go | [fireworks_free_tier.md](fireworks_free_tier.md) |
| DeepSeek | Balance-based | New accounts get initial balance | [deepseek_free_tier.md](deepseek_free_tier.md) |
| NVIDIA | Yes ($1,000 credit) | $1,000 on signup, expires ~30 days | [nvidia_free_tier.md](nvidia_free_tier.md) |
| Gemini | Yes | 15 RPM, 1,500 RPD (Flash) | [gemini_free_tier.md](gemini_free_tier.md) |
| SiliconFlow | Yes (¥14) | Limited credits on signup | [siliconflow_free_tier.md](siliconflow_free_tier.md) |
| Cloudflare | Yes | 10,000 neurons/month | [cloudflare_free_tier.md](cloudflare_free_tier.md) |
| ZAI | Limited | Varies by model | [zai_free_tier.md](zai_free_tier.md) |
| LLM7 | Unknown | Verify on platform | [llm7_free_tier.md](llm7_free_tier.md) |
| Agnes | Unknown | Verify on platform | [agnes_free_tier.md](agnes_free_tier.md) |
| Helyx | Unknown | Verify on platform | [helyx_free_tier.md](helyx_free_tier.md) |
| AnyAPI | Depends on provider | Proxy to other providers | [anyapi_free_tier.md](anyapi_free_tier.md) |
| OpenCodeZen | Unknown | Verify on platform | [opencodezen_free_tier.md](opencodezen_free_tier.md) |
| OpenCodeGo | Unknown | Verify on platform | [opencodego_free_tier.md](opencodego_free_tier.md) |

## Best Free Tiers for Development/Testing

1. **SambaNova** — 20 req/day permanent, no card required
2. **Groq** — Fast inference, generous free limits
3. **Gemini** — 1,500 RPD on Flash models
4. **OpenRouter** — 50 RPD free models, 1,000 with $10 credit
5. **Mistral** — $10/mo API credits on free plan

## Notes

- Some providers (LLM7, Agnes, Helyx, OpenCodeZen, OpenCodeGo) have limited documentation — verify directly on their platforms
- Free tiers are typically for evaluation/development, not production use
- Rate limits and availability may change — check provider documentation for current details
