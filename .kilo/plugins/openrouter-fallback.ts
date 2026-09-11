import type { Plugin } from "@opencode-ai/plugin"

const LADDER = [
  "poolside/laguna-xs-2.1:free",
  "cohere/north-mini-code:free",
  "nvidia/nemotron-3-super-120b-a12b:free",
  "nvidia/nemotron-3-ultra-550b-a55b:free",
]

export const OpenRouterFallbackPlugin: Plugin = async ({ client }) => {
  return {
    "chat.params": async (input, output) => {
      if (input.model.providerID !== "openrouter") return
      if (!LADDER.includes(input.model.id)) return
      if (output.options.models) return

      // OpenRouter API caps `models` at 3 items ("'models' array must have
      // 3 items or fewer"); LADDER is priority-ordered so keep the top 3.
      const fallbacks = LADDER.filter((m) => m !== input.model.id).slice(0, 3)
      output.options.models = fallbacks

      await client.app.log({
        body: {
          service: "openrouter-fallback",
          level: "debug",
          message: "injected model fallbacks",
          extra: { primary: input.model.id, fallbacks },
        },
      })
    },
  }
}
