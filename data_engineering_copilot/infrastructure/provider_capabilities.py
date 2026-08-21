"""Provider capability flags for the generation layer.

Used to keep generation hyperparameters portable across the many LLM
providers this stack supports. Some parameters (sampling penalties, seed,
structured outputs) are rejected by certain providers (notably Anthropic and
reasoning models), so we only emit them when the provider advertises support.
"""

from __future__ import annotations

# Providers that accept OpenAI-style sampling penalties (frequency_penalty,
# presence_penalty) and top_p. Anthropic's Messages API does not expose these.
SUPPORTS_SAMPLING_PENALTIES: frozenset[str] = frozenset(
    {
        "openai",
        "openrouter",
        "ollama",
        "nvidia",
        "groq",
        "cerebras",
        "gemini",
        "mistral",
        "deepseek",
        "sambanova",
        "cloudflare",
        "zai",
        "siliconflow",
        "together",
        "fireworks",
    }
)

# Providers that accept a `seed` for best-effort deterministic generation.
SUPPORTS_SEED: frozenset[str] = frozenset(
    {
        "openai",
        "openrouter",
        "nvidia",
        "groq",
        "cerebras",
        "gemini",
        "mistral",
        "deepseek",
    }
)

# Providers that support schema-enforced structured outputs (JSON mode /
# JSON schema / guided decoding). Ollama uses `format`; OpenAI-style providers
# use `response_format` with a strict json_schema.
SUPPORTS_STRUCTURED_OUTPUT: frozenset[str] = frozenset(
    {
        "openai",
        "openrouter",
        "ollama",
        "vllm",
        "nvidia",
        "groq",
        "cerebras",
        "gemini",
        "mistral",
        "deepseek",
        "sambanova",
    }
)


def supports_sampling_penalties(provider: str | None) -> bool:
    return provider in SUPPORTS_SAMPLING_PENALTIES


def supports_seed(provider: str | None) -> bool:
    return provider in SUPPORTS_SEED


def supports_structured_output(provider: str | None) -> bool:
    return provider in SUPPORTS_STRUCTURED_OUTPUT
