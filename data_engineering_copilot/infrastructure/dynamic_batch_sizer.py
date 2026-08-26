"""Dynamic embedding batch size calculator.

Computes optimal batch size at runtime based on:
1. Model context window (tokens)
2. Average token length of input texts
3. Provider hard batch limit
4. Safety margin for tokenization variance

Formula:
    max_tokens = context_window * safety_margin
    max_by_tokens = max_tokens / avg_tokens_per_text
    batch_size = min(max_by_tokens, provider_limit)
    batch_size = floor(batch_size / 32) * 32  # round down to multiple of 32

Usage:
    sizer = DynamicBatchSizer(settings)
    batch_size = sizer.compute_batch_size("nvidia", texts)
"""

from __future__ import annotations

import statistics

import structlog

logger = structlog.get_logger(__name__)


class DynamicBatchSizer:
    """Compute optimal embedding batch size at runtime."""

    def __init__(self, app_settings) -> None:
        self._settings = app_settings
        self._context_windows = app_settings.embedding_model_context_windows
        self._provider_limits = app_settings.embedding_provider_batch_limits
        self._safety_margin = app_settings.embedding_safety_margin
        self._default_batch = app_settings.embedding_batch_size

    def compute_batch_size(self, provider: str, texts: list[str], model_name: str | None = None) -> int:
        """Compute optimal batch size for a provider given sample texts.

        Args:
            provider: Provider name (nvidia, openrouter, etc.)
            texts: Sample texts to measure token lengths from
            model_name: Model identifier for context window lookup

        Returns:
            Optimal batch size (multiple of 32)
        """
        if not texts:
            return self._default_batch

        provider_limit = self._provider_limits.get(provider, self._default_batch)
        context_window = self._get_context_window(provider, model_name)

        # Measure token lengths from sample texts
        token_counts = [self._count_tokens(t) for t in texts]
        avg_tokens = statistics.mean(token_counts)
        max_tokens = max(token_counts)

        if avg_tokens <= 0:
            return self._default_batch

        # Calculate batch size based on context window
        usable_tokens = context_window * self._safety_margin
        # Ensure even the longest text fits with headroom
        max_texts_by_avg = int(usable_tokens / avg_tokens)
        max_texts_by_max = int(usable_tokens / max_tokens) if max_tokens > 0 else max_texts_by_avg

        # Use the more conservative of avg-based and max-based estimates
        max_texts_by_context = min(max_texts_by_avg, max_texts_by_max)

        # Apply provider hard limit
        batch_size = min(max_texts_by_context, provider_limit)

        # Round down to multiple of 32
        batch_size = (batch_size // 32) * 32

        # Ensure minimum of 32
        batch_size = max(batch_size, 32)

        logger.info(
            "dynamic_batch_size",
            provider=provider,
            model=model_name,
            context_window=context_window,
            avg_tokens=round(avg_tokens, 1),
            max_tokens=max_tokens,
            max_by_context=max_texts_by_context,
            provider_limit=provider_limit,
            result=batch_size,
        )

        return batch_size

    def _get_context_window(self, provider: str, model_name: str | None) -> int:
        """Get context window for a model, falling back to provider default."""
        if model_name and model_name in self._context_windows:
            return self._context_windows[model_name]

        # Try provider-specific defaults
        defaults = {
            "nvidia": 131072,
            "openrouter": 16384,
            "gemini": 8192,
            "huggingface": 512,
        }
        return defaults.get(provider, 8192)

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Estimate token count from character length.

        Uses ~4 chars/token as a fast estimate. For production accuracy,
        could use tiktoken, but this is sufficient for batch sizing.
        """
        return max(len(text) // 4, 1)
