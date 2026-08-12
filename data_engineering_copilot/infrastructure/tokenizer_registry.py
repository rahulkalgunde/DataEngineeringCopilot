"""Resolve the token encoder matching an embedding model and declare its
input limits.

The pre-flight token budget for embedding inputs must count with the same
encoder the provider uses, otherwise a text judged safe by a proxy encoder
(cl100k) could exceed the provider's real limit and be silently truncated or
rejected mid-ingest. This module resolves the actual model tokenizer when it
can be loaded (one-time, cached) and falls back to the shared cl100k encoder
otherwise.

Known input limits are declared per provider/model from empirical probes, not
guessed: NVIDIA serves nemotron-3-embed-1b with a 65,536 character cap and
OpenRouter with a 4,096 token cap (reported by the API error itself).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from data_engineering_copilot.infrastructure.token_budget import _ENCODER, TokenEncoder

logger = logging.getLogger(__name__)


# Input limit per model slug: (unit, value). ``unit`` is either "chars" or
# "tokens"; ``value`` is the maximum accepted in that unit, established by
# probing the live endpoints (NVIDIA returns a 65,536-character cap, OpenRouter
# a 4,096-token cap reported in its own API error message).
KNOWN_INPUT_LIMITS: dict[str, tuple[str, int]] = {
    "nvidia/nemotron-3-embed-1b": ("chars", 65536),
    "nvidia/nemotron-3-embed-1b:free": ("tokens", 4096),
    "nvidia/Nemotron-3-Embed-1B-BF16": ("tokens", 8192),
}

# Mapping from embedding model slug to the HuggingFace repo id whose tokenizer
# matches the provider-served model.
_MODEL_TOKENIZER_REPO: dict[str, str] = {
    "nvidia/nemotron-3-embed-1b": "nvidia/Nemotron-3-Embed-1B-BF16",
    "nvidia/nemotron-3-embed-1b:free": "nvidia/Nemotron-3-Embed-1B-BF16",
    "nvidia/Nemotron-3-Embed-1B-BF16": "nvidia/Nemotron-3-Embed-1B-BF16",
}


def declared_input_limit(model_name: str) -> tuple[str, int] | None:
    """Return the empirically-declared ``(unit, limit)`` for *model_name*."""
    return KNOWN_INPUT_LIMITS.get(model_name)


def resolve_token_encoder(model_name: str) -> TokenEncoder:
    """Return a cached encoder matching *model_name*.

    Prefers the real model tokenizer (loaded from HuggingFace once, cached);
    falls back to the shared cl100k encoder when the model is unknown or the
    tokenizer cannot be loaded (offline, gated repo, etc.). Never raises.
    """
    if model_name not in _MODEL_TOKENIZER_REPO:
        return _ENCODER
    return _load_model_tokenizer(model_name)


_tokenizer_lock = threading.Lock()
_tokenizer_cache: dict[str, TokenEncoder] = {}


def _load_model_tokenizer(model_name: str) -> TokenEncoder:
    with _tokenizer_lock:
        cached = _tokenizer_cache.get(model_name)
        if cached is not None:
            return cached
        repo = _MODEL_TOKENIZER_REPO[model_name]
        encoder: TokenEncoder = _ENCODER
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(repo)
            encoder = _TransformersEncoder(tokenizer)
            logger.info("resolved_model_tokenizer model=%s repo=%s", model_name, repo)
        except Exception as exc:  # noqa: BLE001 - offline/gated/any failure is a soft fallback
            logger.warning(
                "model_tokenizer_unavailable model=%s repo=%s err=%s; using cl100k fallback", model_name, repo, exc
            )
        _tokenizer_cache[model_name] = encoder
        return encoder


def reset_tokenizer_cache() -> None:
    """Clear the cached tokenizers (test seam; also drops the cached HF handle)."""
    with _tokenizer_lock:
        _tokenizer_cache.clear()


class _TransformersEncoder:
    """Adapter exposing ``.encode(text)`` over a ``transformers`` tokenizer.

    The raw token sequence (token ids) is used only for counting; ``len`` of
    the ids is the token count, matching the provider's own tokenization.
    """

    __slots__ = ("_tokenizer",)

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text))


def token_counter_for(model_name: str) -> Callable[[str], int]:
    """Return ``text -> int`` token counter for *model_name* (cached encoder)."""
    encoder = resolve_token_encoder(model_name)

    def _count(text: str) -> int:
        return len(encoder.encode(text))

    return _count
