"""Tests for DynamicBatchSizer."""

from __future__ import annotations

import pytest

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.infrastructure.dynamic_batch_sizer import DynamicBatchSizer


@pytest.fixture
def sizer() -> DynamicBatchSizer:
    return DynamicBatchSizer(AppSettings())


def test_short_texts_get_max_batch(sizer: DynamicBatchSizer) -> None:
    texts = ["short text " + str(i) for i in range(100)]
    batch = sizer.compute_batch_size("nvidia", texts)
    assert batch == 1024  # max provider limit for nvidia


def test_long_texts_reduce_batch(sizer: DynamicBatchSizer) -> None:
    texts = ["x" * 4000 for _ in range(100)]  # ~1000 tokens each
    batch = sizer.compute_batch_size("nvidia", texts)
    assert batch < 1024  # reduced by context window


def test_openrouter_lower_limit(sizer: DynamicBatchSizer) -> None:
    texts = ["short text " + str(i) for i in range(100)]
    batch = sizer.compute_batch_size("openrouter", texts)
    assert batch == 256  # max provider limit for openrouter


def test_batch_always_multiple_of_32(sizer: DynamicBatchSizer) -> None:
    for length in [50, 100, 200, 500, 1000]:
        texts = ["x" * length for _ in range(100)]
        batch = sizer.compute_batch_size("nvidia", texts)
        assert batch % 32 == 0
        assert batch >= 32


def test_empty_texts_returns_default(sizer: DynamicBatchSizer) -> None:
    batch = sizer.compute_batch_size("nvidia", [])
    assert batch == 64  # default embedding_batch_size


def test_unknown_provider_uses_default(sizer: DynamicBatchSizer) -> None:
    texts = ["short text " + str(i) for i in range(100)]
    batch = sizer.compute_batch_size("unknown_provider", texts)
    assert batch >= 32


def test_model_specific_context_window(sizer: DynamicBatchSizer) -> None:
    texts = ["short text " + str(i) for i in range(100)]
    # OpenRouter free tier has 16K context vs NVIDIA 131K
    batch_free = sizer.compute_batch_size("openrouter", texts, "nvidia/nemotron-3-embed-1b:free")
    batch_nvidia = sizer.compute_batch_size("nvidia", texts, "nvidia/nemotron-3-embed-1b")
    assert batch_nvidia > batch_free
