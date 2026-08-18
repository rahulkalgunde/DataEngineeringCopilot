"""Tests for OpenAICompatibleEmbeddings — async httpx-based OpenAI-compatible embedding provider."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import OpenAICompatibleEmbeddings


@pytest.fixture
def embeddings():
    return OpenAICompatibleEmbeddings(
        api_key="sk-or-v1-test-key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        embedding_dimension=2048,
        batch_size=4,
    )


def test_init(embeddings):
    assert embeddings.model_name == "nvidia/nemotron-3-embed-1b:free"
    assert embeddings._embedding_dimension == 2048
    assert embeddings._batch_size == 4


@pytest.mark.asyncio
async def test_embed_single_text(embeddings):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        result = await embeddings.embed_texts(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 2048


@pytest.mark.asyncio
async def test_embed_multiple_texts(embeddings):
    vectors = [[0.1] * 2048, [0.2] * 2048]
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": vectors[0], "index": 0},
                        {"embedding": vectors[1], "index": 1},
                    ]
                },
            )
        )
        result = await embeddings.embed_texts(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == vectors[0]


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector(embeddings):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        result = await embeddings.embed_query("query")
        assert len(result) == 2048


@pytest.mark.asyncio
async def test_embed_sends_correct_payload(embeddings):
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embeddings.embed_texts(["hello world"])
        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "nvidia/nemotron-3-embed-1b:free"
        assert body["input"] == ["hello world"]


@pytest.mark.asyncio
async def test_embed_texts_sends_passage_input_type(embeddings):
    """Index-time chunks must be embedded in passage mode for dual-mode models."""
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embeddings.embed_texts(["chunk content"])
        body = json.loads(route.calls.last.request.content)
        assert body["input_type"] == "passage"


@pytest.mark.asyncio
async def test_embed_query_sends_query_input_type(embeddings):
    """Live search prompts must be embedded in query mode (not via passage)."""
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embeddings.embed_query("what is spark")
        body = json.loads(route.calls.last.request.content)
        assert body["input_type"] == "query"


@pytest.mark.asyncio
async def test_call_forwards_embedding_request_mode(embeddings):
    """The chain-facing ``call`` forwards an EmbeddingRequest's mode verbatim."""
    from data_engineering_copilot.domain.models import EmbeddingRequest

    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embeddings.call(EmbeddingRequest(input_type="query", texts=["live question"]))
        body = json.loads(route.calls.last.request.content)
        assert body["input_type"] == "query"
        assert body["input"] == ["live question"]


@pytest.mark.asyncio
async def test_call_plain_list_defaults_to_passage(embeddings):
    """Legacy ``call(list[str])`` (no mode) falls back to passage mode."""
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embeddings.call(["legacy chunk"])
        body = json.loads(route.calls.last.request.content)
        assert body["input_type"] == "passage"


@pytest.mark.asyncio
async def test_embed_sends_auth_header(embeddings):
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embeddings.embed_texts(["test"])
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-or-v1-test-key"


@pytest.mark.asyncio
async def test_embed_wrong_dimension_raises(embeddings):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 768, "index": 0}]},
            )
        )
        from data_engineering_copilot.domain.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="dimension 768"):
            await embeddings.embed_texts(["test"])


@pytest.mark.asyncio
async def test_embed_empty_list(embeddings):
    result = await embeddings.embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_http_error(embeddings):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )
        from data_engineering_copilot.domain.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="Failed to get embeddings"):
            await embeddings.embed_texts(["test"])


@pytest.mark.asyncio
async def test_embed_5xx_retries_then_succeeds(tmp_path):
    from tenacity.wait import wait_fixed

    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    emb = OpenAICompatibleEmbeddings(
        api_key="key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        embedding_dimension=2048,
        retry_wait=wait_fixed(0.01),
    )
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            side_effect=[
                httpx.Response(503, json={"error": "Service Unavailable"}),
                httpx.Response(200, json={"data": [{"embedding": [0.1] * 2048, "index": 0}]}),
            ]
        )
        result = await emb.embed_texts(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 2048
        assert len(route.calls) == 2


@pytest.mark.asyncio
async def test_embed_5xx_exhausted_raises_http_status_error(tmp_path):
    from tenacity.wait import wait_fixed

    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    emb = OpenAICompatibleEmbeddings(
        api_key="key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        embedding_dimension=2048,
        retry_wait=wait_fixed(0.01),
    )
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(502, json={"error": "Bad Gateway"})
        )
        # 5xx must surface as the raw HTTPStatusError after retries are
        # exhausted (so the fallback categorizer maps it TEMPORARY_UNAVAILABLE),
        # but the in-provider retry must have happened too.
        try:
            await emb.embed_texts(["hello"])
        except httpx.HTTPStatusError as exc:
            assert exc.response.status_code == 502
        else:
            pytest.fail("expected httpx.HTTPStatusError")
        assert len(route.calls) == 5


def test_reject_over_budget_text():
    from data_engineering_copilot.domain.exceptions import EmbeddingError
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    embs = OpenAICompatibleEmbeddings(api_key="key", max_tokens_per_input=10)
    with pytest.raises(EmbeddingError, match="exceeds budget"):
        embs._reject_over_budget(["hello world " * 100])


def test_reject_blank_input_text():
    """Whitespace-only input must fail loudly, not surface as a cryptic
    provider HTTP 400 (OpenRouter/NVIDIA reject blank strings)."""
    from data_engineering_copilot.domain.exceptions import EmbeddingError
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    embs = OpenAICompatibleEmbeddings(api_key="key", max_tokens_per_input=100)
    with pytest.raises(EmbeddingError, match="blank"):
        embs._reject_over_budget(["valid text", "\n\n", ""])


def test_accepts_within_budget_text():
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    embs = OpenAICompatibleEmbeddings(api_key="key", max_tokens_per_input=100)
    embs._reject_over_budget(["short text"])


def test_reject_over_budget_uses_injected_token_counter():
    from data_engineering_copilot.domain.exceptions import EmbeddingError
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    # A counter that reports every text as 5000 tokens regardless of content.
    embs = OpenAICompatibleEmbeddings(
        api_key="key",
        max_tokens_per_input=100,
        token_counter=lambda text: 5000,
    )
    with pytest.raises(EmbeddingError, match="tokens=5000"):
        embs._reject_over_budget(["tiny"])


def test_accepts_within_budget_using_injected_token_counter():
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    embs = OpenAICompatibleEmbeddings(
        api_key="key",
        max_tokens_per_input=100,
        token_counter=lambda text: 50,
    )
    embs._reject_over_budget(["not short at all " * 50])


def test_declared_input_limit_exceeded_raises():
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    with pytest.raises(ValueError, match="exceeds provider-declared"):
        OpenAICompatibleEmbeddings(
            api_key="key",
            model_name="nvidia/nemotron-3-embed-1b:free",
            max_tokens_per_input=5000,
            declared_input_limit=("tokens", 4096),
        )


def test_declared_input_limit_within_raises_no_error():
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    OpenAICompatibleEmbeddings(
        api_key="key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        max_tokens_per_input=3800,
        declared_input_limit=("tokens", 4096),
    )


@pytest.mark.asyncio
async def test_embed_has_no_truncation_option():
    embs = OpenAICompatibleEmbeddings(api_key="key")
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embs.embed_texts(["hello"])
        body = json.loads(route.calls.last.request.content)
        assert body["provider"] == {}


@pytest.mark.asyncio
async def test_embed_handles_error_in_response(embeddings):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"error": {"message": "Input too long", "code": 422}},
            )
        )
        from data_engineering_copilot.domain.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="Embedding API returned error"):
            await embeddings.embed_texts(["test"])


@pytest.mark.asyncio
async def test_embed_batching():
    embs = OpenAICompatibleEmbeddings(
        api_key="key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        embedding_dimension=2048,
        batch_size=2,
    )
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": [0.1] * 2048, "index": 0},
                        {"embedding": [0.2] * 2048, "index": 1},
                    ]
                },
            )
        )
        result = await embs.embed_texts(["a", "b", "c", "d"])
        assert len(result) == 4
        assert route.call_count == 2


@pytest.mark.asyncio
async def test_embed_fails_fast_when_rate_limit_window_exhausted():
    """When the RPM window is full, the client must NOT call the provider —
    it raises so the fallback chain can fail over to the next provider."""
    from data_engineering_copilot.domain.exceptions import EmbeddingError
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )
    from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

    # A limiter whose RPM window is already exhausted (1 rpm, 1 recorded slot).
    limiter = SlidingWindowRateLimiter(rpm_limit=1, rpd_limit=0)
    await limiter.try_acquire()

    embs = OpenAICompatibleEmbeddings(
        api_key="key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        embedding_dimension=2048,
        rate_limiter=limiter,
    )
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        with pytest.raises(EmbeddingError, match="[Rr]ate limit"):
            await embs.embed_texts(["hello"])
        # No HTTP request was made — the provider was never called.
        assert route.called is False


@pytest.mark.asyncio
async def test_embed_acquires_slot_when_available():
    """When a rate-limit slot is free, the request proceeds normally."""
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )
    from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

    limiter = SlidingWindowRateLimiter(rpm_limit=10, rpd_limit=0)
    embs = OpenAICompatibleEmbeddings(
        api_key="key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        embedding_dimension=2048,
        rate_limiter=limiter,
    )
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        result = await embs.embed_texts(["hello"])
        assert len(result) == 1
        assert route.called is True
