"""Tests for RagasEvaluator — RAGAS evaluation wrapper (ragas_evaluation.py:61).

``datasets`` and ``ragas`` are real dev dependencies now, so these tests use the
real ``datasets.Dataset`` and only mock the evaluation call itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.infrastructure.provider_fallback import (
    FallbackChainConfig,
    ProviderConfig,
    ProviderFallbackChain,
)
from data_engineering_copilot.services.ragas_evaluation import (
    RagasEvaluator,
    _install_vertexai_shim,
)

MOCK_METRICS = {
    "context_recall": 0.8,
    "context_precision": 0.9,
    "faithfulness": 0.95,
    "answer_relevancy": 0.85,
}


def _mock_result(values: dict[str, float]) -> MagicMock:
    mock_result = MagicMock()

    def _getitem(key):
        if key not in values:
            raise KeyError(key)
        return [values[key]]

    mock_result._scores_dict = dict(values)
    mock_result.__getitem__ = MagicMock(side_effect=_getitem)
    return mock_result


class TestVertexaiShim:
    def test_vertexai_shim_makes_module_importable(self):
        _install_vertexai_shim()

        import importlib

        module = importlib.import_module("langchain_community.chat_models.vertexai")
        assert isinstance(module.ChatVertexAI, type)


class TestRagasEvaluator:
    def test_ragas_not_installed_returns_none(self):
        ev = RagasEvaluator()
        with patch.object(ev, "_lazy_init", return_value=False):
            result = ev.evaluate(questions=["q"], answers=["a"], contexts=[["c"]])
        assert result is None

    def test_ragas_installed_returns_result(self):
        ev = RagasEvaluator()
        with (
            patch.object(ev, "_lazy_init", return_value=True),
            patch.object(ev, "_build_runtime", return_value=(MagicMock(), MagicMock())),
            patch.object(ev, "_evaluate", return_value=_mock_result(MOCK_METRICS)),
        ):
            result = ev.evaluate(
                questions=["What is Spark?"],
                answers=["Spark is a engine."],
                contexts=[["Spark documentation"]],
            )

        assert result is not None
        assert result.context_recall == 0.8
        assert result.context_precision == 0.9
        assert result.faithfulness == 0.95
        assert result.answer_relevancy == 0.85
        expected_overall = round(0.8 * 0.3 + 0.95 * 0.4 + 0.85 * 0.3, 4)
        assert result.overall == expected_overall

    def test_with_ground_truth(self):
        ev = RagasEvaluator()
        with (
            patch.object(ev, "_lazy_init", return_value=True),
            patch.object(ev, "_build_runtime", return_value=(MagicMock(), MagicMock())),
            patch.object(ev, "_evaluate", return_value=_mock_result(MOCK_METRICS)),
        ):
            result = ev.evaluate(
                questions=["q"],
                answers=["a"],
                contexts=[["c"]],
                ground_truth=["gt"],
            )
        assert result is not None

    def test_missing_keys_default_to_zero(self):
        ev = RagasEvaluator()
        with (
            patch.object(ev, "_lazy_init", return_value=True),
            patch.object(ev, "_build_runtime", return_value=(MagicMock(), MagicMock())),
            patch.object(ev, "_evaluate", return_value=_mock_result({})),
        ):
            result = ev.evaluate(
                questions=["q"],
                answers=["a"],
                contexts=[["c"]],
            )

        assert result is not None
        assert result.context_recall == 0.0
        assert result.faithfulness == 0.0
        assert result.answer_relevancy == 0.0

    def test_lazy_init_caches_success(self):
        ev = RagasEvaluator()
        with patch("builtins.__import__") as mock_import:
            mock_ragas = MagicMock()
            mock_import.return_value = mock_ragas
            result = ev._lazy_init()
        assert result is True
        assert ev._evaluate is not None

    def test_lazy_init_failure_returns_false(self):
        ev = RagasEvaluator()
        with patch("builtins.__import__", side_effect=ImportError):
            result = ev._lazy_init()
        assert result is False

    def test_real_lazy_init_with_shim(self):
        ev = RagasEvaluator()
        assert ev._lazy_init() is True
        assert ev._metrics is not None
        assert len(ev._metrics) == 4

    def test_wires_llm_and_embeddings_into_metrics(self):
        ev = RagasEvaluator()
        captured = {}

        class FakeMetric:
            def __init__(self):
                self.llm = None
                self.embeddings = None

        metrics = [FakeMetric() for _ in range(4)]

        def fake_evaluate(dataset, metrics=None, llm=None, **kwargs):
            captured["llm"] = llm
            captured["metrics"] = metrics
            return _mock_result(MOCK_METRICS)

        fake_llm_wrapper = MagicMock()
        fake_embed_wrapper = MagicMock()
        with (
            patch.object(ev, "_lazy_init", return_value=True),
            patch.object(ev, "_evaluate", side_effect=fake_evaluate),
            patch.object(ev, "_build_runtime", return_value=(fake_llm_wrapper, fake_embed_wrapper)),
        ):
            ev._metrics = metrics
            result = ev.evaluate(
                questions=["q"],
                answers=["a"],
                contexts=[["c"]],
            )

        assert captured["llm"] is fake_llm_wrapper
        assert captured["metrics"] is metrics
        for metric in metrics:
            assert metric.llm is fake_llm_wrapper
            assert metric.embeddings is fake_embed_wrapper
        assert result is not None
        assert result.context_recall == 0.8

    def test_build_runtime_defaults_use_adaptive_routing(self):
        from data_engineering_copilot.infrastructure.llm_client import LLMClient
        from data_engineering_copilot.services.ragas_adapters import (
            AdaptiveRagasEmbeddings,
            AdaptiveRagasLLM,
        )
        from tests.conftest import make_settings

        app_settings = make_settings()
        llm_wrapper, embeddings_wrapper = RagasEvaluator._build_runtime(app_settings=app_settings)
        assert isinstance(llm_wrapper, AdaptiveRagasLLM)
        assert isinstance(embeddings_wrapper, AdaptiveRagasEmbeddings)
        # LLM: purpose='evaluation' fallback chain. No provider keys in the
        # hermetic settings -> degrades to a bare local Ollama LLMClient.
        assert isinstance(llm_wrapper.client, LLMClient)
        assert llm_wrapper.client.model == (app_settings.ollama_model or "llama3.2:3b")
        # Embeddings: external nvidia/openrouter preferred, skipped without
        # keys -> local Ollama degraded fallback (never a paid provider).
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain

        if isinstance(embeddings_wrapper._chain, ProviderFallbackChain):
            providers = [p.name for p in embeddings_wrapper._chain._config.providers]
            if embeddings_wrapper._chain._config.degraded_fallback:
                providers.append(embeddings_wrapper._chain._config.degraded_fallback.name)
        else:
            providers = ["ollama"]  # bare EmbedderProtocol
        assert providers == ["ollama"]

    def test_build_runtime_prefers_external_embedding_providers(self):
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings
        from tests.conftest import make_settings

        app_settings = make_settings(
            nvidia_api_key="placeholder",
            openrouter_api_key="placeholder",
            embedding_provider="ollama",
        )
        llm_wrapper, embeddings_wrapper = RagasEvaluator._build_runtime(app_settings=app_settings)
        assert isinstance(embeddings_wrapper, AdaptiveRagasEmbeddings)
        if isinstance(embeddings_wrapper._chain, ProviderFallbackChain):
            providers = [p.name for p in embeddings_wrapper._chain._config.providers]
            if embeddings_wrapper._chain._config.degraded_fallback:
                providers.append(embeddings_wrapper._chain._config.degraded_fallback.name)
        else:
            providers = ["ollama"]
        # NVIDIA + OpenRouter as main chain, Ollama as degraded fallback
        assert providers == ["nvidia", "openrouter", "ollama"]
        assert llm_wrapper is not None

    def test_build_runtime_adaptive_judge_has_no_pinned_primary(self):
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasLLM
        from tests.conftest import make_settings

        app_settings = make_settings(
            cloudflare_api_key="placeholder",
            groq_api_key="placeholder",
            openrouter_api_key="placeholder",
            gemini_api_key="placeholder",
            cerebras_api_key="placeholder",
            nvidia_api_key="placeholder",
        )
        llm_wrapper, _ = RagasEvaluator._build_runtime(app_settings=app_settings)
        assert isinstance(llm_wrapper, AdaptiveRagasLLM)
        assert isinstance(llm_wrapper.client, ProviderFallbackChain)
        # No forced primary: the chain follows llm_fallback_order verbatim, so
        # each call can pick the first currently-available provider.
        providers = [p.name for p in llm_wrapper.client._config.providers]
        if llm_wrapper.client._config.degraded_fallback:
            providers.append(llm_wrapper.client._config.degraded_fallback.name)
        assert providers == [p.lower() for p in app_settings.llm_fallback_order]

    def test_build_runtime_pinned_evaluation_provider_is_primary(self):
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasLLM
        from tests.conftest import make_settings

        app_settings = make_settings(
            evaluation_llm_provider="groq",
            evaluation_llm_model="llama-3.1-8b-instant",
            groq_api_key="placeholder",
            _test_allow_non_ollama=True,
        )
        llm_wrapper, _ = RagasEvaluator._build_runtime(app_settings=app_settings)
        assert isinstance(llm_wrapper, AdaptiveRagasLLM)
        assert isinstance(llm_wrapper.client, ProviderFallbackChain)
        primary_provider = llm_wrapper.client._config.providers[0].name
        primary_client = llm_wrapper.client._config.providers[0].client
        assert primary_provider == "groq"
        assert primary_client.model == "llama-3.1-8b-instant"
        # Remaining providers after the primary (key-less ones are skipped).
        rest = [p.name for p in llm_wrapper.client._config.providers[1:]]
        if llm_wrapper.client._config.degraded_fallback:
            rest.append(llm_wrapper.client._config.degraded_fallback.name)
        assert rest == ["ollama"]

    def test_build_runtime_wraps_explicit_langchain_objects(self):
        _install_vertexai_shim()

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from pydantic import SecretStr
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms.base import LangchainLLMWrapper

        chat = ChatOpenAI(
            model="fake-model",
            api_key=SecretStr("k"),
            base_url="http://localhost:1/v1",
        )
        emb = OpenAIEmbeddings(
            model="fake-embed",
            api_key=SecretStr("k"),
            base_url="http://localhost:1/v1",
            check_embedding_ctx_length=False,
        )
        llm_wrapper, embeddings_wrapper = RagasEvaluator._build_runtime(llm=chat, embeddings=emb)
        assert isinstance(llm_wrapper, LangchainLLMWrapper)
        assert isinstance(embeddings_wrapper, LangchainEmbeddingsWrapper)
        assert llm_wrapper.langchain_llm is chat
        assert embeddings_wrapper.embeddings is emb


class _StubEmbedder:
    """Async embedder double honoring EmbedderProtocol and ProviderClient protocol.

    ``tag`` is a distinct float per provider, repeated to ``dim`` length, so
    callers can assert which provider produced a vector.
    """

    def __init__(self, dim: int, name: str, tag: float, fail_attempts: int = 0, fail_on_call: int = 0):
        self.dim = dim
        self.name = name
        self.tag = tag
        self.fail_attempts = fail_attempts
        self.fail_on_call = fail_on_call
        self.calls = 0

    def _maybe_raise(self):
        if self.fail_attempts > 0:
            self.fail_attempts -= 1
            raise RuntimeError(f"{self.name} simulated failure")
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise RuntimeError(f"{self.name} simulated failure")

    # EmbedderProtocol methods
    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        self._maybe_raise()
        return [self.tag] * self.dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self._maybe_raise()
        return [[self.tag] * self.dim for _ in texts]

    async def close(self) -> None:
        pass

    # ProviderClient protocol method
    async def call(self, request: list[str]) -> list[list[float]]:
        return await self.embed_texts(request)

    @property
    def model(self) -> str:
        return self.name

    @property
    def last_usage(self):
        from data_engineering_copilot.domain.models import LLMUsage

        return LLMUsage()


class _FakeLLMClient:
    """Fake LLM client for testing."""

    model = "fake-model"

    def __init__(self):
        self.calls = []

    async def call(self, request: str) -> str:
        self.calls.append(request)
        return f"out-{len(self.calls)}"

    async def generate(self, prompt, temperature=None, max_tokens=None):
        return await self.call(prompt)

    @property
    def last_usage(self):
        from data_engineering_copilot.domain.models import LLMUsage

        return LLMUsage()


class TestAdaptiveRagasLLM:
    def test_agenerate_text_makes_one_call_per_completion(self):
        import asyncio

        from langchain_core.prompt_values import StringPromptValue

        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasLLM

        client = _FakeLLMClient()
        llm = AdaptiveRagasLLM(client)
        result = asyncio.run(llm.agenerate_text(StringPromptValue(text="hi"), n=3, temperature=0.3))
        assert len(result.generations) == 1
        assert [g.text for g in result.generations[0]] == ["out-1", "out-2", "out-3"]
        assert len(client.calls) == 3
        assert all(prompt == "hi" for prompt in client.calls)

    def test_is_finished_rejects_blank_generation(self):
        from langchain_core.outputs import Generation, LLMResult

        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasLLM

        client = _FakeLLMClient()
        llm = AdaptiveRagasLLM(client)
        assert llm.is_finished(LLMResult(generations=[[Generation(text="done")]])) is True
        assert llm.is_finished(LLMResult(generations=[[Generation(text="   ")]])) is False
        assert llm.is_finished(LLMResult(generations=[[Generation(text="")]])) is False


class TestAdaptiveRagasEmbeddings:
    def test_requires_at_least_one_client(self):
        from unittest.mock import MagicMock

        from data_engineering_copilot.infrastructure.provider_fallback import (
            FallbackChainConfig,
            ProviderFallbackChain,
        )
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings

        mock_health = MagicMock()
        mock_health.get_provider_health.return_value = None
        mock_health.get_model_health.return_value = None
        with pytest.raises(ValueError):
            AdaptiveRagasEmbeddings(ProviderFallbackChain(FallbackChainConfig(providers=[]), mock_health))

    def test_prefers_first_provider_sticky(self):
        from unittest.mock import MagicMock

        from data_engineering_copilot.infrastructure.provider_fallback import (
            FallbackChainConfig,
            ProviderConfig,
            ProviderFallbackChain,
        )
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings

        nvidia = _StubEmbedder(dim=2048, name="nvidia", tag=1.0)
        openrouter = _StubEmbedder(dim=2048, name="openrouter", tag=2.0)

        config = FallbackChainConfig(
            providers=[
                ProviderConfig("nvidia", nvidia),
                ProviderConfig("openrouter", openrouter),
            ]
        )
        mock_health = MagicMock()
        mock_health.get_provider_health.return_value = None
        mock_health.get_model_health.return_value = None
        chain = ProviderFallbackChain(config, mock_health)
        embeddings = AdaptiveRagasEmbeddings(chain)

        query_vec = embeddings.embed_query("q")
        assert query_vec == [1.0] * 2048

        doc_vecs = embeddings.embed_documents(["a", "b"])
        assert doc_vecs[0] == [1.0] * 2048
        assert nvidia.calls == 2
        assert openrouter.calls == 0

    def test_fails_over_to_next_provider(self):
        from unittest.mock import MagicMock

        from data_engineering_copilot.infrastructure.provider_fallback import (
            FallbackChainConfig,
            ProviderConfig,
            ProviderFallbackChain,
        )
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings

        # nvidia fails on both calls; openrouter must serve both (no stickiness —
        # each call re-evaluates availability through the chain).
        nvidia = _StubEmbedder(dim=2048, name="nvidia", tag=1.0, fail_attempts=2)
        openrouter = _StubEmbedder(dim=2048, name="openrouter", tag=2.0)

        config = FallbackChainConfig(
            providers=[
                ProviderConfig("nvidia", nvidia),
                ProviderConfig("openrouter", openrouter),
            ]
        )
        mock_health = MagicMock()
        mock_health.get_provider_health.return_value = None
        mock_health.get_model_health.return_value = None
        chain = ProviderFallbackChain(config, mock_health)
        embeddings = AdaptiveRagasEmbeddings(chain)

        query_vec = embeddings.embed_query("q")
        assert query_vec == [2.0] * 2048

        doc_vecs = embeddings.embed_documents(["a"])
        assert doc_vecs[0] == [2.0] * 2048

    def test_recovers_primary_when_it_comes_back(self):
        from unittest.mock import MagicMock

        from data_engineering_copilot.infrastructure.provider_fallback import (
            FallbackChainConfig,
            ProviderConfig,
            ProviderFallbackChain,
        )
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings

        # nvidia fails once, then recovers — the next call goes back to it
        # because the chain re-evaluates availability per call.
        nvidia = _StubEmbedder(dim=2048, name="nvidia", tag=1.0, fail_attempts=1)
        openrouter = _StubEmbedder(dim=2048, name="openrouter", tag=2.0)

        config = FallbackChainConfig(
            providers=[
                ProviderConfig("nvidia", nvidia),
                ProviderConfig("openrouter", openrouter),
            ]
        )
        mock_health = MagicMock()
        mock_health.get_provider_health.return_value = None
        mock_health.get_model_health.return_value = None
        chain = ProviderFallbackChain(config, mock_health)
        embeddings = AdaptiveRagasEmbeddings(chain)

        query_vec = embeddings.embed_query("q")
        assert query_vec == [2.0] * 2048

        doc_vecs = embeddings.embed_documents(["a"])
        assert doc_vecs[0] == [1.0] * 2048
        assert nvidia.calls == 2

    def test_failover_uses_backup_provider_dimension(self):
        from unittest.mock import MagicMock

        from data_engineering_copilot.infrastructure.provider_fallback import (
            FallbackChainConfig,
            ProviderConfig,
            ProviderFallbackChain,
        )
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings

        # Dimension consistency is enforced at chain-build time (factory pins
        # providers to the same model/dim). At call time the promoted provider
        # simply serves its own vectors.
        nvidia = _StubEmbedder(dim=768, name="nvidia", tag=1.0, fail_on_call=2)
        openrouter = _StubEmbedder(dim=2048, name="openrouter", tag=2.0)

        config = FallbackChainConfig(
            providers=[
                ProviderConfig("nvidia", nvidia),
                ProviderConfig("openrouter", openrouter),
            ]
        )
        mock_health = MagicMock()
        mock_health.get_provider_health.return_value = None
        mock_health.get_model_health.return_value = None
        chain = ProviderFallbackChain(config, mock_health)
        embeddings = AdaptiveRagasEmbeddings(chain)

        query_vec = embeddings.embed_query("q")
        assert query_vec == [1.0] * 768

        doc_vecs = embeddings.embed_documents(["a"])
        assert doc_vecs[0] == [2.0] * 2048

    def test_raises_when_all_providers_fail(self):
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings

        nvidia = _StubEmbedder(dim=2048, name="nvidia", tag=1.0, fail_attempts=1)
        openrouter = _StubEmbedder(dim=2048, name="openrouter", tag=2.0, fail_attempts=1)

        config = FallbackChainConfig(
            providers=[
                ProviderConfig("nvidia", nvidia),
                ProviderConfig("openrouter", openrouter),
            ]
        )
        from unittest.mock import MagicMock

        mock_health = MagicMock()
        mock_health.get_provider_health.return_value = None
        mock_health.get_model_health.return_value = None
        chain = ProviderFallbackChain(config, mock_health)
        embeddings = AdaptiveRagasEmbeddings(chain)

        with pytest.raises(RuntimeError, match="All providers in fallback chain failed"):
            embeddings.embed_query("q")

    def test_async_methods_route_through_same_worker_loop(self):
        import asyncio

        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings

        nvidia = _StubEmbedder(dim=2048, name="nvidia", tag=1.0)

        config = FallbackChainConfig(providers=[ProviderConfig("nvidia", nvidia)])
        from unittest.mock import MagicMock

        mock_health = MagicMock()
        mock_health.get_provider_health.return_value = None
        mock_health.get_model_health.return_value = None
        chain = ProviderFallbackChain(config, mock_health)
        embeddings = AdaptiveRagasEmbeddings(chain)

        async def _run():
            query_vec = await embeddings.aembed_query("q")
            doc_vecs = await embeddings.aembed_documents(["a", "b"])
            return query_vec, doc_vecs

        query_vec, doc_vecs = asyncio.run(_run())
        assert query_vec == [1.0] * 2048
        assert doc_vecs[0] == [1.0] * 2048

    def test_sync_embed_works_across_fresh_event_loops(self):
        """Regression: RAGAS calls the sync ``embed_query``/``embed_documents``
        from inside its own per-call event loops. Each sync call bridges via a
        fresh ``asyncio.run`` loop, so the underlying httpx provider must not
        reuse a client bound to a closed loop. Uses the production
        ``OpenAICompatibleEmbeddings`` client (the adaptive multi-provider
        fallback chain used for NVIDIA/OpenRouter)."""
        import httpx
        import respx

        from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
            OpenAICompatibleEmbeddings,
        )
        from data_engineering_copilot.services.ragas_adapters import AdaptiveRagasEmbeddings

        nvidia = OpenAICompatibleEmbeddings(
            api_key="test-key",
            model_name="nvidia/nemotron-3-embed-1b",
            base_url="http://localhost:2",
        )
        mock_health = MagicMock()
        mock_health.get_provider_health.return_value = None
        mock_health.get_model_health.return_value = None
        chain = ProviderFallbackChain(
            FallbackChainConfig(providers=[ProviderConfig("nvidia", nvidia)]),
            mock_health,
        )
        embeddings = AdaptiveRagasEmbeddings(chain)

        with respx.mock:
            respx.post("http://localhost:2/embeddings").mock(
                return_value=httpx.Response(
                    200,
                    json={"data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 2048}]},
                )
            )
            # Each call bridges via its own event loop — the httpx client
            # created in the first loop dies with it.
            first = embeddings.embed_query("q")
            first_client = nvidia._client
            second = embeddings.embed_query("q2")
            second_client = nvidia._client

        assert first == [0.1] * 2048
        assert second == [0.1] * 2048
        # The loop-bound guard must have recreated the client for the fresh
        # loop (on the unfixed code the same client object is reused).
        assert first_client is not None
        assert first_client is not second_client
