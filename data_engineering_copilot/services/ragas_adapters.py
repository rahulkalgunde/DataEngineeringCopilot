"""RAGAS adapters — thin bridges from repo's unified fallback chains to ragas interfaces.

``RagasEvaluator`` (``ragas_evaluation.py``) builds ragas LLM + embeddings via
these adapters instead of a fixed local Ollama model:

- ``AdaptiveRagasLLM`` wraps the repo's unified LLM fallback chain (a
  ``ProviderFallbackChain`` when ≥2 providers, else a bare ``LLMClient``).
  Every ragas generation is one ``call()`` call, and ``n`` requested completions
  are produced with ``n`` independent calls so multi-sample metrics (e.g.
  answer_relevancy, which requests ``n=3``) get real diversity rather than a
  repeated completion.

- ``AdaptiveRagasEmbeddings`` wraps the repo's unified embedding fallback chain.
  It simply adapts the async ``execute()`` interface to ragas's sync
  ``embed_query``/``embed_documents`` interface. No fallback logic here —
  that's all in the unified chain.

Only imported from ``RagasEvaluator._build_runtime`` (after
``_install_vertexai_shim``), so the rest of the system never requires ragas.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain

logger = logging.getLogger(__name__)


def _ensure_vertexai_shim() -> None:
    """Make ``langchain_community.chat_models.vertexai`` importable.

    ragas 0.3.x imports ``ChatVertexAI`` from this module at package import
    time even for non-Vertex users. langchain-community 0.4.x removed it, so we
    inject a placeholder (never instantiated — used only in isinstance checks).
    """
    try:
        __import__("langchain_community.chat_models.vertexai")
    except ModuleNotFoundError:
        module = types.ModuleType("langchain_community.chat_models.vertexai")
        module.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[reportAttributeAccessIssue]
        sys.modules["langchain_community.chat_models.vertexai"] = module


_ensure_vertexai_shim()

from langchain_core.outputs import Generation, LLMResult  # noqa: E402  # shim must install first
from ragas.embeddings import BaseRagasEmbeddings  # noqa: E402  # shim must install first
from ragas.llms import BaseRagasLLM  # noqa: E402  # shim must install first

from data_engineering_copilot.domain.protocols import EmbedderProtocol  # noqa: E402


class _AdaptiveLLMProtocol(Protocol):
    @property
    def model(self) -> str: ...

    async def call(self, request: str) -> str: ...

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


class AdaptiveRagasLLM(BaseRagasLLM):
    """Bridge a repo unified LLM fallback chain into ragas's LLM interface.

    ``client`` is the purpose-``evaluation`` fallback chain (a
    ``ProviderFallbackChain`` when ≥2 providers are configured, else a bare
    ``LLMClient``). Each requested generation maps to one ``call()`` call.
    """

    def __init__(self, client: _AdaptiveLLMProtocol | ProviderFallbackChain[str, str]) -> None:
        super().__init__()
        self.client = client

    def is_finished(self, response: LLMResult) -> bool:
        for row in response.flatten():
            if not row.generations:
                return False
            for generation in row.generations[0]:
                if not generation.text or not generation.text.strip():
                    return False
        return True

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        text = prompt.to_string()
        if isinstance(self.client, ProviderFallbackChain):
            texts = [await self.client.execute(text) for _ in range(n)]
        else:
            texts = [await self.client.generate(prompt=text, temperature=temperature) for _ in range(n)]
        return LLMResult(generations=[[Generation(text=t) for t in texts]])

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return _run_async(
            self.agenerate_text(prompt=prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks)
        )


class AdaptiveRagasEmbeddings(BaseRagasEmbeddings):
    """Bridge repo unified embedding fallback chain into ragas's embeddings interface.

    ``chain`` is a ``ProviderFallbackChain[list[str], list[list[float]]]`` or a
    bare ``EmbedderProtocol``. No fallback logic here — just adapts the async
    ``execute()`` interface to ragas's sync ``embed_query``/``embed_documents``.
    """

    def __init__(
        self,
        chain: ProviderFallbackChain[list[str], list[list[float]]] | EmbedderProtocol,
    ) -> None:
        super().__init__()
        from ragas.run_config import RunConfig  # lazy optional dep

        self._chain = chain
        self.set_run_config(RunConfig())

    def embed_query(self, text: str) -> list[float]:
        return _run_async(self._execute_embed([text]))[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _run_async(self._execute_embed(texts))

    async def aembed_query(self, text: str) -> list[float]:
        result = await self._execute_embed([text])
        return result[0] if result else []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._execute_embed(texts)

    async def _execute_embed(self, texts: list[str]) -> list[list[float]]:
        if isinstance(self._chain, ProviderFallbackChain):
            return await self._chain.execute(texts)
        else:
            return await self._chain.embed_texts(texts)

    async def close(self) -> None:
        if hasattr(self._chain, "close"):
            await self._chain.close()


def _run_async(coro: Any) -> Any:
    """Run an async function to completion from sync code.

    ragas's answer_relevancy calls the *sync* ``embed_query``/``embed_documents``
    from inside its own event loop, so ``asyncio.run`` cannot be used directly;
    the coroutine runs in a fresh worker-thread event loop instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()
