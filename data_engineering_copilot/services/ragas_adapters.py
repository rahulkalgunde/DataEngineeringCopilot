"""RAGAS adapters that route metrics through the repo's adaptive providers.

``RagasEvaluator`` (``ragas_evaluation.py``) builds ragas LLM + embeddings via
these adapters instead of a fixed local Ollama model:

- ``AdaptiveRagasLLM`` wraps the repo's ``AdaptiveLLMRouter`` / ``LLMClient``
  (the purpose-``evaluation`` fallback chain). Every ragas generation is one
  ``generate()`` call, and ``n`` requested completions are produced with ``n``
  independent calls so multi-sample metrics (e.g. answer_relevancy, which
  requests ``n=3``) get real diversity rather than a repeated completion.
- ``AdaptiveRagasEmbeddings`` wraps a priority-ordered list of async
  embedders (NVIDIA → OpenRouter by default). The first provider that returns
  a result becomes the sticky choice for the rest of the run; a failing
  provider fails over to the next. All embedding calls run through a single
  worker-thread event loop so one HTTP client per provider is never shared
  across event loops.

Dimension consistency: ragas computes cosine similarity between vectors from
the same embedder, so the active provider must return a constant dimension.
``AdaptiveRagasEmbeddings`` records the first successful provider's dimension
and rejects a promoted provider that returns a different one. The default
NVIDIA and OpenRouter models are both ``nvidia/nemotron-3-embed-1b`` (2048-dim),
so a mid-run failover stays valid.

Only imported from ``RagasEvaluator._build_runtime`` (after
``_install_vertexai_shim``), so the rest of the system never requires ragas.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

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

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str: ...


class AdaptiveRagasLLM(BaseRagasLLM):
    """Bridge a repo adaptive LLM client into ragas's LLM interface.

    ``client`` is the purpose-``evaluation`` fallback chain (an
    ``AdaptiveLLMRouter`` when ≥2 providers are configured, else a bare
    ``LLMClient``). Each requested generation maps to one ``generate()`` call.
    """

    def __init__(self, client: _AdaptiveLLMProtocol) -> None:
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
    """Bridge repo async embedders into ragas's embeddings interface.

    ``clients`` is a priority-ordered list of ``(provider, embedder)`` pairs.
    The first provider to return a result becomes the sticky choice for the
    rest of the run; a failing provider fails over to the next.

    All embedding calls run on one persistent worker-thread event loop, so the
    embedders' cached ``httpx.AsyncClient`` is never reused across a dead loop
    (which would raise "Event loop is closed" on every call).
    """

    def __init__(self, clients: list[tuple[str, EmbedderProtocol]]) -> None:
        if not clients:
            raise ValueError("AdaptiveRagasEmbeddings requires at least one embedder client")
        super().__init__()
        from ragas.run_config import RunConfig  # lazy optional dep

        self._clients = clients
        self._selected_index: int | None = None
        self._expected_dim: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self.set_run_config(RunConfig())

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None:
                ready = threading.Event()
                holder: dict[str, asyncio.AbstractEventLoop] = {}

                def _run_loop() -> None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    holder["loop"] = loop
                    ready.set()
                    loop.run_forever()

                self._loop_thread = threading.Thread(
                    target=_run_loop,
                    name="ragas-eval-embedding-loop",
                    daemon=True,
                )
                self._loop_thread.start()
                ready.wait()
                self._loop = holder["loop"]
            return self._loop

    def embed_query(self, text: str) -> list[float]:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._embed_with_failover(lambda client: client.embed_query(text)),
            loop,
        )
        return future.result()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._embed_with_failover(lambda client: client.embed_texts(texts)),
            loop,
        )
        return future.result()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.get_running_loop().run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.get_running_loop().run_in_executor(None, self.embed_documents, texts)

    async def close(self) -> None:
        with self._loop_lock:
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
            self._loop_thread = None

    async def _embed_with_failover(self, factory: Any) -> Any:
        start = self._selected_index if self._selected_index is not None else 0
        last_error: Exception | None = None
        for offset in range(len(self._clients)):
            idx = (start + offset) % len(self._clients)
            provider, client = self._clients[idx]
            try:
                vectors = await factory(client)
                dimension = _dimension_of(vectors)
                if self._expected_dim is not None and dimension != self._expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: provider={provider} returned dim={dimension}, "
                        f"expected {self._expected_dim}"
                    )
                self._expected_dim = dimension
                self._selected_index = idx
                return vectors
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "evaluation_embedding_failover provider=%s error=%s",
                    provider,
                    exc,
                )
        raise RuntimeError(f"All evaluation embedding providers failed: {last_error}") from last_error


def _dimension_of(vectors: Any) -> int:
    if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
        return len(vectors[0])
    return len(vectors) if isinstance(vectors, list) else 0


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
