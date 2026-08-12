"""EmbedderProtocol adapter over the unified embedding fallback chain.

``build_embedding_fallback_chain`` returns a ``ProviderFallbackChain`` when ≥2
providers are configured (e.g. ``embedding_fallback_order=["nvidia",
"openrouter"]``), else a bare ``EmbedderProtocol``. RAG/ingestion callers use
the plain ``build_embedder`` single-provider path; offline batch pipelines
(Spark index build) want the same adaptive NVIDIA→OpenRouter→Ollama fallback
used everywhere else. This adapter exposes that chain through the standard
``EmbedderProtocol`` interface.
"""

from __future__ import annotations

from typing import Any

from data_engineering_copilot.domain.models import EmbeddingRequest


class FallbackEmbedder:
    """Adapt a ``ProviderFallbackChain[list[str], list[list[float]]]`` or a
    bare ``EmbedderProtocol`` into a uniform ``EmbedderProtocol``.

    Fallback logic lives entirely in the chain; this class only bridges the
    ``execute()`` interface to ``embed_texts``/``embed_query``/``close``.
    Requests carry an ``EmbeddingRequest`` role (passage/query) so dual-mode
    embedding models receive the correct ``input_type`` even through the chain.
    """

    def __init__(
        self,
        chain: Any,
    ) -> None:
        self._chain: Any = chain

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self._chain, "execute"):
            return await self._chain.execute(EmbeddingRequest(input_type="passage", texts=list(texts)))
        return await self._chain.embed_texts(texts)

    async def embed_query(self, text: str) -> list[float]:
        if hasattr(self._chain, "execute"):
            results = await self._chain.execute(EmbeddingRequest(input_type="query", texts=[text]))
        else:
            results = await self._chain.embed_texts([text])
        if not results or results[0] is None:
            raise ValueError("embed_query returned no embedding")
        return results[0]

    async def close(self) -> None:
        if hasattr(self._chain, "close"):
            await self._chain.close()

    @property
    def inner(self) -> Any:
        return self._chain
