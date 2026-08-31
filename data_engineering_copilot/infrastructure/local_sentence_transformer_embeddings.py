"""Local HuggingFace embedding provider using sentence-transformers.

Runs ``SentenceTransformer`` inference on the local CPU, mirroring how the
cross-encoder reranker runs locally. This eliminates dependence on flaky /
free-tier embedding APIs (NVIDIA/OpenRouter 503s) while producing *identical*
embeddings to the hosted ``nvidia/nemotron-3-embed-1b`` model (verified: cosine
~1.0 against the NVIDIA API in the same mode).

The model is downloaded once from HuggingFace (~1.14 GB for
``nvidia/Nemotron-3-Embed-1B-BF16``) and cached on disk like the reranker.
Batch inference is CPU-bound and blocking, so it is offloaded to a thread via
``asyncio.to_thread``.

Dual-mode (``input_type``): the hosted model distinguishes ``query`` /
``passage`` via the ``query:`` / ``passage:`` prefixes (model-card guidance for
the OpenAI-compatible endpoint). The local model applies the same prefixes, so
``embed_query`` and ``embed_texts`` produce the same subspaces as the API.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from data_engineering_copilot.domain.exceptions import EmbeddingError
from data_engineering_copilot.domain.models import EmbeddingRequest, LLMUsage

logger = logging.getLogger(__name__)

# Module-level singleton so a long-running process keeps the model warm.
_model_lock = threading.Lock()
_model_cache: dict[str, Any] = {}


def _load_model(model_name: str):
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        _model_cache[model_name] = model
        logger.info("Loaded local embedding model %s", model_name)
        return model


def clear_model_cache() -> None:
    """Drop cached models (test seam)."""
    with _model_lock:
        _model_cache.clear()


class LocalSentenceTransformerEmbeddings:
    """Local ``SentenceTransformer`` embedding provider (2048-dim by default).

    Satisfies ``EmbedderProtocol`` and the ``ProviderClient`` shape used by the
    fallback chain (``call``, ``model``, ``last_usage``, ``close``).
    """

    def __init__(
        self,
        model_name: str = "nvidia/Nemotron-3-Embed-1B-BF16",
        embedding_dimension: int = 2048,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        # No HTTP endpoint (runs locally). Present for shape-compat with the
        # probe tooling, which skips local providers.
        self.base_url = ""
        self._embedding_dimension = embedding_dimension
        self._batch_size = batch_size

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def last_usage(self) -> LLMUsage:
        return LLMUsage()

    def _encode_sync(self, texts: list[str], prefix: str | None) -> list[list[float]]:
        model = _load_model(self.model_name)
        # Exact model-card prefixes (with colon): the hosted model's
        # ``input_type=query|passage`` maps to ``query:`` / ``passage:`` on the
        # OpenAI-compatible endpoint. Verified: with these prefixes local
        # vectors are cos~1.0 to the NVIDIA API in the same mode.
        prepared = [f"{prefix}: {t}" if prefix else t for t in texts]
        vectors = model.encode(prepared, batch_size=self._batch_size, show_progress_bar=False, convert_to_numpy=True)
        return [[float(x) for x in v] for v in vectors]

    async def _encode(self, texts: list[str], prefix: str | None) -> list[list[float]]:
        try:
            return await asyncio.to_thread(self._encode_sync, texts, prefix)
        except Exception as exc:
            raise EmbeddingError(f"Local embedding failed ({self.model_name}): {exc}") from exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._encode(texts, prefix="passage")

    async def embed_query(self, text: str) -> list[float]:
        results = await self._encode([text], prefix="query")
        if not results or results[0] is None:
            raise EmbeddingError(f"Local embedding returned empty result for query: {text[:80]!r}")
        return results[0]

    async def call(self, request: list[str] | EmbeddingRequest) -> list[list[float]]:
        if isinstance(request, EmbeddingRequest):
            # Local model maps input_type -> query/passage prefix; unknown/None
            # values embed without a prefix (plain mode).
            prefix = {"query": "query", "passage": "passage"}.get(request.input_type) if request.input_type else None
            texts = request.texts
        else:
            prefix = "passage"
            texts = list(request)
        if not texts:
            return []
        return await self._encode(texts, prefix=prefix)

    async def close(self) -> None:
        # Model stays cached for the process lifetime (like the reranker).
        return None


_transformer_cache: dict[str, tuple[Any, Any]] = {}


def _encode_batch_pure_transformers_cached(texts: list[str]) -> list[list[float]]:
    """Cached version of :func:`_encode_batch_pure_transformers`.

    Reuses the model and tokenizer across calls within a persistent
    ``ProcessPoolExecutor`` worker, avoiding the ~30s model-reloading
    overhead that made batch-by-batch fallback unusably slow.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "nvidia/Nemotron-3-Embed-1B-BF16"
    if model_name not in _transformer_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        _transformer_cache[model_name] = (model, tokenizer)

    model, tokenizer = _transformer_cache[model_name]

    prepared = [f"passage: {t}" for t in texts]
    inputs = tokenizer(prepared, padding=True, truncation=True, return_tensors="pt", max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.tolist()


def _encode_batch_pure_transformers(texts: list[str]) -> list[list[float]]:
    """Embed texts using pure transformers (mean pooling + L2 norm).

    Fallback for when ``sentence-transformers`` crashes (segfault in native
    code). Verified to produce cos=0.999996-identical vectors to
    ``sentence-transformers`` for the Nemotron model, with zero crash risk.
    Runs in a subprocess worker — safe from segfaults in the main process.

    .. deprecated::
        Use :func:`_encode_batch_pure_transformers_cached` instead to avoid
        reloading the model on every call.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "nvidia/Nemotron-3-Embed-1B-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    prepared = [f"passage: {t}" for t in texts]
    inputs = tokenizer(prepared, padding=True, truncation=True, return_tensors="pt", max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.tolist()
