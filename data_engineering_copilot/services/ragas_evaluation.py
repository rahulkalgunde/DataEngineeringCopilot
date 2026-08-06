"""RAGAS evaluation integration — CI-gated RAG quality metrics.

Lazily imports ``ragas`` and ``datasets`` so the system works without them.
Metrics: context_recall, context_precision, faithfulness, answer_relevancy.

Compatibility notes:
- ragas 0.3.x unconditionally imports ``langchain_community.chat_models.vertexai``
  at ``ragas.llms.base`` import time, purely for ``isinstance`` checks.
  langchain-community 0.4.x removed that module (VertexAI moved to a standalone
  integration package), which breaks the import. ``_install_vertexai_shim``
  injects a placeholder module so ragas can be imported unchanged.
- Metrics route through the repo's adaptive provider routing instead of a fixed
  local model (see ``ragas_adapters.py``): the evaluation LLM is not pinned to a
  specific model — unless ``evaluation_llm_provider`` is explicitly set, every
  call routes through ``llm_fallback_order`` and picks the first currently
  available provider (local Ollama as degraded last resort). Embeddings use
  NVIDIA → OpenRouter with local Ollama only as a key-less degraded fallback.
  Both external embedding models default to ``nvidia/nemotron-3-embed-1b``
  (2048-dim), so failover keeps cosine similarity valid.
- Importing ragas is only safe *through this wrapper* (or ``ragas_adapters.py``),
  which installs the vertexai shim first. A bare ``import ragas`` in a fresh
  process raises ``ModuleNotFoundError`` for the missing vertexai module — do
  not import ragas directly outside these modules.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ragas starts a background analytics flush thread on import; that thread can
# destabilize later native-extension imports (torch) in threaded processes.
# Opt out of analytics before ragas is ever imported.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def _install_vertexai_shim() -> None:
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


@dataclass
class RagasEvalResult:
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevancy: float
    overall: float


class RagasEvaluator:
    """Wraps the RAGAS framework for production evaluation.

    Requires ``ragas`` and ``datasets`` packages. Lazily imported so the
    system works without them.

    ``run_timeout`` / ``max_workers`` tune the per-metric job timeout and
    concurrency; the ragas defaults (180s, 16 workers) are wrong for the local
    Ollama degraded fallback, which is slow and single-model (parallel jobs
    compete for the same GPU, and calls easily exceed 180s).
    """

    run_timeout: int = 1800
    max_workers: int = 2

    def __init__(self) -> None:
        self._evaluate: Callable[..., Any] | None = None
        self._metrics: list[Any] | None = None

    def _lazy_init(self) -> bool:
        if self._evaluate is not None:
            return True
        try:
            _install_vertexai_shim()
            from ragas import evaluate  # type: ignore[import-not-found]  # lazy optional dep
            from ragas.metrics import (  # type: ignore[import-not-found]  # lazy optional dep
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            self._evaluate = evaluate
            self._metrics = [
                context_recall,
                context_precision,
                faithfulness,
                answer_relevancy,
            ]
            return True
        except ImportError:
            logger.debug("ragas package not installed — evaluation unavailable")
            return False

    @staticmethod
    def _build_runtime(
        llm: Any = None,
        embeddings: Any = None,
        app_settings: Any = None,
    ) -> tuple[Any, Any]:
        """Build ragas LLM + embeddings wrappers (adaptive providers by default).

        Accepts raw langchain ``BaseLanguageModel`` / ``BaseEmbeddings`` objects
        (wrapped for ragas); when omitted, defaults are derived from
        ``AppSettings`` via the factory:

        - LLM: when ``evaluation_llm_provider`` is explicitly set it is the
          pinned primary of the purpose-``evaluation`` adaptive fallback chain
          (``build_llm_fallback_chain``). When it is empty (the default), no
          provider is forced: the chain routes every call
          through ``llm_fallback_order`` and picks the first currently
          available provider, with local Ollama as the degraded last resort.
        - Embeddings: ``build_embedding_fallback_chain`` — NVIDIA then OpenRouter
          (both 2048-dim ``nemotron-3-embed-1b``), local Ollama only when no
          external provider has an API key.

        The RAGAS LLM/embeddings adapters live in ``ragas_adapters.py`` and are
        only imported here (after ``_install_vertexai_shim``), keeping the rest
        of the system free of the ragas dependency.
        """
        _install_vertexai_shim()

        from data_engineering_copilot.config.settings import settings as live_settings

        app_settings = app_settings or live_settings

        from ragas.embeddings import BaseRagasEmbeddings, LangchainEmbeddingsWrapper

        from data_engineering_copilot.factory import (
            _build_provider_health_registry,
            _build_provider_rate_limiters,
            build_embedding_fallback_chain,
            build_llm_fallback_chain,
        )
        from data_engineering_copilot.services.ragas_adapters import (
            AdaptiveRagasEmbeddings,
            AdaptiveRagasLLM,
        )

        provider_rate_limiters = _build_provider_rate_limiters(app_settings)
        health_registry = _build_provider_health_registry(app_settings)

        if llm is None:
            # Build unified LLM fallback chain for evaluation
            client = build_llm_fallback_chain(
                purpose="evaluation",
                app_settings=app_settings,
                provider_rate_limiters=provider_rate_limiters,
                health_registry=health_registry,
                purpose_provider=app_settings.evaluation_llm_provider,
                purpose_model=app_settings.evaluation_llm_model,
            )
            if client is None:
                raise ValueError(
                    "No LLM client could be built for RAGAS evaluation. "
                    "Check evaluation_llm_provider / llm_provider configuration."
                )
            llm = AdaptiveRagasLLM(client)

        from ragas.llms import BaseRagasLLM, LangchainLLMWrapper

        if not isinstance(llm, BaseRagasLLM):
            llm = LangchainLLMWrapper(llm)

        if embeddings is None:
            # Build unified embedding fallback chain for evaluation
            embedding_chain = build_embedding_fallback_chain(
                purpose="evaluation",
                app_settings=app_settings,
                provider_rate_limiters=provider_rate_limiters,
                health_registry=health_registry,
            )
            embeddings = AdaptiveRagasEmbeddings(embedding_chain)
        elif not isinstance(embeddings, BaseRagasEmbeddings):
            embeddings = LangchainEmbeddingsWrapper(embeddings)

        return llm, embeddings

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truth: list[str] | None = None,
        llm: Any = None,
        embeddings: Any = None,
    ) -> RagasEvalResult | None:
        """Run RAGAS evaluation on a batch of Q&A pairs.

        Returns ``RagasEvalResult`` or ``None`` if ragas unavailable.
        """
        if not self._lazy_init():
            return None

        _install_vertexai_shim()

        from datasets import Dataset  # type: ignore[import-not-found]  # lazy optional dep

        data: dict[str, list] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truth:
            data["ground_truth"] = ground_truth

        dataset = Dataset.from_dict(data)
        llm_wrapper, embeddings_wrapper = self._build_runtime(llm, embeddings)
        if self._metrics:
            for metric in self._metrics:
                metric.llm = llm_wrapper
                metric.embeddings = embeddings_wrapper

        evaluate_fn = self._evaluate
        assert evaluate_fn is not None
        from ragas.run_config import RunConfig  # type: ignore[import-not-found]  # lazy optional dep

        result = evaluate_fn(
            dataset=dataset,
            metrics=self._metrics,
            llm=llm_wrapper,
            run_config=RunConfig(timeout=self.run_timeout, max_workers=self.max_workers),
        )

        # RAGAS returns an EvaluationResult; `result[key]` is a per-sample score
        # list (KeyError when a metric did not run).
        def _score(key: str) -> float:
            try:
                scores = result[key]
            except KeyError:
                logger.warning("RAGAS metric %r did not run — scoring 0.0", key)
                return 0.0
            return float(sum(scores) / len(scores)) if scores else 0.0

        recall = _score("context_recall")
        precision = _score("context_precision")
        faithful = _score("faithfulness")
        relevancy = _score("answer_relevancy")

        return RagasEvalResult(
            context_recall=recall,
            context_precision=precision,
            faithfulness=faithful,
            answer_relevancy=relevancy,
            overall=recall * 0.3 + faithful * 0.4 + relevancy * 0.3,
        )
