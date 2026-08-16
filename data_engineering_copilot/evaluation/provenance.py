"""Provenance capture for RAG evaluation runs.

Captures the code and configuration fingerprint of an evaluation so results
can be attributed to a specific revision, generation, and retrieval
configuration. ``config_fingerprint`` feeds drift analysis — a metric change
that coincides with a fingerprint change is expected, not a regression.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_FIELDS: dict[str, str] = {
    "openrouter": "openrouter_embedding_model",
    "nvidia": "nvidia_embedding_model",
    "gemini": "gemini_embedding_model",
    "local-hf": "local_hf_embedding_model",
    "huggingface": "huggingface_embedding_model",
    "ollama": "embedding_model_name",
    "local": "embedding_model_name",
}

_RERANK_MODEL_FIELDS: dict[str, str] = {
    "openrouter": "openrouter_rerank_model",
    "nvidia": "nvidia_rerank_model",
    "huggingface": "huggingface_rerank_model",
}


def git_commit(repo_root: str | Path | None = None, short: bool = True) -> str:
    """Return the current git commit (``''`` when unavailable).

    Never raises: absence of git, a missing repo, or a slow ``git`` call all
    yield an empty string so evaluation never fails on provenance capture.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(repo_root) if repo_root else None,
        )
        if proc.returncode != 0:
            return ""
        sha = proc.stdout.strip()
        return sha[:12] if short else sha
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def active_generation(settings) -> str:
    """Resolve the active index generation from settings."""
    return str(getattr(settings, "active_index_generation", "") or "").strip()


def embedding_model(settings) -> str:
    """Resolve the effective embedding model for the configured provider."""
    provider = str(getattr(settings, "embedding_provider", "") or "").lower()
    field = _EMBEDDING_MODEL_FIELDS.get(provider, "embedding_model_name")
    return str(getattr(settings, field, "") or "")


def reranker(settings) -> str:
    """Resolve the effective reranker for the configured order.

    Returns the first configured rerank provider's model, or the local
    cross-encoder when no cloud reranker is configured.
    """
    if not getattr(settings, "reranker_enabled", False):
        return ""
    for provider in getattr(settings, "rerank_fallback_order", []) or []:
        field = _RERANK_MODEL_FIELDS.get(str(provider).lower())
        if field and getattr(settings, field, ""):
            return str(getattr(settings, field))
    return str(getattr(settings, "reranker_model", "") or "")


def eval_environment(settings) -> dict:
    """Stable, serialisable description of the evaluation environment."""
    return {
        "git_commit": git_commit(),
        "generation": active_generation(settings),
        "embedding_model": embedding_model(settings),
        "reranker": reranker(settings),
        "chunk_size": int(getattr(settings, "chunk_size_words", 0) or 0),
        "chunk_overlap": int(getattr(settings, "chunk_overlap_words", 0) or 0),
        "retrieval_top_k": int(getattr(settings, "retrieval_top_k", 0) or 0),
    }


def config_fingerprint(settings) -> str:
    """Hash of the eval-relevant configuration (provenance without git state).

    Only configuration knobs that materially change retrieval/generation output
    are hashed — provider keys and transient state are excluded.
    """
    env = eval_environment(settings)
    env.pop("git_commit", None)
    blob = json.dumps(env, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


_ANSWER_CONFIG_FIELDS: tuple[str, ...] = (
    "answer_llm_provider",
    "answer_llm_model",
    "rewrite_llm_provider",
    "groundedness_llm_provider",
    "scope_check_enabled",
)


def answer_config_fingerprint(settings) -> str:
    """Hash of the answer-generation LLM configuration (cache scoping).

    The query cache must never serve an answer produced under a different LLM
    chain: a scope-gate verdict or generation is provider/model specific. This
    hashes the answer-purpose provider + model plus the surrounding per-purpose
    overrides and the scope-gate switch, so a config change invalidates cached
    answers structurally (different cache key), not by expiry.
    """
    payload = {}
    for field in _ANSWER_CONFIG_FIELDS:
        payload[field] = getattr(settings, field, "")
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
