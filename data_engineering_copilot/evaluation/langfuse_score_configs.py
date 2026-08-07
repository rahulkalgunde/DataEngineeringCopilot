"""Idempotent Langfuse score-config seeding (Phase 7, Task 7.3).

Defines the score configs used by the RAG pipeline and evaluators, and seeds
any missing ones into Langfuse via the v4 ``api.score_configs.create`` client.
See ``docs/langfuse_evaluators.md`` for the catalog and UI instructions.
"""

from __future__ import annotations

import logging
from typing import Any

from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

logger = logging.getLogger(__name__)

# name -> (data_type, {extra create kwargs}). Keep in sync with
# ``docs/langfuse_evaluators.md`` score-config table.
SCORE_CONFIGS: dict[str, tuple[str, dict[str, Any]]] = {
    "confidence": ("NUMERIC", {"min_value": 0.0, "max_value": 1.0, "description": "Answer confidence (0-1)"}),
    "groundedness": ("NUMERIC", {"min_value": 0.0, "max_value": 1.0, "description": "Groundedness in retrieved docs"}),
    "relevance": ("NUMERIC", {"min_value": 0.0, "max_value": 1.0, "description": "Answer relevance to the question"}),
    "faithfulness": ("NUMERIC", {"min_value": 0.0, "max_value": 1.0, "description": "LLM-as-judge faithfulness score"}),
    "user_feedback": ("NUMERIC", {"min_value": 0.0, "max_value": 1.0, "description": "Thumbs up/down (1/0)"}),
    "completeness": ("NUMERIC", {"min_value": 0.0, "max_value": 1.0, "description": "Answer completeness heuristic"}),
    "cache_hit": ("BOOLEAN", {"description": "Answer served from the query cache"}),
    "out_of_scope": ("BOOLEAN", {"description": "Question not answerable from the docs"}),
    "intent": (
        "CATEGORICAL",
        {
            "categories": [
                {"value": 0.0, "label": "factual"},
                {"value": 1.0, "label": "code_example"},
                {"value": 2.0, "label": "api_lookup"},
                {"value": 3.0, "label": "comparative"},
                {"value": 4.0, "label": "debugging"},
                {"value": 5.0, "label": "how_to"},
            ],
            "description": "Rewriter intent classification",
        },
    ),
    "ragas_context_recall": ("NUMERIC", {"min_value": 0.0, "max_value": 1.0, "description": "RAGAS context recall"}),
    "ragas_context_precision": (
        "NUMERIC",
        {"min_value": 0.0, "max_value": 1.0, "description": "RAGAS context precision"},
    ),
    "ragas_faithfulness": ("NUMERIC", {"min_value": 0.0, "max_value": 1.0, "description": "RAGAS faithfulness"}),
    "ragas_answer_relevancy": (
        "NUMERIC",
        {"min_value": 0.0, "max_value": 1.0, "description": "RAGAS answer relevancy"},
    ),
}

_CONFIG_KEY = "name"

# name -> score-config id, resolved lazily and cached. Clearing resets it.
_CONFIG_ID_CACHE: dict[str, str | None] = {}


def get_score_config_id(name: str) -> str | None:
    """Return the Langfuse score-config id for ``name`` (cached, None on miss).

    Used by the RAG pipeline to emit config-bound categorical scores so labels
    render in the UI. Never raises.
    """
    if name in _CONFIG_ID_CACHE:
        return _CONFIG_ID_CACHE[name]
    config_id: str | None = None
    client = get_langfuse_instance()
    if client is not None:
        try:
            score_configs = client._client.api.score_configs
            page = 1
            while config_id is None:
                result = score_configs.get(page=page, limit=100)
                items = getattr(result, "data", None) or []
                if not items:
                    break
                for item in items:
                    if getattr(item, "name", None) == name:
                        config_id = getattr(item, "id", None)
                        break
                if len(items) < 100:
                    break
                page += 1
        except Exception as exc:
            logger.warning("Failed to resolve score-config id for %r: %s", name, exc)
    _CONFIG_ID_CACHE[name] = config_id
    return config_id


def _existing_configs(score_configs_client) -> set[str]:
    """Return the set of existing score-config names (paged list)."""
    names: set[str] = set()
    page = 1
    while True:
        result = score_configs_client.get(page=page, limit=100)
        items = getattr(result, "data", None) or getattr(result, "items", None) or []
        if not items:
            break
        for item in items:
            name = getattr(item, "name", None)
            if name:
                names.add(name)
        total = getattr(result, "meta", None)
        total_count = getattr(total, "total_items", None) if total is not None else None
        fetched = page * 100
        if total_count is not None and fetched >= total_count:
            break
        if len(items) < 100:
            break
        page += 1
    return names


def seed_score_configs(description_suffix: str | None = None) -> dict[str, bool]:
    """Idempotently create any missing score configs.

    Returns ``{name: created}``. Skips configs that already exist (matched by
    name) so re-runs are safe.
    """
    from langfuse.api.commons.types.score_config_data_type import ScoreConfigDataType

    client = get_langfuse_instance()
    if client is None:
        raise RuntimeError("Langfuse is unavailable; cannot seed score configs")
    score_configs = client._client.api.score_configs

    existing = _existing_configs(score_configs)
    created: dict[str, bool] = {}
    for name, (data_type, extra) in SCORE_CONFIGS.items():
        if name in existing:
            _reconcile_existing_config(score_configs, name, data_type, extra)
            created[name] = False
            continue
        kwargs = dict(extra)
        if description_suffix:
            kwargs["description"] = f"{kwargs.get('description', name)} ({description_suffix})"
        try:
            score_configs.create(
                name=name,
                data_type=ScoreConfigDataType(data_type),
                **kwargs,
            )
            created[name] = True
            logger.info("Created score config %r (%s)", name, data_type)
        except Exception as exc:
            logger.warning("Failed to create score config %r: %s", name, exc)
            created[name] = False
    return created


def _reconcile_existing_config(score_configs, name: str, data_type: str, extra: dict[str, Any]) -> None:
    """Update an existing config whose type/categories/range drifted from the catalog.

    Best-effort: mismatches (e.g. a categorical config missing newly added
    intent categories) are reconciled via ``update``; failures are logged.
    """
    expected_categories = extra.get("categories")
    if expected_categories is None:
        return
    try:
        result = score_configs.get(limit=100)
        for item in getattr(result, "data", None) or []:
            if getattr(item, "name", None) != name:
                continue
            current = getattr(item, "categories", None) or []
            current_labels = {c.label for c in current}
            expected_labels = {c["label"] for c in expected_categories}
            if current_labels != expected_labels:
                score_configs.update(
                    item.id,
                    categories=[{"value": c["value"], "label": c["label"]} for c in expected_categories],
                )
                logger.info("Updated score config %r categories", name)
            return
    except Exception as exc:
        logger.warning("Failed to reconcile score config %r: %s", name, exc)
