"""Provider catalog: free_forever model inventory + probe-driven fallback ranking.

Loads ``config/free_tier_models.json`` (curated, committed) and the live probe
output ``data/provider_catalog.json`` (gitignored) to produce smart fallback
orders. Fail-open: missing/stale catalog falls back to ``settings.llm_fallback_order``.

Only ``tier == free_forever`` models are ranked — ``free_credit`` / paid are
excluded by construction (the curated file never lists them).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

CATALOG_STALE_DAYS = 7


@dataclass(frozen=True, slots=True)
class CatalogModel:
    provider: str
    model: str
    tier: str = "free_forever"
    context_window: int = 8192
    max_tokens_field: str = "max_tokens"
    supports_structured_output: bool = False
    rag_suitable: bool = True
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ProbeEntry:
    provider: str
    model: str
    status: str  # OK | FAIL | SKIP
    latency_ms: float | None = None
    http_status: int | None = None
    category: str | None = None
    retry_after: float | None = None
    message: str = ""
    tier: str = "free_forever"
    rag_suitable: bool = True
    context_window: int = 8192
    supports_structured_output: bool = False
    kind: str = "llm"  # llm | embedding | rerank
    dimension: int | None = None


@dataclass
class ProviderCatalog:
    generated_at: str = ""
    probes: list[ProbeEntry] = field(default_factory=list)
    recommended_fallback_order: dict[str, list[str]] = field(default_factory=dict)
    # Dedicated fallback lists for embedding + rerank (each maps to settings embedding_fallback_order / rerank_fallback_order)
    embedding_probes: list[ProbeEntry] = field(default_factory=list)
    embedding_fallback_order: list[str] = field(default_factory=list)
    rerank_probes: list[ProbeEntry] = field(default_factory=list)
    rerank_fallback_order: list[str] = field(default_factory=list)


def _load_models_list(data: dict, key: str, path: Path) -> list[CatalogModel]:
    raw_models = data.get(key)
    if raw_models is None:
        return []
    if not isinstance(raw_models, list):
        raise ValueError(f"free tier models file '{key}' must be a list: {path}")
    out: list[CatalogModel] = []
    seen: set[tuple[str, str]] = set()
    for idx, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise ValueError(f"{key}[{idx}] must be an object")
        provider = str(raw.get("provider", "")).strip().lower()
        model = str(raw.get("model", "")).strip()
        if not provider or not model:
            raise ValueError(f"{key}[{idx}] must define non-empty provider/model")
        cur_key = (provider, model)
        if cur_key in seen:
            raise ValueError(f"duplicate provider/model at {key}[{idx}]: {provider}/{model}")
        seen.add(cur_key)
        tier = str(raw.get("tier", "free_forever")).strip()
        if tier not in ("free_forever", "promotion_free"):
            raise ValueError(f"{key}[{idx}] tier must be 'free_forever' or 'promotion_free', got {tier!r}")
        out.append(
            CatalogModel(
                provider=provider,
                model=model,
                tier=tier,
                context_window=int(raw.get("context_window", 8192)),
                max_tokens_field=str(raw.get("max_tokens_field", "max_tokens")),
                supports_structured_output=bool(raw.get("supports_structured_output", False)),
                rag_suitable=bool(raw.get("rag_suitable", True)),
                notes=str(raw.get("notes", "")),
            )
        )
    return out


def load_free_tier_models(path: Path) -> list[CatalogModel]:
    """Load ``config/free_tier_models.json`` (LLM models at key ``models``).

    Raises ``ValueError`` / ``OSError`` on invalid file.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out = _load_models_list(data, "models", path)
    if not out:
        raise ValueError(f"free tier models file must contain non-empty 'models' list: {path}")
    return out


def load_embedding_models(path: Path) -> list[CatalogModel]:
    """Load embedding inventory at ``embedding_models`` (may be empty)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return _load_models_list(data, "embedding_models", path)


def load_rerank_models(path: Path) -> list[CatalogModel]:
    """Load rerank inventory at ``rerank_models`` (may be empty)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return _load_models_list(data, "rerank_models", path)


def is_rag_suitable(model: CatalogModel, purpose: str | None = None) -> bool:
    """Whether *model* is suitable for RAG *purpose*.

    Hard gates: ``rag_suitable`` must be true and ``context_window >= 8192``.
    For ``answer``/``code`` intents we additionally require structured output
    support (schema-enforced JSON) — otherwise answer parsing degrades.
    """
    if not model.rag_suitable:
        return False
    if model.context_window < 8192:
        return False
    if purpose in ("answer", "code") and not model.supports_structured_output:  # noqa: SIM103
        return False
    return True


def filter_rag_suitable(models: list[CatalogModel], purpose: str | None = None) -> list[CatalogModel]:
    return [m for m in models if is_rag_suitable(m, purpose)]


def compute_recommended_order(
    probes: list[ProbeEntry],
    purpose: str | None = None,
) -> list[str]:
    """Rank providers by fastest OK probe per provider (dedup).

    Only ``status == OK`` probes that are ``rag_suitable`` (and purpose-appropriate)
    participate. Within each provider we keep the fastest latency model; then
    sort providers by latency ascending.
    """
    # Keep fastest OK per provider
    best: dict[str, ProbeEntry] = {}
    for p in probes:
        if p.status != "OK":
            continue
        # Respect rag suitability per purpose (reuse same gate as static catalog)
        # ProbeEntry mirrors CatalogModel fields, so apply same logic:
        if not p.rag_suitable:
            continue
        if p.context_window < 8192:
            continue
        if purpose in ("answer", "code") and not p.supports_structured_output:
            continue
        # ollama is degraded_fallback — never in ranked order
        if p.provider == "ollama":
            continue
        cur = best.get(p.provider)
        if (
            cur is None
            or (p.latency_ms is not None and cur.latency_ms is not None and p.latency_ms < cur.latency_ms)
            or cur is None
            and p.latency_ms is None
            or cur is not None
            and cur.latency_ms is None
            and p.latency_ms is not None
        ):
            best[p.provider] = p

    ranked = sorted(best.values(), key=lambda e: e.latency_ms if e.latency_ms is not None else 1e9)
    return [e.provider for e in ranked]


def _parse_probe_list(raw_list: Any) -> list[ProbeEntry]:
    out: list[ProbeEntry] = []
    if not isinstance(raw_list, list):
        return out
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        out.append(
            ProbeEntry(
                provider=str(raw.get("provider", "")).lower(),
                model=str(raw.get("model", "")),
                status=str(raw.get("status", "FAIL")),
                latency_ms=raw.get("latency_ms"),
                http_status=raw.get("http_status"),
                category=raw.get("category"),
                retry_after=raw.get("retry_after"),
                message=str(raw.get("message", "")),
                tier=str(raw.get("tier", "free_forever")),
                rag_suitable=bool(raw.get("rag_suitable", True)),
                context_window=int(raw.get("context_window", 8192)),
                supports_structured_output=bool(raw.get("supports_structured_output", False)),
                kind=str(raw.get("kind", "llm")),
                dimension=raw.get("dimension"),
            )
        )
    return out


def load_provider_catalog(path: Path) -> ProviderCatalog | None:
    """Load ``data/provider_catalog.json`` if present, else ``None`` (fail-open)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    probes = _parse_probe_list(data.get("probes", []))
    embedding_probes = _parse_probe_list(data.get("embedding_probes", []))
    rerank_probes = _parse_probe_list(data.get("rerank_probes", []))
    # Backward compat: older catalogs stored only llm probes in 'probes'
    rec = data.get("recommended_fallback_order", {})
    if not isinstance(rec, dict):
        rec = {}
    norm_rec: dict[str, list[str]] = {}
    for k, v in rec.items():
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            norm_rec[str(k)] = [str(x).lower() for x in v]
    emb_order = data.get("embedding_fallback_order", [])
    if not isinstance(emb_order, list) or not all(isinstance(x, str) for x in emb_order):
        emb_order = []
    rerank_order = data.get("rerank_fallback_order", [])
    if not isinstance(rerank_order, list) or not all(isinstance(x, str) for x in rerank_order):
        rerank_order = []
    return ProviderCatalog(
        generated_at=str(data.get("generated_at", "")),
        probes=probes,
        recommended_fallback_order=norm_rec,
        embedding_probes=embedding_probes,
        embedding_fallback_order=[str(x).lower() for x in emb_order],
        rerank_probes=rerank_probes,
        rerank_fallback_order=[str(x).lower() for x in rerank_order],
    )


def is_catalog_stale(catalog: ProviderCatalog | None, stale_days: int = CATALOG_STALE_DAYS) -> bool:
    """Return True when catalog is missing or older than *stale_days*."""
    if catalog is None or not catalog.generated_at:
        return True
    try:
        # generated_at is ISO8601; compare via epoch if parseable, else stale
        from datetime import datetime

        # Handle trailing Z
        ts = catalog.generated_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - dt).total_seconds() / 86400
        return age_days > stale_days
    except Exception:
        return True


def get_catalog_fallback_order(
    purpose: str,
    catalog: ProviderCatalog | None,
) -> list[str] | None:
    """Return catalog-driven order for *purpose* if available, else ``None``.

    Fail-open: missing catalog, empty order, or stale-with-no-order all return
    ``None`` so callers fall back to ``settings.llm_fallback_order``.
    """
    if catalog is None or not catalog.recommended_fallback_order:
        return None
    # Per-purpose order wins; fallback to global
    order = catalog.recommended_fallback_order.get(purpose)
    if not order:
        order = catalog.recommended_fallback_order.get("global")
    if not order:
        return None
    return list(order)


def build_catalog_probe_entries(models: list[CatalogModel], kind: str = "llm") -> list[ProbeEntry]:
    """Convert static catalog models to ProbeEntry skeletons (status=SKIP)."""
    return [
        ProbeEntry(
            provider=m.provider,
            model=m.model,
            status="SKIP",
            tier=m.tier,
            rag_suitable=m.rag_suitable,
            context_window=m.context_window,
            supports_structured_output=m.supports_structured_output,
            kind=kind,
        )
        for m in models
    ]


def compute_embedding_order(probes: list[ProbeEntry]) -> list[str]:
    """Fastest OK embedding provider per provider (dedup).

    local-hf is excluded — it was removed from the fallback chain to avoid
    slow CPU-bound embedding when external providers are available.
    """
    best: dict[str, ProbeEntry] = {}
    for p in probes:
        if p.status != "OK":
            continue
        if p.kind != "embedding":
            continue
        if p.provider == "local-hf":
            continue
        cur = best.get(p.provider)
        if (
            cur is None
            or (p.latency_ms is not None and cur.latency_ms is not None and p.latency_ms < cur.latency_ms)
            or cur is None
            and p.latency_ms is None
        ):
            best[p.provider] = p
    ranked = sorted(best.values(), key=lambda e: e.latency_ms if e.latency_ms is not None else 1e9)
    return [e.provider for e in ranked]


def compute_rerank_order(probes: list[ProbeEntry]) -> list[str]:
    """Fastest OK rerank provider per provider."""
    best: dict[str, ProbeEntry] = {}
    for p in probes:
        if p.status != "OK":
            continue
        if p.kind != "rerank":
            continue
        cur = best.get(p.provider)
        if (
            cur is None
            or (p.latency_ms is not None and cur.latency_ms is not None and p.latency_ms < cur.latency_ms)
            or cur is None
            and p.latency_ms is None
        ):
            best[p.provider] = p
    ranked = sorted(best.values(), key=lambda e: e.latency_ms if e.latency_ms is not None else 1e9)
    return [e.provider for e in ranked]


def serialize_catalog(catalog: ProviderCatalog) -> dict[str, Any]:
    return {
        "generated_at": catalog.generated_at,
        "probes": [asdict(p) for p in catalog.probes],
        "recommended_fallback_order": catalog.recommended_fallback_order,
        "embedding_probes": [asdict(p) for p in catalog.embedding_probes],
        "embedding_fallback_order": catalog.embedding_fallback_order,
        "rerank_probes": [asdict(p) for p in catalog.rerank_probes],
        "rerank_fallback_order": catalog.rerank_fallback_order,
    }
