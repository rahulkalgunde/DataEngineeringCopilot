"""CLI: probe free_forever catalog and emit ranked fallback orders.

Usage:
    dec probe-catalog [--providers groq,zai] [--purpose answer] [--timeout 10] [--json] [--offline] [--output path]
    dec_venv/bin/dec probe-catalog --json   # live probes (requires API keys)
    dec_venv/bin/dec probe-catalog --offline  # skeleton without network

Writes ``data/provider_catalog.json`` (and respects ``settings.provider_catalog_path``)
with ``generated_at``, ``probes``, ``recommended_fallback_order`` per purpose,
plus ``embedding_probes``/``embedding_fallback_order`` and ``rerank_probes``/``rerank_fallback_order``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.services.provider_catalog import (
    CatalogModel,
    ProbeEntry,
    ProviderCatalog,
    compute_embedding_order,
    compute_recommended_order,
    compute_rerank_order,
    is_catalog_stale,
    load_embedding_models,
    load_free_tier_models,
    load_rerank_models,
    serialize_catalog,
)

try:
    from data_engineering_copilot.cli_llm_probe import ProbeTarget, _probe_llm_target  # type: ignore[import-not-found]
except Exception:
    _probe_llm_target = None  # type: ignore[assignment]
    ProbeTarget = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_PURPOSES = ["global", "answer", "rewrite", "groundedness", "intent", "enrichment", "evaluation", "code"]


def _catalog_probe_targets(models: list[CatalogModel], only_providers: set[str] | None) -> list[CatalogModel]:
    if not only_providers:
        return models
    return [m for m in models if m.provider.lower() in only_providers]


async def _probe_one(model: CatalogModel, app_settings: AppSettings, prompt: str, timeout: float) -> ProbeEntry:
    if _probe_llm_target is None or ProbeTarget is None:
        return ProbeEntry(
            provider=model.provider,
            model=model.model,
            status="SKIP",
            message="probe unavailable",
            tier=model.tier,
            rag_suitable=model.rag_suitable,
            context_window=model.context_window,
            supports_structured_output=model.supports_structured_output,
            kind="llm",
        )
    if model.provider in ("local-hf",):
        return ProbeEntry(
            provider=model.provider,
            model=model.model,
            status="SKIP",
            message="embedding-only, not LLM-probed",
            tier=model.tier,
            rag_suitable=model.rag_suitable,
            context_window=model.context_window,
            supports_structured_output=model.supports_structured_output,
            kind="llm",
        )
    target = ProbeTarget(kind="llm", provider=model.provider, model=model.model, purpose=None)
    result = await _probe_llm_target(target, app_settings, prompt, timeout)  # type: ignore[arg-type]
    return ProbeEntry(
        provider=result.target.provider,
        model=result.target.model,
        status=result.status,
        latency_ms=result.latency_ms,
        http_status=result.http_status,
        category=result.category,
        retry_after=result.retry_after,
        message=result.message,
        tier=model.tier,
        rag_suitable=model.rag_suitable,
        context_window=model.context_window,
        supports_structured_output=model.supports_structured_output,
        kind="llm",
    )


async def _run_probes(
    models: list[CatalogModel],
    app_settings: AppSettings,
    prompt: str,
    timeout: float,
) -> list[ProbeEntry]:
    entries: list[ProbeEntry] = []
    for m in models:
        entries.append(await _probe_one(m, app_settings, prompt, timeout))
    return entries


async def _probe_embedding_one(model: CatalogModel, app_settings: AppSettings, timeout: float) -> ProbeEntry:
    import time

    try:
        if model.provider == "local-hf":
            from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import (
                LocalSentenceTransformerEmbeddings,
            )

            dimension = app_settings.embedding_model_dimensions.get(
                model.model, app_settings.default_embedding_dimension
            )
            client = LocalSentenceTransformerEmbeddings(
                model_name=model.model, embedding_dimension=dimension, batch_size=32
            )
            start = time.monotonic()
            vecs = await client.embed_texts(["hello world"])
            latency = (time.monotonic() - start) * 1000
            return ProbeEntry(
                provider=model.provider,
                model=model.model,
                status="OK" if vecs and len(vecs[0]) == dimension else "FAIL",
                latency_ms=round(latency, 1),
                tier=model.tier,
                rag_suitable=model.rag_suitable,
                kind="embedding",
                dimension=len(vecs[0]) if vecs else None,
            )
        from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
            OpenAICompatibleEmbeddings,
        )
        from data_engineering_copilot.infrastructure.huggingface_serverless_embeddings import (
            HuggingFaceServerlessEmbeddings,
        )

        dimension = 2048
        if model.provider == "nvidia":
            api_key = app_settings.nvidia_api_key.get_secret_value()
            base_url = app_settings.nvidia_base_url
            dimension = app_settings.embedding_model_dimensions.get(model.model, 2048)
            if not api_key:
                return ProbeEntry(
                    provider=model.provider,
                    model=model.model,
                    status="SKIP",
                    message="CONFIG: NVIDIA_API_KEY missing",
                    tier=model.tier,
                    kind="embedding",
                    dimension=dimension,
                )
            client = OpenAICompatibleEmbeddings(
                api_key=api_key,
                model_name=model.model,
                base_url=base_url,
                embedding_dimension=dimension,
                batch_size=32,
                include_provider_param=False,
            )
        elif model.provider == "openrouter":
            api_key = app_settings.openrouter_api_key.get_secret_value()
            base_url = app_settings.openrouter_base_url
            dimension = app_settings.embedding_model_dimensions.get(model.model, 2048)
            if not api_key:
                return ProbeEntry(
                    provider=model.provider,
                    model=model.model,
                    status="SKIP",
                    message="CONFIG: OPENROUTER_API_KEY missing",
                    tier=model.tier,
                    kind="embedding",
                    dimension=dimension,
                )
            client = OpenAICompatibleEmbeddings(
                api_key=api_key,
                model_name=model.model,
                base_url=base_url,
                embedding_dimension=dimension,
                batch_size=32,
                include_provider_param=True,
            )
        elif model.provider == "huggingface":
            api_key = app_settings.huggingface_api_key.get_secret_value()
            base_url = app_settings.huggingface_base_url
            dimension = app_settings.embedding_model_dimensions.get(model.model, 2048)
            if not api_key:
                return ProbeEntry(
                    provider=model.provider,
                    model=model.model,
                    status="SKIP",
                    message="CONFIG: HUGGINGFACE_API_KEY missing",
                    tier=model.tier,
                    kind="embedding",
                    dimension=dimension,
                )
            client = HuggingFaceServerlessEmbeddings(
                api_key=api_key, model_name=model.model, base_url=base_url, embedding_dimension=dimension, batch_size=32
            )
        else:
            return ProbeEntry(
                provider=model.provider,
                model=model.model,
                status="SKIP",
                message=f"unsupported embedding provider {model.provider}",
                tier=model.tier,
                kind="embedding",
            )
        start = time.monotonic()
        try:
            vecs = await asyncio.wait_for(client.embed_texts(["hello world"]), timeout=timeout)
        except TimeoutError as exc:
            return ProbeEntry(
                provider=model.provider,
                model=model.model,
                status="FAIL",
                message=f"Timeout: {exc}",
                tier=model.tier,
                kind="embedding",
                dimension=dimension,
            )
        latency = (time.monotonic() - start) * 1000
        dim_ok = vecs and len(vecs[0]) == dimension
        return ProbeEntry(
            provider=model.provider,
            model=model.model,
            status="OK" if dim_ok else "FAIL",
            latency_ms=round(latency, 1),
            message="" if dim_ok else f"dimension mismatch expected {dimension} got {len(vecs[0]) if vecs else 0}",
            tier=model.tier,
            kind="embedding",
            dimension=len(vecs[0]) if vecs else None,
        )
    except Exception as exc:
        return ProbeEntry(
            provider=model.provider,
            model=model.model,
            status="FAIL",
            message=f"{type(exc).__name__}: {exc}",
            tier=model.tier,
            kind="embedding",
        )


async def _probe_rerank_one(model: CatalogModel, app_settings: AppSettings, timeout: float) -> ProbeEntry:
    import time

    try:
        from data_engineering_copilot.infrastructure.rerank_clients import (
            HuggingFaceRerankClient,
            NvidiaRerankClient,
            OpenRouterRerankClient,
        )

        client = None
        if model.provider == "openrouter":
            api_key = app_settings.openrouter_api_key.get_secret_value()
            if not api_key:
                return ProbeEntry(
                    provider=model.provider,
                    model=model.model,
                    status="SKIP",
                    message="CONFIG: OPENROUTER_API_KEY missing",
                    tier=model.tier,
                    kind="rerank",
                )
            client = OpenRouterRerankClient(
                api_key=api_key,
                model_name=model.model,
                base_url=app_settings.openrouter_rerank_url,
                timeout_seconds=int(timeout),
            )
        elif model.provider == "nvidia":
            api_key = app_settings.nvidia_api_key.get_secret_value()
            if not api_key:
                return ProbeEntry(
                    provider=model.provider,
                    model=model.model,
                    status="SKIP",
                    message="CONFIG: NVIDIA_API_KEY missing",
                    tier=model.tier,
                    kind="rerank",
                )
            client = NvidiaRerankClient(
                api_key=api_key,
                model_name=model.model,
                base_url=app_settings.nvidia_rerank_url,
                timeout_seconds=int(timeout),
            )
        elif model.provider == "huggingface":
            api_key = app_settings.huggingface_api_key.get_secret_value()
            if not api_key:
                return ProbeEntry(
                    provider=model.provider,
                    model=model.model,
                    status="SKIP",
                    message="CONFIG: HUGGINGFACE_API_KEY missing",
                    tier=model.tier,
                    kind="rerank",
                )
            client = HuggingFaceRerankClient(
                api_key=api_key,
                model_name=model.model,
                base_url=app_settings.huggingface_base_url,
                timeout_seconds=int(timeout),
            )
        else:
            return ProbeEntry(
                provider=model.provider,
                model=model.model,
                status="SKIP",
                message=f"unsupported rerank provider {model.provider}",
                tier=model.tier,
                kind="rerank",
            )
        from data_engineering_copilot.domain.models import RerankRequest

        req = RerankRequest(query="hello", documents=["hello world", "goodbye"], top_n=1)
        start = time.monotonic()
        try:
            res = await asyncio.wait_for(client.call(req), timeout=timeout)  # type: ignore[arg-type]
        except TimeoutError as exc:
            return ProbeEntry(
                provider=model.provider,
                model=model.model,
                status="FAIL",
                message=f"Timeout: {exc}",
                tier=model.tier,
                kind="rerank",
            )
        latency = (time.monotonic() - start) * 1000
        ok = bool(res and getattr(res, "rankings", None) is not None)
        return ProbeEntry(
            provider=model.provider,
            model=model.model,
            status="OK" if ok else "FAIL",
            latency_ms=round(latency, 1),
            message="" if ok else "empty rerank result",
            tier=model.tier,
            kind="rerank",
        )
    except Exception as exc:
        return ProbeEntry(
            provider=model.provider,
            model=model.model,
            status="FAIL",
            message=f"{type(exc).__name__}: {exc}",
            tier=model.tier,
            kind="rerank",
        )


def _build_recommended(probes: list[ProbeEntry]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for purpose in _PURPOSES:
        out[purpose] = compute_recommended_order(probes, purpose if purpose != "global" else None)
    return out


def main(
    providers: list[str] | None = None,
    purpose: str | None = None,
    prompt: str = "Reply with exactly: pong",
    timeout: float = 10.0,
    json_output: bool = False,
    offline: bool = False,
    output: str | None = None,
) -> None:
    for name in (
        "data_engineering_copilot.factory",
        "data_engineering_copilot.infrastructure.async_openai_compatible_embeddings",
        "httpx",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    app_settings = settings
    free_tier_path = app_settings.free_tier_models_path
    catalog_path = Path(output) if output else app_settings.provider_catalog_path

    try:
        models = load_free_tier_models(free_tier_path)
        embedding_models = load_embedding_models(free_tier_path)
        rerank_models = load_rerank_models(free_tier_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load free tier models {free_tier_path}: {exc}", file=sys.stderr)
        sys.exit(2)

    only = {p.lower() for p in providers} if providers else None
    models = _catalog_probe_targets(models, only)
    embedding_models = _catalog_probe_targets(embedding_models, only)
    rerank_models = _catalog_probe_targets(rerank_models, only)
    if purpose and purpose not in _PURPOSES:
        print(f"ERROR: unknown purpose {purpose!r}, choose from {_PURPOSES}", file=sys.stderr)
        sys.exit(2)

    # LLM probes
    if not models:
        print("No LLM models match filter — nothing to probe.", file=sys.stderr)
        probes: list[ProbeEntry] = []
    elif offline:
        probes = [
            ProbeEntry(
                provider=m.provider,
                model=m.model,
                status="SKIP",
                message="offline mode — no probe",
                tier=m.tier,
                rag_suitable=m.rag_suitable,
                context_window=m.context_window,
                supports_structured_output=m.supports_structured_output,
                kind="llm",
            )
            for m in models
        ]
    else:
        probes = asyncio.run(_run_probes(models, app_settings, prompt, timeout))

    # Embedding probes
    if offline:
        embedding_probes = [
            ProbeEntry(
                provider=m.provider,
                model=m.model,
                status="SKIP",
                message="offline mode — no probe",
                tier=m.tier,
                kind="embedding",
            )
            for m in embedding_models
        ]
        rerank_probes = [
            ProbeEntry(
                provider=m.provider,
                model=m.model,
                status="SKIP",
                message="offline mode — no probe",
                tier=m.tier,
                kind="rerank",
            )
            for m in rerank_models
        ]
        embedding_order: list[str] = []
        rerank_order: list[str] = []
    else:

        async def _run_emb_rerank() -> tuple[list[ProbeEntry], list[ProbeEntry]]:
            emb: list[ProbeEntry] = []
            for m in embedding_models:
                emb.append(await _probe_embedding_one(m, app_settings, timeout))
            rer: list[ProbeEntry] = []
            for m in rerank_models:
                rer.append(await _probe_rerank_one(m, app_settings, timeout))
            return emb, rer

        embedding_probes, rerank_probes = asyncio.run(_run_emb_rerank())
        embedding_order = compute_embedding_order(embedding_probes)
        rerank_order = compute_rerank_order(rerank_probes)

    recommended = _build_recommended(probes)
    catalog = ProviderCatalog(
        generated_at=datetime.now(UTC).isoformat(),
        probes=probes,
        recommended_fallback_order=recommended,
        embedding_probes=embedding_probes,
        embedding_fallback_order=embedding_order,
        rerank_probes=rerank_probes,
        rerank_fallback_order=rerank_order,
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = catalog_path.with_suffix(catalog_path.suffix + ".tmp")
    tmp.write_text(json.dumps(serialize_catalog(catalog), indent=2), encoding="utf-8")
    tmp.replace(catalog_path)

    if json_output:
        print(json.dumps(serialize_catalog(catalog), indent=2))
    else:
        print(
            f"✅ Catalog written: {catalog_path} ({len(probes)} LLM, {len(embedding_probes)} embedding, {len(rerank_probes)} rerank probes)"
        )
        if is_catalog_stale(catalog, stale_days=app_settings.catalog_stale_days):
            print(f"⚠️  catalog stale threshold {app_settings.catalog_stale_days}d exceeded")
        ok = sum(1 for p in probes if p.status == "OK")
        fail = sum(1 for p in probes if p.status == "FAIL")
        skip = sum(1 for p in probes if p.status == "SKIP")
        print(f"  LLM — OK: {ok}  FAIL: {fail}  SKIP: {skip}")
        if embedding_probes:
            e_ok = sum(1 for p in embedding_probes if p.status == "OK")
            print(
                f"  Embedding — OK: {e_ok}/{len(embedding_probes)}  order: {', '.join(embedding_order) if embedding_order else '—'}"
            )
        if rerank_probes:
            r_ok = sum(1 for p in rerank_probes if p.status == "OK")
            print(
                f"  Rerank — OK: {r_ok}/{len(rerank_probes)}  order: {', '.join(rerank_order) if rerank_order else '—'}"
            )
        for purpose_key in _PURPOSES:
            order = recommended.get(purpose_key, [])
            if order:
                print(f"  {purpose_key}: {', '.join(order) if order else '—'}")
        if fail:
            for p in probes:
                if p.status == "FAIL":
                    print(f"  ❌ {p.provider}/{p.model}: {p.category or 'error'} — {p.message[:120]}")
        if skip and not offline:
            for p in probes:
                if p.status == "SKIP":
                    print(f"  ⚠️  {p.provider}/{p.model}: {p.message}")

    sys.exit(0)
