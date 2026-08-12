from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field

import httpx

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.factory import _build_purpose_llm_client, build_embedder
from data_engineering_copilot.infrastructure.adaptive_llm_router import _categorize_llm_error
from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import OpenAICompatibleEmbeddings
from data_engineering_copilot.infrastructure.llm_client import LLMClient, build_chat_messages

_PURPOSES = ["answer", "rewrite", "groundedness", "intent", "enrichment", "evaluation", "code"]

_DEFAULT_PROMPT = "Reply with exactly: pong"

# Noisy loggers whose INFO output would interleave with the probe report.
_QUIET_LOGGERS = (
    "data_engineering_copilot.factory",
    "data_engineering_copilot.infrastructure.async_openai_compatible_embeddings",
    "httpx",
)


@dataclass
class ProbeTarget:
    kind: str  # "llm" | "embedding"
    provider: str
    model: str
    roles: list[str] = field(default_factory=list)
    purpose: str | None = None

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.provider}/{self.model}"


@dataclass
class ProbeResult:
    target: ProbeTarget
    status: str  # "OK" | "FAIL" | "SKIP"
    http_status: int | None = None
    latency_ms: float | None = None
    category: str | None = None
    retry_after: float | None = None
    message: str = ""
    response_preview: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dimension: int | None = None
    expected_dimension: int | None = None
    request_headers: dict[str, str] = field(default_factory=dict)


def _redact_headers(headers: httpx.Headers | dict) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() == "authorization":
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = str(value)
    return redacted


def _enumerate_llm_targets(app_settings: AppSettings, only_providers: set[str] | None) -> list[ProbeTarget]:
    """Enumerate (provider, model, role) targets the app would actually use.

    Uses the same ``_build_purpose_llm_client`` resolution the factory relies
    on, but the probed model is the *provider default* (``{provider}_model``
    or global ``llm_model``).  Targets are keyed by provider and deduplicated
    — exactly one probe per provider — while roles (global / purposes /
    fallback) accumulate on that single target.

    Providers whose client cannot be built (e.g. missing API key) are still
    emitted with a best-effort model so the probe can report a ``SKIP`` /
    ``CONFIG`` result instead of hiding the provider.
    """
    targets: list[ProbeTarget] = []
    index: dict[str, ProbeTarget] = {}

    def _add(provider: str, role: str) -> None:
        provider = provider.lower()
        if only_providers and provider not in only_providers:
            return
        target = index.get(provider)
        if target is None:
            model = _resolve(provider, "", None)
            target = ProbeTarget(kind="llm", provider=provider, model=model, purpose=None)
            index[provider] = target
            targets.append(target)
        if role not in target.roles:
            target.roles.append(role)

    def _resolve(provider: str, model: str, purpose: str | None) -> str:
        try:
            client = _build_purpose_llm_client(
                provider=provider,
                model=model,
                app_settings=app_settings,
                purpose=purpose,
            )
        except ValueError:
            return model or app_settings.llm_model
        return client.model if client is not None else (model or app_settings.llm_model)

    # Global client (mirrors build_global_llm_client: model stays at priority 4)
    _add(app_settings.llm_provider, "global")

    # Per-purpose primaries
    for purpose in _PURPOSES:
        provider = getattr(app_settings, f"{purpose}_llm_provider")
        if not provider:
            continue
        _add(provider, purpose)

    # Fallback-order providers (skipped nvidia — never used in LLM chains)
    for provider in app_settings.llm_fallback_order:
        provider = provider.lower()
        if provider == "nvidia":
            continue
        _add(provider, "fallback")

    return targets


def _build_llm_request_payload(client: LLMClient, prompt: str) -> dict:
    payload: dict = {
        "model": client.model,
        "messages": build_chat_messages(prompt),
        "temperature": client._temperature,
    }
    if client._max_tokens is not None and client._max_tokens > 0:
        payload[client._max_tokens_field] = client._max_tokens
    if client._extra_body:
        payload.update(client._extra_body)
    return payload


async def _probe_llm_target(
    target: ProbeTarget,
    app_settings: AppSettings,
    prompt: str,
    timeout: float,
) -> ProbeResult:
    result = ProbeResult(target=target, status="SKIP")
    try:
        client = _build_purpose_llm_client(
            provider=target.provider,
            model=target.model,
            app_settings=app_settings,
            timeout_seconds=int(timeout),
            purpose=target.purpose,
        )
    except ValueError as exc:
        result.message = f"CONFIG: {exc}"
        return result
    if client is None:
        result.message = "CONFIG: no client could be built"
        return result

    client_kwargs = client._make_client_kwargs()
    headers = client_kwargs.get("headers", {})
    result.request_headers = _redact_headers(headers)
    payload = _build_llm_request_payload(client, prompt)

    try:
        async with httpx.AsyncClient(
            base_url=client.base_url,
            timeout=httpx.Timeout(timeout),
            headers=headers,
        ) as http:
            start = time.monotonic()
            response = await http.post(client._endpoint_path, json=payload)
            latency_ms = (time.monotonic() - start) * 1000
            result.latency_ms = round(latency_ms, 1)
            result.http_status = response.status_code
            result.request_headers = _redact_headers(response.request.headers)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        result.retry_after = float(retry_after)
                    except ValueError:
                        result.retry_after = None
                raise httpx.HTTPStatusError(
                    "Rate limited by provider",
                    request=response.request,
                    response=response,
                )

            response.raise_for_status()
            data = response.json()
            result.status = "OK"
            usage = data.get("usage", {})
            result.prompt_tokens = int(usage.get("prompt_tokens", 0))
            result.completion_tokens = int(usage.get("completion_tokens", 0))
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            result.response_preview = content.strip()[:200]
    except httpx.HTTPStatusError as exc:
        p_err = _categorize_llm_error(exc, target.provider, target.model)
        result.status = "FAIL"
        result.category = p_err.category.value
        result.http_status = exc.response.status_code
        result.message = _summarize_error(exc)
    except (httpx.TimeoutException, httpx.ConnectError, OSError) as exc:
        result.status = "FAIL"
        result.category = "retryable"
        result.message = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        result.status = "FAIL"
        result.category = "permanent_error"
        result.message = f"{type(exc).__name__}: {exc}"

    return result


def _summarize_error(exc: httpx.HTTPStatusError) -> str:
    status = exc.response.status_code
    body = exc.response.text[:300].strip()
    if status == 429:
        retry_after = exc.response.headers.get("Retry-After")
        hint = f" (Retry-After: {retry_after}s)" if retry_after else ""
        return f"RATE_LIMITED{hint}: {body}"
    if status in (401, 403):
        return f"AUTH: HTTP {status}: {body}"
    return f"HTTP {status}: {body}"


async def _probe_embedding(
    app_settings: AppSettings,
    timeout: float,
    prompt: str,
) -> ProbeResult:
    provider = app_settings.embedding_provider.lower()
    target = ProbeTarget(kind="embedding", provider=provider, model="", roles=["embedding"])
    result = ProbeResult(target=target, status="SKIP")
    try:
        embedder = build_embedder(app_settings)
    except ValueError as exc:
        result.message = f"CONFIG: {exc}"
        return result

    if isinstance(embedder, OpenAICompatibleEmbeddings):
        target.model = embedder.model_name
        payload: dict = {"model": embedder.model_name, "input": [prompt]}
        if embedder._include_provider_param:
            payload["provider"] = {"truncate": "END"}
        endpoint = "/embeddings"
        headers = {"Authorization": f"Bearer {embedder.api_key}"}
    elif getattr(type(embedder), "__module__", "").endswith("local_sentence_transformer_embeddings"):
        # Local HF provider: no HTTP endpoint. Verify by running a local embed.
        target.model = embedder.model_name
        result.expected_dimension = app_settings.get_embedding_dimension()
        try:
            start = time.monotonic()
            vectors = await embedder.embed_texts([prompt])
            latency_ms = (time.monotonic() - start) * 1000
            result.latency_ms = round(latency_ms, 1)
            result.status = "OK"
            result.dimension = len(vectors[0]) if vectors else 0
        except Exception as exc:
            result.status = "FAIL"
            result.category = "permanent_error"
            result.message = f"{type(exc).__name__}: {exc}"
        return result
    else:
        target.model = embedder.model_name
        payload = {"model": embedder.model_name, "input": [prompt]}
        endpoint = "/api/embed"
        headers = {}

    result.expected_dimension = app_settings.get_embedding_dimension()
    result.request_headers = _redact_headers(headers)
    base_url = embedder.base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers=headers,
        ) as http:
            start = time.monotonic()
            response = await http.post(endpoint, json=payload)
            latency_ms = (time.monotonic() - start) * 1000
            result.latency_ms = round(latency_ms, 1)
            result.http_status = response.status_code
            if response.status_code == 429:
                raise httpx.HTTPStatusError(
                    "Rate limited by embedding provider",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            data = response.json()
            result.status = "OK"
            if isinstance(embedder, OpenAICompatibleEmbeddings):
                embeddings = data.get("data", [])
            else:
                embeddings = data.get("embeddings", [])
            if embeddings:
                result.dimension = len(embeddings[0])
    except httpx.HTTPStatusError as exc:
        result.status = "FAIL"
        result.category = "rate_limited" if exc.response.status_code == 429 else "http_error"
        result.http_status = exc.response.status_code
        result.message = _summarize_error(exc)
    except (httpx.TimeoutException, httpx.ConnectError, OSError) as exc:
        result.status = "FAIL"
        result.category = "retryable"
        result.message = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        result.status = "FAIL"
        result.category = "permanent_error"
        result.message = f"{type(exc).__name__}: {exc}"

    return result


def _verdict(result: ProbeResult) -> str:
    if result.status == "OK":
        return "OK"
    if result.status == "SKIP":
        return "SKIP"
    return "FAIL"


def _format_table(results: list[ProbeResult]) -> str:
    headers = ["Kind", "Provider", "Model", "Roles", "Status", "HTTP", "Latency", "Verdict"]
    rows: list[list[str]] = []
    for r in results:
        latency = f"{r.latency_ms:,.0f}ms" if r.latency_ms is not None else "—"
        http = str(r.http_status) if r.http_status is not None else "—"
        rows.append(
            [
                r.target.kind,
                r.target.provider,
                r.target.model,
                ", ".join(r.target.roles),
                r.status,
                http,
                latency,
                _verdict(r),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines = []
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def _serialize(results: list[ProbeResult]) -> list[dict]:
    out: list[dict] = []
    for r in results:
        entry = {
            "kind": r.target.kind,
            "provider": r.target.provider,
            "model": r.target.model,
            "roles": r.target.roles,
            "status": r.status,
            "http_status": r.http_status,
            "latency_ms": r.latency_ms,
            "category": r.category,
            "retry_after": r.retry_after,
            "message": r.message,
            "response_preview": r.response_preview,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "dimension": r.dimension,
            "expected_dimension": r.expected_dimension,
        }
        out.append(entry)
    return out


async def _run_probes(
    app_settings: AppSettings,
    providers: list[str] | None,
    purpose: str | None,
    prompt: str,
    timeout: float,
    include_embeddings: bool,
) -> list[ProbeResult]:
    only_providers = {p.lower() for p in providers} if providers else None
    targets = _enumerate_llm_targets(app_settings, only_providers)

    if purpose:
        filtered = [t for t in targets if purpose in t.roles]
        if not filtered:
            raise ValueError(
                f"No LLM target found for purpose {purpose!r}. "
                f"Available purposes: {sorted({role for t in targets for role in t.roles})}"
            )
        targets = filtered

    # nvidia is never used in LLM chains — only probe it when explicitly requested
    if only_providers and "nvidia" in only_providers:
        try:
            client = _build_purpose_llm_client(
                provider="nvidia",
                model="",
                app_settings=app_settings,
            )
            if client is not None:
                target = ProbeTarget(kind="llm", provider="nvidia", model=client.model, purpose=None)
                if target not in targets:
                    targets.append(target)
        except ValueError:
            target = ProbeTarget(kind="llm", provider="nvidia", model=app_settings.llm_model, purpose=None)
            if target not in targets:
                targets.append(target)

    results: list[ProbeResult] = []
    for target in targets:
        results.append(await _probe_llm_target(target, app_settings, prompt, timeout))

    if (
        include_embeddings
        and not purpose
        and (not only_providers or app_settings.embedding_provider.lower() in only_providers)
    ):
        results.append(await _probe_embedding(app_settings, timeout, prompt))

    return results


def main(
    providers: list[str] | None = None,
    purpose: str | None = None,
    prompt: str = _DEFAULT_PROMPT,
    timeout: float = 10.0,
    json_output: bool = False,
    verbose: bool = False,
    no_embeddings: bool = False,
) -> None:
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    include_embeddings = not no_embeddings

    try:
        results = asyncio.run(
            _run_probes(
                app_settings=settings,
                providers=providers,
                purpose=purpose,
                prompt=prompt,
                timeout=timeout,
                include_embeddings=include_embeddings,
            )
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if json_output:
        print(json.dumps(_serialize(results), indent=2))
    else:
        if results:
            print(_format_table(results))
            print()
        for r in results:
            if r.status == "SKIP":
                print(f"  ⚠️  {r.target.label}: {r.message}")
            elif r.status == "FAIL":
                print(f"  ❌ {r.target.label}: {r.category or 'error'} — {r.message}")
        if verbose:
            for r in results:
                print()
                print(f"  ── {r.target.label} ──")
                for key, value in r.request_headers.items():
                    print(f"    {key}: {value}")
                if r.response_preview:
                    print(f"    response: {r.response_preview!r}")
                if r.dimension is not None:
                    print(f"    dimension: {r.dimension} (expected {r.expected_dimension})")
                if r.prompt_tokens or r.completion_tokens:
                    print(f"    tokens: {r.prompt_tokens} prompt / {r.completion_tokens} completion")

    failed = [r for r in results if r.status == "FAIL"]
    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
