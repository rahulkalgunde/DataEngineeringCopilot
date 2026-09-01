"""Deep health-check module — single seam for Docker/Qdrant/Redis/Provider chains.

Design: small interface (HealthReport + probe_* functions), large implementation
hidden behind it. Both ``dec health`` (fast gate) and ``dec status`` (verbose
inventory) call this module so Docker/Qdrant/Redis probes are not duplicated.
Two adapters exist (real probes vs test fakes) so the seam is real.

Fail-open: a probe that cannot reach a service returns unhealthy rather than
raising; callers decide exit codes.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

from data_engineering_copilot.config.settings import AppSettings


@dataclass
class DockerHealth:
    ok: bool
    detail: str
    git_sha: str = ""
    deps_fingerprint_ok: bool | None = None


@dataclass
class QdrantHealth:
    ok: bool
    detail: str
    status_code: int | None = None


@dataclass
class RedisHealth:
    ok: bool
    detail: str
    key_count: int | None = None


@dataclass
class ProviderChainEntry:
    provider: str
    model: str
    has_key: bool
    dimension: int | None = None


@dataclass
class ProviderChainHealth:
    provider: str
    model: str
    has_key: bool
    is_local: bool  # local-hf / ollama have no API key


@dataclass
class HealthReport:
    docker: DockerHealth
    qdrant: QdrantHealth
    redis: RedisHealth
    embedding_chain: list[ProviderChainHealth] = field(default_factory=list)
    llm_chain: list[ProviderChainHealth] = field(default_factory=list)
    # Active alias target for status verbose mode
    qdrant_alias_target: str = ""
    qdrant_points: int | None = None

    @property
    def all_healthy(self) -> bool:
        if not self.qdrant.ok or not self.redis.ok:
            return False
        # Docker not reachable is not a hard failure for local dev, but
        # stale deps is.
        return self.docker.deps_fingerprint_ok is not False


def _provider_has_key(settings: AppSettings, provider: str) -> tuple[bool, bool]:
    """Return (has_key, is_local) for embedding/LLM provider."""
    provider = provider.lower()
    if provider in ("local-hf", "ollama"):
        return True, True
    # Map provider -> settings attr for API key
    key_map: dict[str, str] = {
        "nvidia": "nvidia_api_key",
        "openrouter": "openrouter_api_key",
        "huggingface": "huggingface_api_key",
        "gemini": "gemini_api_key",
        "groq": "groq_api_key",
        "cerebras": "cerebras_api_key",
        "cloudflare": "cloudflare_api_key",
        "opencodezen": "opencodezen_api_key",
        "opencodego": "opencodego_api_key",
        "sambanova": "sambanova_api_key",
        "mistral": "mistral_api_key",
        "deepseek": "deepseek_api_key",
        "zai": "zai_api_key",
        "siliconflow": "siliconflow_api_key",
        "together": "together_api_key",
        "fireworks": "fireworks_api_key",
        "llm7": "llm7_api_key",
        "agnes": "agnes_api_key",
        "ollama_cloud": "ollama_cloud_api_key",
        "helyx": "helyx_api_key",
        "anyapi": "anyapi_api_key",
    }
    attr = key_map.get(provider)
    if attr is None:
        return False, False
    val = getattr(settings, attr, None)
    if val is None:
        return False, False
    # SecretStr vs str
    try:
        secret = val.get_secret_value() if hasattr(val, "get_secret_value") else str(val)
    except Exception:
        secret = str(val)
    return bool(secret and str(secret).strip()), False


def _embedding_model_for(settings: AppSettings, provider: str) -> str:
    mapping = {
        "nvidia": settings.nvidia_embedding_model,
        "openrouter": settings.openrouter_embedding_model,
        "huggingface": settings.huggingface_embedding_model,
        "gemini": settings.gemini_embedding_model if hasattr(settings, "gemini_embedding_model") else "",
        "local-hf": settings.local_hf_embedding_model,
        "ollama": settings.ollama_model,
    }
    return mapping.get(provider.lower(), "")


def _llm_model_for(settings: AppSettings, provider: str) -> str:
    mapping = {
        "groq": settings.groq_model,
        "cerebras": settings.cerebras_model,
        "nvidia": settings.nvidia_model,
        "cloudflare": settings.cloudflare_model,
        "openrouter": settings.openrouter_model,
        "gemini": settings.gemini_model,
        "agnes": settings.agnes_model,
        "ollama_cloud": settings.ollama_cloud_model,
        "ollama": settings.ollama_model,
        "opencodezen": settings.opencodezen_model,
        "opencodego": settings.opencodego_model,
        "sambanova": settings.sambanova_model,
        "mistral": settings.mistral_model,
        "deepseek": settings.deepseek_model,
        "zai": settings.zai_model,
        "siliconflow": settings.siliconflow_model,
        "together": settings.together_model,
        "fireworks": settings.fireworks_model,
        "llm7": settings.llm7_model,
        "helyx": settings.helyx_model,
        "anyapi": settings.anyapi_model,
    }
    return mapping.get(provider.lower(), "")


def probe_docker(timeout: float = 5.0) -> DockerHealth:
    try:
        req = urllib.request.Request("http://localhost:8000/api/v1/version")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            deps_ok = data.get("deps_fingerprint_ok")
            git_sha = data.get("git_sha", "") or ""
            if deps_ok is True:
                return DockerHealth(ok=True, detail="fresh", git_sha=git_sha, deps_fingerprint_ok=True)
            if deps_ok is False:
                return DockerHealth(
                    ok=False,
                    detail="STALE — run `make docker-dev`",
                    git_sha=git_sha,
                    deps_fingerprint_ok=False,
                )
            return DockerHealth(ok=True, detail="unknown (not running in Docker)", git_sha=git_sha)
    except Exception as exc:
        return DockerHealth(ok=False, detail=f"API not reachable at localhost:8000 ({exc})")


def probe_qdrant(settings: AppSettings, timeout: float = 3.0) -> QdrantHealth:
    try:
        req = urllib.request.Request(f"{settings.qdrant_url}/", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return QdrantHealth(ok=True, detail="200 OK", status_code=200)
            return QdrantHealth(ok=False, detail=f"status {resp.status}", status_code=resp.status)
    except Exception as exc:
        return QdrantHealth(ok=False, detail=f"Unreachable: {exc}")


def probe_qdrant_collection(settings: AppSettings, timeout: float = 10.0) -> tuple[QdrantHealth, str, int | None]:
    """Probe the active Qdrant collection via HTTP (for status verbose).

    Uses a longer timeout than the health liveness probe because a 70k
    collection under merge can take >5s. Falls back to QdrantClient if HTTP
    times out.
    """
    name = settings.collection_name
    # Prefer alias if active generation set
    target = settings.active_collection_name or name
    # Try HTTP first
    try:
        req = urllib.request.Request(f"{settings.qdrant_url}/collections/{target}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            if "result" in data:
                result = data["result"]
                points = result.get("points_count") or result.get("vectors_count")
                return (
                    QdrantHealth(ok=True, detail=f"Collection {target} {result.get('status', '')}", status_code=200),
                    target,
                    points,
                )
            return QdrantHealth(ok=False, detail="Collection not found"), target, None
    except Exception:
        pass
    # Fallback to QdrantClient (more resilient under load)
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=int(timeout))
        # Try alias resolution
        try:
            info = client.get_collection(target)
            pts = getattr(info, "points_count", None)
            return QdrantHealth(ok=True, detail=f"Collection {target} {info.status}", status_code=200), target, pts
        except Exception as exc:
            return QdrantHealth(ok=False, detail=str(exc)), target, None
    except Exception as exc:
        return QdrantHealth(ok=False, detail=f"Unreachable: {exc}"), target, None


def probe_redis(settings: AppSettings, timeout: float = 3.0) -> RedisHealth:
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_timeout=timeout)
        ok = client.ping()
        if not ok:
            return RedisHealth(ok=False, detail="no PONG")
        try:
            info = client.info()
            keys = info.get("db0", {}).get("keys", 0) if "db0" in info else 0
        except Exception:
            keys = None
        client.close()
        return RedisHealth(ok=True, detail="PONG", key_count=keys)
    except Exception as exc:
        return RedisHealth(ok=False, detail=f"Unreachable: {exc}")


def probe_embedding_chain(settings: AppSettings) -> list[ProviderChainHealth]:
    chain: list[ProviderChainHealth] = []
    for prov in settings.embedding_fallback_order:
        model = _embedding_model_for(settings, prov)
        has_key, is_local = _provider_has_key(settings, prov)
        chain.append(ProviderChainHealth(provider=prov, model=model, has_key=has_key, is_local=is_local))
    # Also surface the primary provider if not in fallback (shouldn't happen)
    primary = settings.embedding_provider.lower()
    if primary not in [c.provider.lower() for c in chain]:
        model = _embedding_model_for(settings, primary)
        has_key, is_local = _provider_has_key(settings, primary)
        chain.insert(0, ProviderChainHealth(provider=primary, model=model, has_key=has_key, is_local=is_local))
    return chain


def probe_llm_chain(settings: AppSettings) -> list[ProviderChainHealth]:
    chain: list[ProviderChainHealth] = []
    for prov in settings.llm_fallback_order:
        model = _llm_model_for(settings, prov)
        has_key, is_local = _provider_has_key(settings, prov)
        chain.append(ProviderChainHealth(provider=prov, model=model, has_key=has_key, is_local=is_local))
    primary = settings.llm_provider.lower()
    if primary not in [c.provider.lower() for c in chain]:
        model = _llm_model_for(settings, primary)
        has_key, is_local = _provider_has_key(settings, primary)
        chain.insert(0, ProviderChainHealth(provider=primary, model=model, has_key=has_key, is_local=is_local))
    return chain


def build_health_report(settings: AppSettings) -> HealthReport:
    docker = probe_docker()
    qdrant = probe_qdrant(settings)
    redis_h = probe_redis(settings)
    emb = probe_embedding_chain(settings)
    llm = probe_llm_chain(settings)
    return HealthReport(docker=docker, qdrant=qdrant, redis=redis_h, embedding_chain=emb, llm_chain=llm)


def build_status_report(settings: AppSettings) -> HealthReport:
    docker = probe_docker()
    qdrant_h, alias_target, points = probe_qdrant_collection(settings, timeout=10.0)
    redis_h = probe_redis(settings)
    emb = probe_embedding_chain(settings)
    llm = probe_llm_chain(settings)
    return HealthReport(
        docker=docker,
        qdrant=qdrant_h,
        redis=redis_h,
        embedding_chain=emb,
        llm_chain=llm,
        qdrant_alias_target=alias_target,
        qdrant_points=points,
    )
