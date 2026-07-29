import asyncio
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from data_engineering_copilot.api.auth import ApiKeyAuthMiddleware
from data_engineering_copilot.api.middleware import RateLimitMiddleware
from data_engineering_copilot.config.settings import settings

from .routes import router

app = FastAPI(
    title="DataEngineeringCopilot API",
    description="Async ingestion and RAG service endpoints",
    version="1.0.0",
)

# Rate limiting middleware: per-route (60/min for /ask, 10/min for /ingest)
app.add_middleware(RateLimitMiddleware)

# API key authentication (no-op if API_KEY env var not set)
app.add_middleware(
    ApiKeyAuthMiddleware,
    rbac_enabled=settings.rbac_enabled,
    rbac_users_json=settings.rbac_users_json,
)

# CORS — allow all origins for local development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Module-level tracker singletons (shared with factory)
_retrieval_tracker = None
_token_tracker = None


def set_trackers(retrieval_tracker=None, token_tracker=None):
    """Set tracker instances for metrics endpoint."""
    global _retrieval_tracker, _token_tracker
    _retrieval_tracker = retrieval_tracker
    _token_tracker = token_tracker


async def _check_url(url: str, timeout: float = 3.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


async def _check_tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        return True
    except (OSError, TimeoutError):
        return False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    results: dict[str, bool] = {}

    # Qdrant
    qdrant_parsed = urlparse(settings.qdrant_url)
    qdrant_host = qdrant_parsed.hostname or "localhost"
    qdrant_port = qdrant_parsed.port or 6333
    results["qdrant"] = await _check_tcp(qdrant_host, qdrant_port)

    # Ollama
    ollama_parsed = urlparse(settings.ollama_base_url)
    ollama_host = ollama_parsed.hostname or "localhost"
    ollama_port = ollama_parsed.port or 11434
    results["ollama"] = await _check_tcp(ollama_host, ollama_port)

    # Redis
    redis_parsed = urlparse(settings.redis_url)
    redis_host = redis_parsed.hostname or "localhost"
    redis_port = redis_parsed.port or 6379
    results["redis"] = await _check_tcp(redis_host, redis_port)

    all_healthy = all(results.values())
    overall = "healthy" if all_healthy else "unhealthy" if not any(results.values()) else "degraded"
    status_code = 200 if all_healthy else 503
    return JSONResponse(
        content={"status": overall, "checks": results},
        status_code=status_code,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    lines = []

    # Retrieval metrics
    if _retrieval_tracker is not None:
        dist = _retrieval_tracker.get_distribution()
        lines.append("# HELP rag_retrieval_score Retrieval score distribution")
        lines.append("# TYPE rag_retrieval_score summary")
        lines.append(f'rag_retrieval_score{{quantile="0.5"}} {dist["p50"]:.4f}')
        lines.append(f'rag_retrieval_score{{quantile="0.95"}} {dist["p95"]:.4f}')
        lines.append(f'rag_retrieval_score{{quantile="0.99"}} {dist["p99"]:.4f}')
        lines.append(f'rag_retrieval_score{{quantile="mean"}} {dist["mean"]:.4f}')
        lines.append("")
        lines.append("# HELP rag_retrieval_queries_total Total retrieval queries")
        lines.append("# TYPE rag_retrieval_queries_total counter")
        lines.append(f"rag_retrieval_queries_total {dist['queries']}")
        lines.append("")

    # Token usage metrics
    if _token_tracker is not None:
        usage = _token_tracker.get_usage()
        lines.append("# HELP rag_token_usage_total Total LLM tokens used")
        lines.append("# TYPE rag_token_usage_total counter")
        lines.append(f'rag_token_usage_total{{type="prompt"}} {usage.total_prompt_tokens}')
        lines.append(f'rag_token_usage_total{{type="completion"}} {usage.total_completion_tokens}')
        lines.append("")
        lines.append("# HELP rag_llm_calls_total Total LLM calls")
        lines.append("# TYPE rag_llm_calls_total counter")
        lines.append(f"rag_llm_calls_total {usage.total_calls}")

    return Response(content="\n".join(lines), media_type="text/plain")
