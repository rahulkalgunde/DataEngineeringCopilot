import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from data_engineering_copilot.api.middleware import RateLimitMiddleware
from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.services.health_check import HealthChecker

from .routes import router

app = FastAPI(
    title="DataEngineeringCopilot API",
    description="Async ingestion and RAG service endpoints",
    version="1.0.0",
)

# Rate limiting middleware: per-route (60/min for /ask, 10/min for /ingest)
app.add_middleware(RateLimitMiddleware)

app.include_router(router)


def _check_url(url: str, timeout: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _check_tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    checker = HealthChecker()

    # Qdrant
    qdrant_parsed = urlparse(settings.qdrant_url)
    qdrant_host = qdrant_parsed.hostname or "localhost"
    qdrant_port = qdrant_parsed.port or 6333
    checker.register("qdrant", lambda: _check_tcp(qdrant_host, qdrant_port))

    # Ollama
    ollama_parsed = urlparse(settings.ollama_base_url)
    ollama_host = ollama_parsed.hostname or "localhost"
    ollama_port = ollama_parsed.port or 11434
    checker.register("ollama", lambda: _check_tcp(ollama_host, ollama_port))

    # Redis
    redis_parsed = urlparse(settings.redis_url)
    redis_host = redis_parsed.hostname or "localhost"
    redis_port = redis_parsed.port or 6379
    checker.register("redis", lambda: _check_tcp(redis_host, redis_port))

    status = checker.check()
    status_code = 200 if status.overall == "healthy" else 503
    return JSONResponse(
        content={"status": status.overall, "checks": status.services},
        status_code=status_code,
    )
