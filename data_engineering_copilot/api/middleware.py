"""ASGI rate limiting and prompt injection detection middleware."""

from __future__ import annotations

import json
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from data_engineering_copilot.services.prompt_injection import INJECTION_THRESHOLD, detect_prompt_injection
from data_engineering_copilot.services.rate_limiter import DEFAULT_LIMITS, RateLimiter

logger = logging.getLogger(__name__)

# Backwards-compatible alias for tests and any external callers.
_detect_prompt_injection = detect_prompt_injection
_INJECTION_THRESHOLD = INJECTION_THRESHOLD


def _client_ip(request: Request) -> str:
    """Extract client IP from the direct connection."""
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _rate_limit_key(request: Request) -> str:
    """Derive a rate-limit key from API key or client IP.

    API keys get their own quota; unauthenticated requests are grouped by IP.
    """
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ")
    if api_key:
        return api_key[:48]
    return _client_ip(request)


def _attach_correlation_id(request: Request) -> str:
    """Generate and bind a per-request correlation ID.

    The ID is stored on ``request.state.correlation_id`` for the route layer,
    bound into the structlog context so all logs emitted during the request
    carry it, and echoed back on the ``X-Correlation-ID`` response header.
    """
    correlation_id = request.headers.get("X-Correlation-ID", "") or str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    try:
        import structlog

        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    except Exception:
        pass
    return correlation_id


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID and W3C trace context to every request.

    This is registered outermost so even error responses from inner
    middlewares (rate limit, auth, injection) carry ``X-Correlation-ID``.
    """

    async def dispatch(self, request: Request, call_next):
        correlation_id = _attach_correlation_id(request)
        _attach_trace_context(request)
        try:
            response = await call_next(request)
        finally:
            _detach_trace_context(request)
        return _finalize_response(response, correlation_id)


def _attach_trace_context(request: Request) -> None:
    """Attach the W3C trace context from incoming headers.

    When the caller sends ``traceparent``/``tracestate``, the extracted OTel
    context is attached for the duration of the request so spans created by
    the RAG pipeline continue the upstream trace.
    """
    try:
        from opentelemetry import context as otel_context

        from data_engineering_copilot.observability.otel_telemetry import extract_w3c_context

        trace_ctx = extract_w3c_context(dict(request.headers))
        request.state.trace_context = trace_ctx
        if trace_ctx is not None:
            request.state._otel_token = otel_context.attach(trace_ctx)
    except Exception:
        request.state.trace_context = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that applies per-path rate limiting.

    Uses API key as the rate-limit identifier when available,
    falling back to client IP for unauthenticated requests.
    Only enforces limits for paths listed in ``DEFAULT_LIMITS``.
    Other paths pass through without rate limiting.
    """

    def __init__(self, app, **kwargs) -> None:
        super().__init__(app)
        self._limiters: dict[str, RateLimiter] = {}
        for path, (max_calls, period) in DEFAULT_LIMITS.items():
            self._limiters[path] = RateLimiter(path=path, max_calls=max_calls, period_seconds=period)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        correlation_id = getattr(request.state, "correlation_id", "")

        response: Response | None = None

        # Prompt injection detection for /ask endpoints
        if path in ("/api/v1/ask", "/api/v1/ask/stream") and request.method == "POST":
            body_bytes = await request.body()
            try:
                body = json.loads(body_bytes)
                question = body.get("question", "")
                if isinstance(question, str) and question.strip():
                    injection_score = _detect_prompt_injection(question)
                    if injection_score >= _INJECTION_THRESHOLD:
                        logger.warning(
                            "prompt_injection_detected path=%s score=%.2f question=%r correlation_id=%s",
                            path,
                            injection_score,
                            question[:80],
                            correlation_id,
                        )
                        response = _json_rejected(correlation_id)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Malformed JSON bypasses the structured check above. Fall back
                # to scanning the raw body so the injection guard still applies.
                try:
                    raw_text = body_bytes.decode("utf-8", errors="replace")[:4000]
                    injection_score = _detect_prompt_injection(raw_text)
                    if injection_score >= _INJECTION_THRESHOLD:
                        logger.warning(
                            "prompt_injection_detected_raw path=%s score=%.2f correlation_id=%s",
                            path,
                            injection_score,
                            correlation_id,
                        )
                        response = _json_rejected(correlation_id)
                except Exception:
                    logger.warning(
                        "Unable to scan raw request body for prompt injection path=%s correlation_id=%s",
                        path,
                        correlation_id,
                    )

        if response is None:
            limiter = self._limiters.get(path)
            if limiter is not None:
                rl_key = _rate_limit_key(request)
                if not await limiter.allow_async(rl_key):
                    logger.warning(
                        "rate_limit_exceeded path=%s key=%s correlation_id=%s",
                        path,
                        rl_key,
                        correlation_id,
                    )
                    period = getattr(limiter, "_period_seconds", 60)
                    response = _json_limited(period, correlation_id)

        if response is None:
            response = await call_next(request)

        return _finalize_response(response, correlation_id)


def _json_rejected(correlation_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Question rejected: detected potential prompt injection. "
            "Rephrase your question without system-level instructions."
        },
        headers={"X-Correlation-ID": correlation_id},
    )


def _json_limited(period: int, correlation_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
        headers={"Retry-After": str(period), "X-Correlation-ID": correlation_id},
    )


def _detach_trace_context(request: Request) -> None:
    """Detach the request-local OTel context token after the response."""
    import contextlib

    with contextlib.suppress(Exception):
        from opentelemetry import context as otel_context

        token = getattr(request.state, "_otel_token", None)
        if token is not None:
            otel_context.detach(token)
            request.state._otel_token = None


def _finalize_response(response: Response, correlation_id: str) -> Response:
    import contextlib

    with contextlib.suppress(Exception):
        response.headers.setdefault("X-Correlation-ID", correlation_id)
    return response


# Maximum request body size in bytes (protects the API from oversized payloads).
MAX_REQUEST_BODY_BYTES = 1_048_576


class RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds ``MAX_REQUEST_BODY_BYTES``.

    Content-Length is checked first (cheap); chunked bodies are buffered and
    the size verified. The buffered body is cached back on the request so
    downstream handlers can still read it.
    """

    def __init__(self, app, max_bytes: int = MAX_REQUEST_BODY_BYTES, **kwargs) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > self._max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body exceeds the {self._max_bytes}-byte limit."},
            )
        body = await request.body()
        if len(body) > self._max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body exceeds the {self._max_bytes}-byte limit."},
            )
        # Cache the body so downstream handlers can still read it after the
        # outer middleware consumed the stream.
        request._body = body
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach security hardening response headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response
