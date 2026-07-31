"""Tests for request body size limit, security headers, and correlation IDs."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from data_engineering_copilot.api.middleware import (
    CorrelationIdMiddleware,
    RequestBodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
)


def _make_app():
    app = FastAPI()

    @app.post("/api/v1/echo")
    async def echo_route(request: Request):
        return {"correlation_id": getattr(request.state, "correlation_id", None), "size": len(await request.body())}

    @app.get("/api/v1/echo")
    async def echo_get(request: Request):
        return {"correlation_id": getattr(request.state, "correlation_id", None)}

    return app


class TestRequestBodySizeLimit:
    def test_accepts_small_body(self):
        app = _make_app()
        app.add_middleware(RequestBodySizeLimitMiddleware)
        client = TestClient(app)
        resp = client.post("/api/v1/echo", content=b"x" * 100)
        assert resp.status_code == 200
        assert resp.json()["size"] == 100

    def test_content_length_rejects_oversized(self):
        app = _make_app()
        app.add_middleware(RequestBodySizeLimitMiddleware, max_bytes=1024)
        client = TestClient(app)
        resp = client.post(
            "/api/v1/echo",
            content=b"x" * 2048,
            headers={"Content-Length": "2048"},
        )
        assert resp.status_code == 413

    def test_buffered_body_rejects_oversized(self):
        app = _make_app()
        app.add_middleware(RequestBodySizeLimitMiddleware, max_bytes=1024)
        client = TestClient(app)
        # No Content-Length header: body is buffered and measured.
        resp = client.post(
            "/api/v1/echo",
            content=b"x" * 2048,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 413


class TestSecurityHeaders:
    def test_headers_attached(self):
        app = _make_app()
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/api/v1/echo")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert "frame-ancestors 'none'" in resp.headers.get("Content-Security-Policy", "")

    def test_existing_header_not_overwritten(self):
        app = _make_app()
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)

        @app.get("/custom")
        async def custom_route():
            from fastapi.responses import Response

            return Response(status_code=200, headers={"X-Frame-Options": "SAMEORIGIN"})

        resp = client.get("/custom")
        assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"


class TestCorrelationId:
    def test_response_has_correlation_id_header(self):
        app = _make_app()
        app.add_middleware(CorrelationIdMiddleware)
        client = TestClient(app)
        resp = client.get("/api/v1/echo")
        assert resp.headers.get("X-Correlation-ID")

    def test_client_supplied_correlation_id_echoed(self):
        app = _make_app()
        app.add_middleware(CorrelationIdMiddleware)
        client = TestClient(app)
        resp = client.get("/api/v1/echo", headers={"X-Correlation-ID": "my-trace-42"})
        assert resp.headers.get("X-Correlation-ID") == "my-trace-42"
        assert resp.json()["correlation_id"] == "my-trace-42"
