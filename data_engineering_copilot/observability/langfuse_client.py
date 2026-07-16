import logging
import urllib.request
import urllib.error
import json

from data_engineering_copilot.config.settings import settings

logger = logging.getLogger(__name__)


def _check_langfuse_health(host: str, timeout: int = 5) -> bool:
    """
    Verify the Langfuse server is reachable and healthy by calling GET /api/public/health.
    Returns True if the server responds with a healthy status, False otherwise.
    """
    health_url = f"{host.rstrip('/')}/api/public/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                status = data.get("status", "")
                if status == "OK":
                    logger.info("Langfuse health check passed: %s", health_url)
                    return True
                else:
                    logger.warning(
                        "Langfuse health check returned non-OK status: %s (status=%r)",
                        health_url,
                        status,
                    )
                    return False
            else:
                logger.warning(
                    "Langfuse health check returned HTTP %d: %s",
                    resp.status,
                    health_url,
                )
                return False
    except urllib.error.URLError as e:
        logger.warning("Langfuse health check failed — server unreachable at %s: %s", health_url, e)
        return False
    except Exception as e:
        logger.warning("Langfuse health check failed — unexpected error at %s: %s", health_url, e)
        return False


def get_langfuse_instance():
    """
    Create a Langfuse v3 client using centralized settings.
    Uses lazy import to avoid failing when the langfuse package is not installed.
    Validates connectivity with a health check before returning the client.
    Returns None if langfuse is unavailable or cannot be initialized.
    """
    try:
        from langfuse import Langfuse

        host = settings.langfuse_host
        logger.info("Langfuse host: %s", host)

        if not _check_langfuse_health(host):
            logger.warning(
                "Langfuse server is not healthy at %s; traces will not be exported. "
                "Ensure the Langfuse container and its dependencies (ClickHouse, Postgres, MinIO) are running.",
                host,
            )

        lf = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=host,
            debug=True
        )
        print(lf.auth_check())
        logger.info("Langfuse client initialized successfully (host=%s)", host)
        return lf
    except Exception as e:
        logger.info("Langfuse client not available: %s", e)
        return None


# Helper functions for tracing (v3 API)
def start_trace(name: str, **kwargs):
    """Start a Langfuse v3 trace observation."""
    lf = get_langfuse_instance()
    if lf:
        return lf.start_observation(name=name, as_type="trace", **kwargs)
    return None


def end_trace(trace, **kwargs):
    """End a Langfuse v3 trace observation."""
    if trace:
        trace.update(**kwargs)
        trace.end()


def log_event(trace, name: str, **kwargs):
    """Log a span event within a Langfuse v3 trace."""
    if trace:
        span = trace.start_observation(name=name, as_type="span", **kwargs)
        span.end()