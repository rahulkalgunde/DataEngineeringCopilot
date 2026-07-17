import logging
import urllib.request
import urllib.error
import json
from urllib.parse import urlsplit, urlunsplit

from data_engineering_copilot.config.settings import settings

logger = logging.getLogger(__name__)


class _ObservationCompat:
    """Compatibility wrapper for Langfuse v2/v3-style observation objects."""

    def __init__(self, client, observation, kind: str, trace_id=None, parent_observation_id=None):
        self._client = client
        self._observation = observation
        self.kind = kind
        self.trace_id = trace_id
        self.parent_observation_id = parent_observation_id
        self.id = getattr(observation, "id", None)

    def update(self, **kwargs):
        if self._observation is None:
            return self
        if hasattr(self._observation, "update"):
            self._observation.update(**kwargs)
        return self

    def end(self):
        if self._observation is not None and hasattr(self._observation, "end"):
            self._observation.end()
        return self

    def log_event(self, name: str, payload=None, **kwargs):
        if self._observation is None:
            return self

        if payload is not None:
            if isinstance(payload, dict) and not kwargs:
                kwargs = dict(payload)
            else:
                kwargs.setdefault("input", payload)

        if hasattr(self._observation, "log_event"):
            self._observation.log_event(name=name, **kwargs)
        elif hasattr(self._observation, "create_event"):
            self._observation.create_event(name=name, **kwargs)
        else:
            span = self.start_observation(name=name, as_type="span", **kwargs)
            span.end()
        return self

    def start_observation(self, name: str, **kwargs):
        as_type = kwargs.pop("as_type", "span")
        child_kwargs = dict(kwargs)

        if self.kind == "trace" or self.trace_id is not None:
            child_kwargs.setdefault("trace_id", self.trace_id or self.id)
        if self.kind != "trace" and self.id is not None:
            child_kwargs.setdefault("parent_observation_id", self.id)

        if hasattr(self._client._client, "start_observation"):
            child_observation = self._client._client.start_observation(
                name=name,
                as_type=as_type,
                **child_kwargs,
            )
        else:
            method_name = {"trace": "trace", "span": "span", "generation": "generation"}.get(as_type, "span")
            method = getattr(self._client._client, method_name)
            child_observation = method(name=name, **child_kwargs)

        return _ObservationCompat(
            self._client,
            child_observation,
            kind=as_type,
            trace_id=child_kwargs.get("trace_id"),
            parent_observation_id=child_kwargs.get("parent_observation_id"),
        )

    def trace(self, name: str, **kwargs):
        return self.start_observation(name=name, as_type="trace", **kwargs)

    def span(self, name: str, **kwargs):
        return self.start_observation(name=name, as_type="span", **kwargs)

    def generation(self, name: str, **kwargs):
        return self.start_observation(name=name, as_type="generation", **kwargs)

    def __getattr__(self, name):
        return getattr(self._observation, name)


class LangfuseCompat:
    """Adapter for Langfuse clients that expose either v3 start_observation or v2 trace/span/generation APIs."""

    def __init__(self, client):
        self._client = client

    def start_observation(self, name: str, **kwargs):
        as_type = kwargs.pop("as_type", "trace")
        if hasattr(self._client, "start_observation"):
            observation = self._client.start_observation(name=name, as_type=as_type, **kwargs)
            return _ObservationCompat(self, observation, kind=as_type, trace_id=None, parent_observation_id=None)

        method_name = {"trace": "trace", "span": "span", "generation": "generation"}.get(as_type, "trace")
        method = getattr(self._client, method_name)
        observation = method(name=name, **kwargs)
        return _ObservationCompat(self, observation, kind=as_type, trace_id=None, parent_observation_id=None)

    def trace(self, **kwargs):
        return self.start_observation(as_type="trace", **kwargs)

    def span(self, **kwargs):
        return self.start_observation(as_type="span", **kwargs)

    def generation(self, **kwargs):
        return self.start_observation(as_type="generation", **kwargs)

    def auth_check(self):
        return self._client.auth_check()

    def flush(self):
        if hasattr(self._client, "flush"):
            self._client.flush()

    def __getattr__(self, name):
        return getattr(self._client, name)


def _candidate_langfuse_hosts(host: str) -> list[str]:
    """Return a list of host candidates that should be tried for Langfuse auth."""
    if not host:
        return []

    cleaned_host = host.strip().rstrip("/")
    if not cleaned_host:
        return []

    if cleaned_host.startswith(("http://", "https://")):
        parsed = urlsplit(cleaned_host)
    else:
        parsed = urlsplit(f"http://{cleaned_host}")

    candidates: list[str] = []

    def add_candidate(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add_candidate(cleaned_host)

    if parsed.hostname and parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
        for fallback_host in ("localhost", "127.0.0.1"):
            netloc = fallback_host
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            add_candidate(urlunsplit((parsed.scheme or "http", netloc, "", "", "")))
    elif parsed.hostname in {"localhost", "127.0.0.1"}:
        fallback_host = "127.0.0.1" if parsed.hostname == "localhost" else "localhost"
        netloc = fallback_host
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        add_candidate(urlunsplit((parsed.scheme or "http", netloc, "", "", "")))

    return candidates


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
    Tries the configured host first and then localhost/127.0.0.1 fallbacks when
    the current process cannot resolve the Docker service name.
    Returns None if langfuse is unavailable or cannot be initialized.
    """
    try:
        from langfuse import Langfuse

        host = settings.langfuse_host
        logger.info("Langfuse host: %s", host)

        candidate_hosts = _candidate_langfuse_hosts(host)
        for candidate_host in candidate_hosts:
            if not _check_langfuse_health(candidate_host):
                logger.debug("Langfuse health check failed for %s", candidate_host)
                continue

            try:
                lf = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=candidate_host,
                    debug=True,
                )
                auth_ok = lf.auth_check()
                logger.info("Langfuse auth check succeeded for %s: %s", candidate_host, auth_ok)
                if auth_ok:
                    return LangfuseCompat(lf)
                logger.warning("Langfuse auth check returned False for %s", candidate_host)
            except Exception as exc:
                logger.warning("Langfuse client initialization failed for %s: %s", candidate_host, exc)

        logger.warning(
            "Langfuse server could not be initialized from any candidate host %s; traces will not be exported. "
            "Ensure the Langfuse container and its dependencies are running.",
            candidate_hosts,
        )
        return None
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