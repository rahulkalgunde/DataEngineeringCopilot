import json
import logging
import random
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from opentelemetry.trace import format_span_id, format_trace_id

from data_engineering_copilot.config.settings import settings

logger = logging.getLogger(__name__)

# Trace-level attributes that must NOT be forwarded to v4 start_observation()
# (it has no **kwargs and rejects unknown params). These are applied via
# langfuse.propagate_attributes so every observation in the trace inherits them.
_V4_TRACE_LEVEL_KWARGS = {
    "user_id",
    "session_id",
    "tags",
    "environment",
    "release",
    "trace_name",
    "metadata",
}

# Kwargs v4 start_observation() actually accepts.
_V4_OBSERVATION_KWARGS = {
    "name",
    "as_type",
    "input",
    "output",
    "metadata",
    "version",
    "level",
    "status_message",
    "completion_start_time",
    "model",
    "model_parameters",
    "usage_details",
    "cost_details",
    "prompt",
}


def _derive_span_id(observation: Any) -> str | None:
    """Derive the 16-hex Langfuse observation id from a v4 span's OTel context."""
    if observation is None:
        return None
    otel_span = getattr(observation, "_otel_span", None)
    if otel_span is not None and otel_span.context is not None:
        try:
            return format_span_id(otel_span.context.span_id)
        except Exception:
            pass
    return getattr(observation, "id", None)


def _derive_trace_id(observation: Any) -> str | None:
    """Derive the 32-hex Langfuse trace id from a v4 span's OTel context."""
    if observation is None:
        return None
    otel_span = getattr(observation, "_otel_span", None)
    if otel_span is not None and otel_span.context is not None:
        try:
            return format_trace_id(otel_span.context.trace_id)
        except Exception:
            pass
    return getattr(observation, "trace_id", None)


def _enter_propagate(trace_attrs: dict[str, Any] | None):
    """Return a propagate_attributes context manager (or nullcontext) for trace-level attrs.

    ``release`` is excluded: ``propagate_attributes`` does not accept it (it is a
    client-level ``Langfuse(release=...)`` attribute that applies to all spans).
    """
    if trace_attrs:
        from contextlib import nullcontext

        from langfuse import propagate_attributes

        attrs = {k: v for k, v in trace_attrs.items() if k != "release"}
        try:
            return propagate_attributes(**attrs)
        except Exception as exc:
            logger.warning("propagate_attributes failed (%s); proceeding without trace attrs", exc)
            return nullcontext()
    from contextlib import nullcontext

    return nullcontext()


class _ObservationCompat:
    """Compatibility wrapper for Langfuse v2/v3-style observation objects."""

    def __init__(
        self,
        client,
        observation,
        kind: str,
        trace_id=None,
        parent_observation_id=None,
        trace_attrs: dict[str, Any] | None = None,
    ):
        self._client = client
        self._observation = observation
        self.kind = kind
        self._parent_trace_id = trace_id
        self.parent_observation_id = parent_observation_id
        self._trace_attrs = trace_attrs or {}

    @property
    def id(self) -> Any:
        if self._observation is None:
            return None
        return _derive_span_id(self._observation)

    @property
    def trace_id(self) -> Any:
        if self._observation is None:
            return self._parent_trace_id
        return _derive_trace_id(self._observation) or self._parent_trace_id

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

    def score(self, name: str, value: float | str | bool, data_type: str = "NUMERIC", **kwargs):
        """Score this observation (trace, span, or generation)."""
        if self._observation is None:
            return self

        # v4: span objects expose .score() (this observation) and .score_trace() (the trace).
        if hasattr(self._observation, "score"):
            try:
                if self.kind == "trace":
                    method = getattr(self._observation, "score_trace", None)
                    if method is not None:
                        method(name=name, value=value, data_type=data_type, **kwargs)
                        return self
                self._observation.score(name=name, value=value, data_type=data_type, **kwargs)
                return self
            except Exception:
                pass

        # Try to use the native Langfuse score method if available
        if hasattr(self._client._client, "create_score"):
            try:
                # For Langfuse v4, create_score takes trace_id + optional observation_id
                trace_id = self.trace_id or self._parent_trace_id or self.id
                self._client._client.create_score(
                    trace_id=trace_id,
                    observation_id=self.id,
                    name=name,
                    value=value,
                    data_type=data_type,
                    **kwargs,
                )
            except Exception:
                # Fallback to using the observation's score method if available
                if hasattr(self._observation, "score"):
                    self._observation.score(name=name, value=value, data_type=data_type, **kwargs)
        elif hasattr(self._client._client, "score"):
            try:
                # For Langfuse v2/v3, create a score on the trace
                if self.kind == "trace" or self._parent_trace_id is not None:
                    trace_id = self._parent_trace_id or self.id
                    self._client._client.score(
                        trace_id=trace_id,
                        name=name,
                        value=value,
                        data_type=data_type,
                        **kwargs,
                    )
                else:
                    observation_id = self.id
                    if observation_id:
                        self._client._client.score(
                            trace_id=self._parent_trace_id,
                            observation_id=observation_id,
                            name=name,
                            value=value,
                            data_type=data_type,
                            **kwargs,
                        )
            except Exception:
                # Fallback to using the observation's score method if available
                if hasattr(self._observation, "score"):
                    self._observation.score(name=name, value=value, data_type=data_type, **kwargs)
        elif hasattr(self._observation, "score"):
            # Fallback to observation's score method
            self._observation.score(name=name, value=value, data_type=data_type, **kwargs)
        return self

    def start_observation(self, name: str, **kwargs):
        as_type = kwargs.pop("as_type", "span")

        # v4 path: delegate to the wrapped v4 observation (auto-parents via OTel context).
        if self._observation is not None and hasattr(self._observation, "start_observation"):
            obs_kwargs = {k: v for k, v in kwargs.items() if k in _V4_OBSERVATION_KWARGS}
            with _enter_propagate(self._trace_attrs):
                child_observation = self._observation.start_observation(name=name, as_type=as_type, **obs_kwargs)
            return _ObservationCompat(
                self._client,
                child_observation,
                kind=as_type,
                trace_id=self.trace_id,
                parent_observation_id=self.id,
                trace_attrs=self._trace_attrs,
            )

        child_kwargs = dict(kwargs)
        if self.kind == "trace" or self._parent_trace_id is not None:
            child_kwargs.setdefault("trace_id", self._parent_trace_id or self.id)
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
            trace_attrs=self._trace_attrs,
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
    """Adapter for Langfuse clients that expose either v3/v4 start_observation or v2 trace/span/generation APIs."""

    def __init__(self, client):
        self._client = client

    def start_observation(self, name: str, **kwargs):
        as_type = kwargs.pop("as_type", "trace")

        if hasattr(self._client, "start_observation"):
            # v4 path: as_type="trace" is invalid (root spans are type "span"); trace-level
            # attributes go through propagate_attributes so all observations inherit them.
            trace_attrs = {k: kwargs.pop(k) for k in list(kwargs) if k in _V4_TRACE_LEVEL_KWARGS}
            obs_kwargs = {k: v for k, v in kwargs.items() if k in _V4_OBSERVATION_KWARGS}
            v4_type = "span" if as_type == "trace" else as_type
            with _enter_propagate(trace_attrs):
                observation = self._client.start_observation(name=name, as_type=v4_type, **obs_kwargs)
            return _ObservationCompat(
                self,
                observation,
                kind=as_type,
                trace_id=_derive_trace_id(observation),
                parent_observation_id=None,
                trace_attrs=trace_attrs,
            )

        method_name = {"trace": "trace", "span": "span", "generation": "generation"}.get(as_type, "trace")
        method = getattr(self._client, method_name)
        observation = method(name=name, **kwargs)
        return _ObservationCompat(self, observation, kind=as_type, trace_id=None, parent_observation_id=None)

    def trace(self, name: str | None = None, **kwargs):
        if name is not None:
            kwargs["name"] = name
        return self.start_observation(as_type="trace", **kwargs)

    def span(self, name: str | None = None, **kwargs):
        if name is not None:
            kwargs["name"] = name
        return self.start_observation(as_type="span", **kwargs)

    def generation(self, name: str | None = None, **kwargs):
        if name is not None:
            kwargs["name"] = name
        return self.start_observation(as_type="generation", **kwargs)

    def auth_check(self):
        return self._client.auth_check()

    def flush(self):
        if hasattr(self._client, "flush"):
            self._client.flush()

    def score(self, trace_id: str, name: str, value: float | str | bool, data_type: str = "NUMERIC", **kwargs):
        """Score a trace directly."""
        if hasattr(self._client, "create_score"):
            try:
                self._client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    data_type=data_type,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning("Failed to score trace %s: %s", trace_id, exc)
        elif hasattr(self._client, "score"):
            try:
                self._client.score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    data_type=data_type,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning("Failed to score trace %s: %s", trace_id, exc)
        return self

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
    Create a Langfuse client using centralized settings.
    Uses lazy import to avoid failing when the langfuse package is not installed.
    Tries the configured host first and then localhost/127.0.0.1 fallbacks when
    the current process cannot resolve the Docker service name.
    Returns None if langfuse is unavailable or cannot be initialized.
    """
    if not settings.langfuse_enabled:
        logger.info("Langfuse disabled via LANGFUSE_ENABLED=false")
        return None
    if random.random() >= settings.langfuse_sample_rate:
        logger.debug("Langfuse trace sampled out (sample rate %s)", settings.langfuse_sample_rate)
        return None
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
                    public_key=settings.langfuse_public_key.get_secret_value(),
                    secret_key=settings.langfuse_secret_key.get_secret_value(),
                    host=candidate_host,
                    release=settings.image_git_sha,
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
