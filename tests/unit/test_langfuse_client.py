import pytest

from data_engineering_copilot.observability import langfuse_client as langfuse_client_module


class _FakeObservation:
    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.kwargs = kwargs
        self.updated = {}
        self.ended = False
        self.log_events = []

    def update(self, **kwargs):
        self.updated.update(kwargs)
        return self

    def end(self):
        self.ended = True
        return self

    def log_event(self, name, **kwargs):
        self.log_events.append((name, kwargs))
        return self


class _FakeLangfuseClient:
    def __init__(self):
        self.trace_calls = []
        self.span_calls = []
        self.generation_calls = []

    def trace(self, **kwargs):
        self.trace_calls.append(kwargs)
        return _FakeObservation("trace", **kwargs)

    def span(self, **kwargs):
        self.span_calls.append(kwargs)
        return _FakeObservation("span", **kwargs)

    def generation(self, **kwargs):
        self.generation_calls.append(kwargs)
        return _FakeObservation("generation", **kwargs)

    def flush(self):
        return None


def test_candidate_hosts_include_localhost_fallback_for_docker_service_name():
    candidates = langfuse_client_module._candidate_langfuse_hosts("http://langfuse:3000")

    assert "http://langfuse:3000" in candidates
    assert "http://localhost:3000" in candidates
    assert "http://127.0.0.1:3000" in candidates


def test_candidate_hosts_keep_explicit_localhost_first():
    candidates = langfuse_client_module._candidate_langfuse_hosts("http://localhost:3000")

    assert candidates[0] == "http://localhost:3000"
    assert "http://127.0.0.1:3000" in candidates


def test_compat_wrapper_supports_v2_trace_span_generation_api():
    client = _FakeLangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)

    trace = compat.start_observation(name="trace", as_type="trace")
    span = trace.start_observation(name="span", as_type="span")
    generation = trace.start_observation(name="generation", as_type="generation")

    span.update(output="span-output")
    generation.update(output="generation-output")
    trace.update(output="trace-output")

    span.end()
    generation.end()
    trace.end()

    assert trace is not None
    assert isinstance(span, langfuse_client_module._ObservationCompat)
    assert isinstance(generation, langfuse_client_module._ObservationCompat)
    assert len(client.trace_calls) == 1
    assert len(client.span_calls) == 1
    assert len(client.generation_calls) == 1


def test_compat_wrapper_exposes_log_event_for_sdk_objects_without_it():
    client = _FakeLangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)

    trace = compat.start_observation(name="trace", as_type="trace")
    span = trace.start_observation(name="span", as_type="span")

    event = span.log_event(name="event", input="payload")

    assert event is not None
    assert span._observation.log_events[0][0] == "event"
    assert span._observation.log_events[0][1]["input"] == "payload"


def test_compat_wrapper_supports_span_convenience_methods_on_observation():
    client = _FakeLangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)

    trace = compat.trace(name="trace")
    span = trace.span(name="span")

    assert isinstance(trace, langfuse_client_module._ObservationCompat)
    assert isinstance(span, langfuse_client_module._ObservationCompat)
    assert span.kind == "span"


class _FakeV4Span:
    """Minimal stand-in for a v4 LangfuseSpan (start_observation + score + score_trace)."""

    def __init__(self, name, kind, span_id, trace_id):
        self.name = name
        self.kind = kind
        self.id = span_id
        self.trace_id = trace_id
        self.updated = {}
        self.ended = False
        self.scores = []
        self.trace_scores = []
        self._otel_span = _FakeOtelSpan(span_id, trace_id)

    def update(self, **kwargs):
        self.updated.update(kwargs)
        return self

    def end(self):
        self.ended = True
        return self

    def start_observation(self, name, **kwargs):
        as_type = kwargs.pop("as_type", "span")
        child = _FakeV4Span(name, as_type, f"00000000000000{len(self.scores) + 1}", self.trace_id)
        self.children = getattr(self, "children", [])
        self.children.append((child, kwargs))
        return child

    def score(self, **kwargs):
        self.scores.append(kwargs)
        return None

    def score_trace(self, **kwargs):
        self.trace_scores.append(kwargs)
        return None

    def create_event(self, name, **kwargs):
        return None


class _FakeOtelSpan:
    def __init__(self, span_id, trace_id):
        self.context = _FakeSpanContext(span_id, trace_id)


class _FakeSpanContext:
    def __init__(self, span_id, trace_id):
        self.span_id = int(span_id, 16)
        self.trace_id = int(trace_id, 16)


class _FakeV4LangfuseClient:
    def __init__(self):
        self.spans = []
        self.scores = []
        self.datasets = []

    def start_observation(self, name, **kwargs):
        as_type = kwargs.get("as_type", "span")
        span_id = f"00000000000000{len(self.spans) + 1}"
        trace_id = f"{'a' * 32}"
        span = _FakeV4Span(name, as_type, span_id, trace_id)
        self.spans.append((name, kwargs, span))
        return span

    def create_score(self, **kwargs):
        self.scores.append(kwargs)
        return None

    def create_dataset(self, **kwargs):
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs):
        self.datasets.append(kwargs)

    def flush(self):
        return None


def test_v4_root_trace_maps_as_type_trace_to_span_and_propagates_attrs():
    client = _FakeV4LangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)

    trace = compat.start_observation(
        name="rag-query-pipeline",
        input="hello",
        as_type="trace",
        user_id="u1",
        session_id="s1",
        tags=["app:test"],
    )

    assert trace.kind == "trace"
    # v4 has no as_type="trace"; the underlying call must use "span"
    assert client.spans[0][1]["as_type"] == "span"
    assert client.spans[0][1]["input"] == "hello"
    # trace-level kwargs must NOT be forwarded to start_observation (no **kwargs in v4)
    assert "user_id" not in client.spans[0][1]
    assert "tags" not in client.spans[0][1]
    # id/trace_id derived from the OTel span context
    assert trace.id is not None and len(trace.id) == 16
    assert trace.trace_id is not None and len(trace.trace_id) == 32


def test_v4_child_observation_delegates_to_parent_object():
    client = _FakeV4LangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)

    trace = compat.start_observation(name="trace", as_type="trace")
    child = trace.start_observation(name="retrieval", as_type="span")

    assert isinstance(child, langfuse_client_module._ObservationCompat)
    assert child.kind == "span"
    assert child.trace_id == trace.trace_id
    # child links to parent via the v4 parent object's start_observation
    assert len(trace._observation.children) == 1


def test_v4_observation_score_uses_span_score_trace():
    client = _FakeV4LangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)

    trace = compat.start_observation(name="trace", as_type="trace")
    trace.score(name="confidence", value=0.9)

    assert len(trace._observation.trace_scores) == 1
    assert trace._observation.trace_scores[0]["name"] == "confidence"
    assert trace._observation.trace_scores[0]["value"] == 0.9


def test_v4_compat_score_delegates_to_create_score():
    client = _FakeV4LangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)

    compat.score(trace_id="t1", name="relevance", value=0.8, data_type="NUMERIC")

    assert len(client.scores) == 1
    assert client.scores[0]["trace_id"] == "t1"
    assert client.scores[0]["name"] == "relevance"
    assert client.scores[0]["data_type"] == "NUMERIC"


def test_derive_ids_from_v4_otel_span_context():
    from opentelemetry.trace import format_span_id, format_trace_id

    span = _FakeV4Span("x", "span", "0000000000000001", f"{'b' * 32}")
    otel = span._otel_span
    assert langfuse_client_module._derive_span_id(span) == format_span_id(otel.context.span_id)
    assert langfuse_client_module._derive_trace_id(span) == format_trace_id(otel.context.trace_id)


def test_langfuse_datasets_upload_evaluation_dataset_rows_uses_v4_top_level_api(monkeypatch):
    client = _FakeV4LangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)
    monkeypatch.setattr(
        "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
        lambda: compat,
    )

    from data_engineering_copilot.evaluation import langfuse_datasets

    ok = langfuse_datasets.upload_evaluation_dataset_rows(
        dataset_name="test-ds",
        items=[{"input": {"q": "1"}, "expected_output": {"a": "x"}, "metadata": {}}],
    )

    assert ok is True
    assert len(client.datasets) == 2  # create_dataset + create_dataset_item
    assert client.datasets[0]["name"] == "test-ds"
    assert client.datasets[1]["dataset_name"] == "test-ds"


def test_run_experiment_raises_not_implemented_until_phase6(monkeypatch):
    client = _FakeV4LangfuseClient()
    compat = langfuse_client_module.LangfuseCompat(client)
    monkeypatch.setattr(
        "data_engineering_copilot.evaluation.langfuse_datasets.get_langfuse_client",
        lambda: compat,
    )

    from data_engineering_copilot.evaluation import langfuse_datasets

    with pytest.raises(NotImplementedError):
        langfuse_datasets.run_experiment(experiment_name="e1", dataset_name="ds", config_a={}, config_b={})
