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
