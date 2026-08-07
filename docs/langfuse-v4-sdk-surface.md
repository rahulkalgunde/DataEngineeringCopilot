# Langfuse v4 SDK Surface — Factsheet

Verified against `langfuse==4.14.3` (Python 3.12.13). Recorded during Phase 0 (Task 0.0).
Every v4 code task in the plan references this file instead of guessing signatures.

## Client construction

```python
Langfuse(
    *,
    public_key: str | None = None,
    secret_key: str | None = None,
    base_url: str | None = None,      # alias for host
    host: str | None = None,
    timeout: int | None = None,
    debug: bool = False,              # OK — accepts debug=True (verified)
    tracing_enabled: bool | None = True,
    flush_at: int | None = None,
    flush_interval: float | None = None,
    environment: str | None = None,   # client-level env default
    release: str | None = None,       # client-level release default
    sample_rate: float | None = None, # client-level sampling
    mask=..., mask_otel_spans=..., blocked_instrumentation_scopes=...,
    should_export_span=..., additional_headers=..., tracer_provider=...,
    id_generator=..., span_exporter=...,
)
```

- `debug=True` is accepted (no TypeError).
- `auth_check()` **exists** → `-> bool`.
- `flush()`, `shutdown()`, `get_trace_url(*, trace_id=None) -> str | None` all exist.

## `start_observation` — the ONLY creation API (no trace()/span()/generation() methods)

```python
start_observation(
    *,
    trace_context: TraceContext | None = None,   # {"trace_id": str, "parent_span_id"?}
    name: str,
    as_type: (
        Literal['generation', 'embedding'] |
        Literal['span', 'agent', 'tool', 'chain', 'retriever', 'evaluator', 'guardrail']
    ) = 'span',
    input=None, output=None, metadata=None, version=None,
    level=Literal['DEBUG','DEFAULT','WARNING','ERROR'], status_message=None,
    completion_start_time=None,
    model: str | None = None,
    model_parameters=None,
    usage_details: Dict[str, int] | None = None,   # NOT "usage"
    cost_details: Dict[str, float] | None = None,  # NOT "cost"
    prompt: TextPromptClient | ChatPromptClient | None = None,
) -> LangfuseSpan | LangfuseGeneration | ...
```

- **NO `**kwargs`** — passing `user_id`, `session_id`, `tags`, `trace_id`, `data_type`, `cost`, `usage` raises `TypeError`.
- **`as_type` does NOT accept `"trace"`** — a trace is the app-root span (`span` type that becomes the root). The current wrapper must map `as_type="trace"` → `as_type="span"` for the root observation.
- Child linking: `parent.start_observation(...)` on a v4 span object auto-parents via OTel context (`use_span`). Do NOT pass `trace_id`/`parent_observation_id` — they are rejected.

## Trace-level attributes (user/session/tags/environment) — `propagate_attributes`

Module-level context manager, NOT a client method:

```python
from langfuse import propagate_attributes

with propagate_attributes(
    user_id="user_123",
    session_id="session_abc",
    tags=["intent:foo", "app:data-engineering-copilot"],
    environment="production",          # lowercase alnum, ≤40 chars, not starting "langfuse"
    metadata={"app_env": "dev", "git_sha": "abc123"},
    trace_name="rag-query-pipeline",
    # prompt=...  # links prompts to generations in the context
) as _:
    pass  # spans created here inherit these attributes
```

- Sets attributes on the currently-active span AND in OTel context so all child spans created inside the block inherit them.
- **Must be entered BEFORE creating the root span** — attributes only apply to spans created while the context is active. Pre-existing spans are NOT retrofitted.
- `Langfuse(environment=...)` / `release=` / `sample_rate=` on the client apply as defaults for all spans; per-trace `propagate_attributes(environment=...)` overrides for spans created inside it.
- **`release` is client-level only**: `propagate_attributes` does NOT accept `release` (verified `inspect.signature` in 4.14.3). `LangfuseCompat._enter_propagate` filters it out of the attrs dict; set it via `Langfuse(release=...)` in `get_langfuse_instance()` (uses `settings.image_git_sha`).
- Span attribute keys (constants in `langfuse._client.attributes.LangfuseOtelSpanAttributes`):
  - `user.id`, `session.id`, `langfuse.trace.tags`, `langfuse.environment`, `langfuse.release`, `langfuse.trace.name`, `langfuse.trace.metadata.<k>`.

## Scores — `score()` was RENAMED

- `Langfuse.score(...)` **does NOT exist** in v4. Use `create_score`:

```python
create_score(
    *,
    name: str,
    value: float | str,
    session_id: str | None = None,
    dataset_run_id: str | None = None,
    trace_id: str | None = None,
    observation_id: str | None = None,   # the span_id
    score_id: str | None = None,
    data_type: Literal['NUMERIC','CATEGORICAL','BOOLEAN','TEXT','CORRECTION'] | None = None,
    comment: str | None = None,
    config_id: str | None = None,
    metadata: Any = None,
    timestamp: datetime | None = None,
    environment: str | None = None,
) -> None
```

- `data_type` values: `NUMERIC`, `CATEGORICAL`, `BOOLEAN`, `TEXT`, `CORRECTION` (confirmed via `ScoreDataType` StrEnum).
- v4 span/generation objects expose `span.score(*, name, value, data_type, ...)` (score THIS observation) and `span.score_trace(*, name, value, data_type, ...)` (score the whole trace). Use these when holding a span object.

## trace_id / observation_id formats

- Trace ID: **32 lowercase hex chars** (`^[0-9a-f]{32}$`). Span/Observation ID: **16 lowercase hex chars** (`^[0-9a-f]{16}$`).
- Derive from the OTel span: `format_trace_id(span.context.trace_id)` / `format_span_id(span.context.span_id)` (from `opentelemetry.trace`).
- v4 span objects have NO public `.id`/`.trace_id` attributes (confirmed). `getattr(span, "id", None)` returns `None` → the compat wrapper MUST expose `id`/`trace_id` derived from the wrapped OTel span, or scoring falls silent (async_rag.py:740-741 relies on `getattr(trace, "id", None) or getattr(trace, "trace_id", None)`).

## `.update()` on observations

```python
span.update(
    *,
    name=None, input=None, output=None, metadata=None, version=None,
    level=None, status_message=None, completion_start_time=None,
    model=None, model_parameters=None,
    usage_details: Dict[str, int] | None = None,   # NOT "usage"
    cost_details: Dict[str, float] | None = None,  # NOT "cost"
    prompt=None,
    **kwargs,   # ignored
)
```

- Usage/cost fields are `usage_details` (int counts) and `cost_details` (float USD). No separate `cost`/`usage` kwargs (they land in `**kwargs` and are dropped).
- **Verified end-to-end (server 4.6.0):** `usage_details={"input","output","total","unit":"TOKENS"}` and `cost_details={"input","output","total","currency":"USD"}` both land on the generation. The public API exposes them as `usageDetails` + `costDetails`, plus computed `calculatedTotalCost` (NOT a `cost` field — reading `.cost` returns None).

## Prompts

```python
create_prompt(
    *,
    name: str,
    prompt: str | List[ChatMessageDict],   # chat: list of {"role": str, "content": str}
    labels: List[str] = [],
    tags: List[str] | None = None,
    type: Literal['chat', 'text'] = 'text',
    config: Any = None,
    commit_message: str | None = None,
) -> TextPromptClient | ChatPromptClient

get_prompt(
    name: str,
    *,
    version: int | None = None,
    label: str | None = None,
    type: Literal['chat', 'text'] = 'text',
    cache_ttl_seconds: int | None = None,
    fallback: List[ChatMessageDict] | str | None = None,
    max_retries: int | None = None,
    fetch_timeout_seconds: int | None = None,
) -> TextPromptClient | ChatPromptClient

update_prompt(*, name: str, version: int, new_labels: List[str] = []) -> Any
```

- `TextPromptClient.compile(**kwargs) -> str`
- `ChatPromptClient.compile(**kwargs) -> Sequence[Dict | ChatMessageDict | Placeholder]` — i.e. `list[dict]` (verified).
- `get_prompt(..., fallback=...)` accepts a fallback — useful for offline resilience.

## Datasets & experiments

- **Dataset creation is TOP-LEVEL** (Option A confirmed — NOT namespaced):

```python
create_dataset(*, name, description=None, metadata=None, input_schema=None, expected_output_schema=None)
create_dataset_item(*, dataset_name, input=None, expected_output=None, metadata=None,
                    source_trace_id=None, source_observation_id=None, status=None, id=None)
```

- `get_dataset(name) -> DatasetClient` — DatasetClient exposes ONLY `run_experiment` (plus inherited attrs):

```python
dataset.run_experiment(
    *,
    name: str,
    run_name: str | None = None,
    description: str | None = None,
    task: TaskFunction,                          # async def task(*, item, **kwargs) -> Any
    evaluators: List[EvaluatorFunction] = [],
    composite_evaluator=None,
    run_evaluators: List[RunEvaluatorFunction] = [],
    max_concurrency: int = 50,
    metadata=None,
) -> ExperimentResult
```

- **`run_experiment` is SYNC** (not a coroutine — verified `asyncio.iscoroutinefunction == False`). Do NOT wrap in `asyncio.run`. The `task` function itself can be async.
- `Langfuse.run_experiment(...)` also exists (client-level) with a `data:` param taking local items.
- `get_dataset_run(*, dataset_name, run_name)`, `get_dataset_runs(*, dataset_name, page, limit)`, `delete_dataset_run(*, dataset_name, run_name)`.
- `run_batched_evaluation(*, scope='traces'|'observations', mapper, filter=None, evaluators, max_concurrency=5, ...)` — production observation evaluation (Phase 7).

## Events

```python
create_event(*, trace_context=None, name, input=None, output=None, metadata=None,
             version=None, level=None, status_message=None)
```

## Instrumentation / auto-linking

- v4 uses OTel-based spans; `start_as_current_observation` is the context-manager form that makes a span "current" (needed for `propagate_attributes` to attach to the root). `start_observation` alone does NOT set the current span.

## Migration implications for `data_engineering_copilot`

1. `LangfuseCompat.start_observation`: translate `as_type="trace"` → `"span"`; drop/redirect `user_id`/`session_id`/`tags`/`environment`/`release`/`metadata` kwargs to `propagate_attributes` wrapping the root observation creation; never pass unknown kwargs.
2. `_ObservationCompat`: expose `id` (span_id) + `trace_id` from the wrapped OTel span; child `start_observation` must delegate to the v4 parent object (auto-parenting), not pass `trace_id`/`parent_observation_id`.
3. `score()` → `create_score()`; observation-level scores via `span.score()` / `span.score_trace()`.
4. `update(usage=..., cost=...)` → `update(usage_details=..., cost_details=...)`.
5. `create_dataset`/`create_dataset_item` are top-level — Option A in the plan is correct.
6. `dataset.run_experiment` is sync.

## Compatibility wrapper target surface (kept for the rest of the codebase)

`LangfuseCompat` must still expose: `start_observation`, `trace`, `span`, `generation`, `score`, `flush`, `auth_check`, `__getattr__`. `_ObservationCompat` must still expose: `update`, `end`, `log_event`, `score`, `start_observation`, `trace`, `span`, `generation`, `id`, `trace_id`, `__getattr__`.
