# 013: Schema-enforced structured outputs for doc-intent answers

Date: 2026-08-21 · Status: Accepted

## Context

Downstream consumers (CLI, API, eval harnesses) parse answers from free text;
occasional malformed or unparseable LLM output caused retry loops and
inconsistent envelopes. OpenAI-class providers expose constrained decoding
(`response_format=json_schema`, strict mode); Ollama exposes `format=`.

## Decision

Doc-intent (and code-intent) answers use a hand-written strict JSON schema
(`services/structured_output.py::STRUCTURED_RAG_ANSWER_SCHEMA`) enforcing
`{answer: string, citations: string[], missing_info: bool}` — all fields
required, `additionalProperties: false`.

- Emission is capability-gated by `infrastructure/provider_capabilities.py`:
  Ollama gets the `format=` payload key; OpenAI-compatible providers get
  `response_format={"type": "json_schema", ...}`; unsupported providers
  silently omit the schema (never error).
- Parsing: `parse_structured_rag_response` (schema-first, permissive
  fallback); one JSON-retry with a stricter suffix prompt when parsing fails
  on answers >20 chars.
- Post-generation `verify_citations` drops citations not matching retrieved
  source names.

## Consequences

- Provider-side citations APIs (e.g. Anthropic Citations) remain unused —
  they are incompatible with structured outputs and would fragment the
  multi-provider chain.
- The streamed path does not yet apply the JSON-retry loop mid-stream
  (tracked as deferred work in plans/2026-08-21_22-49_gap_fix_plan.md).
