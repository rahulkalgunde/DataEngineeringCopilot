# ADR-007: Speculative Decoding for the Generation Layer

**Date:** 2026-08-21
**Status:** Proposed
**Deciders:** Architecture Team

## Context

The generation-layer optimization plan (see `plans/2026-08-21_08-55_generation_layer_implementation_plan.md`)
identifies speculative decoding as a latency lever: a cheap draft model proposes *k* tokens that a
larger target model verifies in a single forward pass (mathematically lossless). This is distinct
from the app-level streaming work (already implemented) — speculative decoding is a *server-side*
inference optimization, not a change to `LLMClient` or `AsyncRagService`.

## Decision

Do **not** add speculative-decoding code to the application. It is configured on the inference
server:
- **vLLM:** EAGLE / MTP / draft-model / n-gram (`--speculative-config`); benchmark before enabling.
- **Ollama / llama.cpp:** draft-model pairing.
- **SGLang / LMDeploy:** EAGLE-3 / DeepSeek MTP.

The `generation_temperature` (0.0–0.2) we set in P1 is *favorable* to speculative decoding, because
acceptance rate drops under high-temperature / high-entropy generation.

## Consequences

- No app-layer change; keep `LLMClient` provider-agnostic.
- Benefit is workload-dependent: 1.5–3× speedups are reported on predictable, low-temperature,
  structured/code/RAG outputs, but there is **no guaranteed gain** (and possible regression) under
  high concurrency or high-temperature generation. Must be benchmarked on real traffic with
  `dec eval-generation` (rubric/faithfulness) re-run to confirm quality is unchanged.
- Revisit only if TTFT becomes a measured bottleneck after P1–P3 land.
