# ADR-008: Domain Fine-Tuning (DPO) for the Generation Layer

**Date:** 2026-08-21
**Status:** Proposed
**Deciders:** Architecture Team

## Context

The generation-layer plan lists Direct Preference Optimization (DPO, Rafailov et al. 2023) as a
lever to teach the answer model *brevity* and *context-bound answering* via
`(query, context, chosen, rejected)` triplets. This is the highest-effort, slowest-payoff option and
is explicitly a **future lever**, not part of the current implementation (P1–P3).

## Decision

Reserve DPO for a **fixed, self-hosted open-weight model** (e.g. a pinned Ollama/vLLM model) where we
control the weight and can curate triplets. Do not attempt DPO against hosted API models.

Required before any DPO run:
- A well **SFT-tuned** reference model (DPO fine-tunes relative to a reference; a weak reference fails).
- Curated preference data emphasizing conciseness and grounding: `chosen` = concise + fully
  context-backed; `rejected` = verbose or context-violating.
- Use a **length-corrected** DPO variant (IPO / ORPO / SimPO) because vanilla DPO is biased toward
  longer responses, which conflicts with the brevity objective.

## Consequences

- High effort: curated data, GPU time, an evaluation harness, and regression tests.
- Most open-weight chat models are *already* DPO-aligned, so this is further alignment — lower data
  bar but still real cost.
- ROI is lower and slower than prompt/decoding tuning (P1–P3). Apply only when generation quality
  plateaus on verbosity/grounding after P1–P3 and the eval-generation rubric (≥4.0) stops improving.
- Out of scope for the current generation-layer work; tracked here so the option is not forgotten.
