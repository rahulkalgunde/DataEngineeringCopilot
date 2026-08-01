# ADR-001: Circuit Breaker for LLM Providers

**Date:** 2026-07-30  
**Status:** Superseded (2026-08-01) — the per-client `CircuitBreaker` was removed. The provider-health cooldown registry in `infrastructure/provider_health.py` now acts as the single circuit breaker: any failure sets a category-based cooldown and the fail-fast/failover-first `AdaptiveLLMRouter` skips providers in cooldown without a call. See `AGENTS.md` → Adaptive fallback chain.
**Deciders:** Architecture Team  

## Context

LLM provider calls can fail due to transient issues (network blips, overload, rate limiting). Without protection, repeated failures compound latency by waiting for timeouts on each call.

## Decision

Add a `CircuitBreaker` class to `llm_client.py` that wraps `generate()` calls:

- **Closed** → calls pass through normally
- **Open** (after 3 consecutive failures) → all calls fail instantly with `CircuitBreakerError`
- **Half-open** (after 30s recovery window) → one probe request allowed
- Half-open success → close; failure → re-open

The breaker operates on the `call()` method with a hard 10s `asyncio.wait_for` timeout. Any exception (including `TimeoutError`) counts as a failure.

## Consequences

- Positive: Prevents cascading timeouts; total wait drops from `N * timeout` to ~30s
- Positive: Provider unavailability is detected in ~3 requests instead of minutes
- Negative: Failed probes during recovery add ~3 extra failures; acceptable for 30s window
