# ADR-002: Prompt Injection Detection Middleware

**Date:** 2026-07-30  
**Status:** Accepted  
**Deciders:** Architecture Team  

## Context

Exposed `/api/v1/ask` endpoint is vulnerable to prompt injection attacks where users embed system-level instructions (e.g. "ignore previous instructions", "reveal system prompt") that compromise the LLM.

## Decision

Add a middleware layer in `api/middleware.py` that runs before the rate limiter:

- Scans `question` field on `POST /api/v1/ask` and `/api/v1/ask/stream`
- Six regex patterns targeting known injection categories (instruction override, role-playing, system prompt extraction)
- Weighted scoring: 0.3 per pattern hit, capped at 1.0; threshold >0.5
- Rejected requests return HTTP 400 with a descriptive message
- Non-string or missing `question` fields pass through unaffected

## Consequences

- Positive: Catches common injection patterns before they reach the LLM
- Positive: Runs before rate limiter, so blocked requests don't consume quota
- Negative: Pattern-based detection is not foolproof; motivated attackers can craft bypasses
- Negative: Each match adds `re.search` overhead on the question text (~microseconds)
