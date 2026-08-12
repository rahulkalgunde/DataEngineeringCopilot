---
description: Check GitHub Actions CI health and fix any failures (daily + on-demand).
agent: build
---

Run a CI health check for this repository and fix any failures found.

## Steps

1. **List recent runs**: `gh run list --limit 15`
2. **Identify failures**: Find the most recent run on `main` (or the latest PR). If its status is `failure` or `cancelled`, identify the failing job and run ID.
3. **Inspect failed job logs**: 
   - `gh run view <run-id>` — see job-level status
   - `gh run view <run-id> --log-failed 2>&1` — see the actual error/failing test output
   - If a job's `--log-failed` is empty, use `gh api repos/<owner>/<repo>/actions/runs/<run-id>/jobs` and pull the job log via `gh api .../actions/jobs/<job-id>/logs`.
4. **Diagnose**: Determine the root cause from the error output (stale test, code bug, config drift, flaky test, missing secret, infra issue). Do NOT guess — read the actual output.
5. **Check flakiness**: Look at the same test in the previous 3 runs. Repeated failures = real regression to fix. One-off after a green streak = rerun first (`gh run rerun <run-id> --failed`) before treating as a bug.
6. **Fix real failures**: Create a fix commit. Reproduce locally first where possible:
   - `dec_venv/bin/python -m pytest tests/unit/<specific_test> -v -n 0`
   - `dec_venv/bin/python -m ruff check data_engineering_copilot/ tests/ --fix`
   - `dec_venv/bin/python -m ruff format data_engineering_copilot/ tests/`
   - `dec_venv/bin/python -m pyright data_engineering_copilot/ tests/`
   - Only run integration/e2e tests if infra is available (`REQUIRE_INFRA=1`).
7. **Report**: Summarize in a compact table:
   - Run ID, status, failing job(s), root cause, action taken (fixed / reran / escalated).

## Guards

- NEVER run `dec probe-llm` (live paid API calls) without explicit user approval.
- NEVER commit without the user's go-ahead.
- If a failure needs secrets/infra/paid-API access that is unavailable, leave an explicit todo in the session context and note it in `sessions/`.
- If the same test fails 3 consecutive times, HALT and write `plans/BLOCKER_<timestamp>.md` (RULE 9, strike circuit breaker).
