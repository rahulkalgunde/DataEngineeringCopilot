# ADR-004: Dependency Audit

**Date:** 2026-07-30  
**Status:** Informational  

## Installed vs Pinned

| Dependency | Pinned Min | Installed | Status |
|---|---|---|---|
| crawl4ai | >=0.2 | 0.9.0 | **MAJOR GAP** — API surface likely changed significantly |
| sentence-transformers | >=3.0 | 5.6.0 | **MAJOR GAP** — heavy version jump, verify API compatibility |
| qdrant-client | >=1.7 | 1.18.0 | OK — well past minimum |
| testcontainers | >=3.7 | 4.14.2 | OK |
| langfuse | >=2.0 | 2.60.10 | OK |
| celery | >=5.3 | 5.6.3 | OK |
| redis | >=5.0 | 8.0.1 | OK |
| httpx | >=0.27 | 0.28.1 | OK |
| fastapi | >=0.110 | 0.138.2 | OK |
| pydantic | >=2.5 | 2.13.4 | OK |

## Findings

1. **crawl4ai 0.9.0**: Major jump from 0.2. Review `AsyncDocumentationCrawler` for API breakage. The `crawl` method signature may have changed. Requires Playwright/Chromium — this is the heaviest dependency.
2. **sentence-transformers 5.6.0**: Jump from 3.0. Cross-encoder model download path or API may have changed. Used in `services/reranker.py`.
3. **No stale/unused deps**: All listed dependencies are actively imported in the codebase.
4. **Missing version upper bounds**: No `<` or `<=` constraints anywhere. A future `pip install` could silently upgrade to a breaking version.

## Recommendations

1. Pin `crawl4ai` to `<0.10` or update the code to match 0.9 API
2. Pin `sentence-transformers` to `<6.0`
3. Add upper bounds to all critical dependencies in `pyproject.toml`
4. Run integration tests after any major dependency upgrade
