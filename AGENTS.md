# DataEngineeringCopilot — Agent Guide

## Python & Environment
- Always use `dec_venv/bin/python` (project-root venv). Never `python` or `pip`.
- Install: `uv pip install -e ".[dev]"`. Only `uv`, never `pip` or `python -m venv`.
- Hardcoded `dec_venv` in Makefile (`PYTHON := dec_venv/bin/python`).

## Commands & Testing
```bash
make install          # uv pip install -e ".[dev]"
make test             # all tests (parallel: -n auto --dist worksteal)
make test-quick       # unit only, no @slow (~15s)
make test-unit        # all unit tests
make test-integration # integration (sequential, --reruns 2)
make lint             # ruff check data_engineering_copilot/ tests/
make format           # ruff format data_engineering_copilot/ tests/
```
- Pytest: `asyncio_mode = "auto"` in pyproject.toml (async tests do not need `@pytest.mark.asyncio`).
- Integration tests auto-skip when services unreachable.

## Running & CLI
- CLI shortcuts: `dec_venv/bin/python main.py ...` or `dec ...` (`dec ingest`, `dec ask "..."`).
- Source names must **exactly match** entries in `data_engineering_copilot/config/documentation_sources.json`.
- Streamlit UI: `dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py`
- FastAPI API: `dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000`

## Services & Docker
- `make docker-up` → Redis (auth password: `local_secure_password_123`), Qdrant (6333/6334), Ollama (11434), Langfuse (3000), etc.
- Pull Ollama models after starting: `docker exec de_copilot_ollama ollama pull nomic-embed-text` and `llama3.2:3b`.
- CI stack: `make docker-ci-up` (`docker-compose.ci.yml`, containers prefixed `dec_ci_*`).

## Architecture & Gotchas
- **Pipeline**: No LangChain/LlamaIndex — manual pipeline (`AsyncCrawler` → `MarkdownParser` → `Chunker` → `Embeddings` → `QdrantVectorStore`).
- **Ollama Raw Mode**: `AsyncOllamaClient` sends `"raw": True` and strips `<think>` tags. Empty response means output budget exhausted (increase `ollama_num_predict` or reduce context).
- **Config**: `.env` sets `LANGFUSE_BASE_URL`, but `AppSettings` reads `LANGFUSE_HOST`.
- **Index Reset**: `dec reset-index` deletes Qdrant collection, crawl frontier SQLite DB (`data/crawl_frontier.db`), and Redis `crawl:url_registry:*` keys.
- **Dedup**: SHA-256 content hash via Redis `AsyncUrlRegistry`.

### When to Reset the Qdrant Collection

Run `dec reset-index` before re-ingesting in these scenarios:

| Scenario | Reason |
|---|---|
| **Embedding provider switch** (e.g. Ollama → OpenRouter) | Different providers produce vectors of different dimensions (nomic-embed-text: 768, nemotron-3-embed: 2048). Qdrant rejects inserts with mismatched dimensions. |
| **Embedding model change within same provider** (e.g. nomic-embed-text → mxbai-embed-large) | Models may use different vector dimensions or different semantic spaces, making old embeddings incompatible with new queries. |
| **Chunking strategy change** (e.g. fixed_size → sentence_preserving) | New chunks have different boundaries. Old chunks may produce poor retrieval results when mixed with new ones. |
| **Corrupted or incomplete index** | If Qdrant reports errors or ingestion partially failed, reset to start clean. |
| **Major documentation structure changes** | If source URLs changed significantly, old cached crawl data may be stale. |

**When NOT to reset:**
- Incremental ingestion of new/updated pages (dedup via content hash handles this)
- After code-only changes (no embedding or chunking changes)
- Changing LLM provider only (does not affect stored embeddings)

## Agent Guardrails & Behavioral Constraints
- **One Edit Per Turn**: Never modify more than ONE file at a time.
- **Read Before Writing**: Always read a file fully before editing.
- **Single-Command Rule**: Run exactly ONE terminal command at a time. Never chain with `&&` or `;`.
- **Permissions**: Do NOT install new software/utilities or run `sudo` without explicit user permission. Never use destructive delete commands (`--force`, etc.) without manual permission.
- **Anti-Looping**: If an error persists after 2 sequential attempts using the same tool, STOP and present current state.
- **Git Safety**: Never run `git commit`, `push`, `add`, or history-modifying commands — only print commands for the user.
- **Secret Hygiene**: Never include `.env`, `.env.*`, `*.key`, `*.pem`, or any file containing API keys/secrets in `git add` or `git commit` commands. Always verify staged files before printing a commit command. A pre-commit hook enforces this, but agents must also refuse proactively.
- **TDD**: Write unit tests before implementing new code or major features.

## Implementation Phase + Verification Gate

Every implementation must follow this cycle:

1. **Plan**: Write detailed plan with exact file, line numbers, and code changes
2. **Implement**: Execute the plan, one file at a time
3. **Verify**: After EACH phase, independently verify:
   - Re-read every modified file line by line
   - Compare each changed line against the plan specification (exact match required)
   - Run `make lint` (0 errors) and `make test-quick` (no new failures)
   - If ANY deviation, incompleteness, or miss found → re-plan → re-implement → re-verify
   - Only proceed to next phase after 100% verification pass
4. **Log**: Record verification result (pass/fail + details) before continuing

**Never assume implementation was done correctly. Always verify independently.**
