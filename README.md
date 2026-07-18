# DataEngineeringCopilot

Offline question answering for data engineering documentation using Ollama, llama3.2:3b, Qdrant, and Streamlit.

## Project Structure

```text
DataEngineeringCopilot/
  main.py
  README.md
  qdrant_db/ -> qdrant_db/
  data/
  data_engineering_copilot/
    config/
      documentation_sources.json
      settings.py
    domain/
      models.py
    infrastructure/
      crawler.py
      embeddings.py
      html_parser.py
      ollama_client.py
      vector_store.py
    services/
      chunker.py
      ingestion.py
      rag.py
    ui/
      streamlit_app.py
    utils/
      text.py
  scripts/
    download_embedding_model.py
```

## Setup

# Package Management Constraints
- NEVER use standard 'pip' or 'python -m venv' commands.
- This project exclusively uses 'uv' as its Python package and environment manager.
- To create or manage virtual environments, use: `uv venv dec_venv`
- To install packages, use: `uv pip install -e ".[dev]"`
- To add a single package to the environment, use: `uv pip install <package_name>`
- Always ensure you target the correct local virtual environment binary path: `dec_venv/bin/python`

On windows machine, Install and start Ollama, then run the models:

```bash
ollama serve
ollama pull nomic-embed-text:latest
ollama pull qwen3.5:9b
```

Docker

1. Start Docker Desktop on windows machine
2. Login to wsl and go to Project Directory
3. Activate python venv `source dec_venv/bin/activate`
3. Run: `docker compose up -d`

Always Use Python virtual environment located at `dec_venv/` at the project root.

Linux/macOS:

```bash
uv venv dec_venv
source dec_venv/bin/activate
uv pip install -e ".[dev]"
```

No additional embedding model download is required. The system uses Ollama's `nomic-embed-text` model via HTTP API.

## Build the Local Repository

The crawler downloads documentation pages and stores chunks in local Qdrant. After ingestion, question answering is fully local: Qdrant reads from disk, Ollama runs `nomic-embed-text` and `llama3.2:3b` locally.

```bash
python main.py ingest --max-pages 40
```

If Qdrant reports an incomplete local index, reset and ingest again:

```bash
python main.py reset-index
python main.py ingest --max-pages 40
```

The configured documentation sources are:

- Apache Spark Documentation
- Apache Airflow Documentation
- Databricks Documentation
- Delta Lake Documentation

Edit documentation source URLs in:

```text
data_engineering_copilot/config/documentation_sources.json
```

Each chunk stores:

- source name
- title
- original URL
- chunk id
- chunk text

## Ask from the CLI

```bash
python main.py ask "How does Delta Lake time travel work?"
```

If the best retrieval confidence is below the configured threshold, the system returns:

```text
I cannot answer this question because it is outside my knowledge repository.
```

## Run the UI

```bash
python -m streamlit run data_engineering_copilot/ui/streamlit_app.py
```

The sidebar includes a `Refresh Documentation` button. It crawls the configured documentation sources and upserts new or updated chunks into Qdrant. Ingestion requires internet access; answering after ingestion runs locally.

Runtime logs are written under `logs/` in the project workspace:

- `logs/app.log` captures CLI, Streamlit, ingestion, retrieval, vector store, and Ollama events for troubleshooting.
- `logs/ingestion_refresh.log` captures detailed UI refresh events and fetched documentation URLs.

## Architecture

This project intentionally does not use LangChain or LlamaIndex.

- `config`: source URLs and runtime settings
- `domain`: simple dataclasses shared by the app
- `infrastructure`: adapters for HTTP crawling, HTML parsing, embeddings, Qdrant, and Ollama
- `services`: business workflows for ingestion and RAG answering
- `ui`: Streamlit interface

Local generation can take time on CPU. The timeout and generation limits are configured in `data_engineering_copilot/config/settings.py` as `ollama_timeout_seconds`, `ollama_num_ctx`, `ollama_num_predict`, `retrieval_top_k`, and `max_context_chars`.

If Ollama fails due to prompt or output length, the service automatically retries with reduced repository context and then with a larger output budget. You can tune this behavior with `ollama_retry_context_ratio`, `ollama_retry_extra_num_predict`, and `ollama_retry_max_num_predict` in the same settings file.

Default retry settings in `data_engineering_copilot/config/settings.py`:

```python
ollama_retry_context_ratio = 0.5
ollama_retry_extra_num_predict = 512
ollama_retry_max_num_predict = 1024
```
