# DataEngineeringCopilot

Offline question answering for data engineering documentation using Ollama, deepseek-coder:6.7b, ChromaDB, sentence-transformers, and Streamlit.

## Project Structure

```text
DataEngineeringCopilot/
  main.py
  requirements.txt
  README.md
  chroma_db/
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

Create and activate a Python virtual environment for your platform.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install and start Ollama, then pull the model once:

```bash
ollama pull deepseek-coder:6.7b
ollama serve
```

Download the sentence-transformers embedding model once during setup:

```bash
python scripts/download_embedding_model.py
```

## Build the Local Repository

The crawler downloads documentation pages and stores chunks in local ChromaDB. After ingestion, question answering is fully local: ChromaDB reads from disk, sentence-transformers loads from `data/embedding_models`, and Ollama runs `deepseek-coder:6.7b` locally.

```bash
python main.py ingest --max-pages 40
```

If ChromaDB reports an incomplete local index, reset and ingest again:

```bash
python main.py reset-index
python main.py ingest --max-pages 40
```

Note about reusing the index across machines:

If you switch between Windows and Unix environments, the local `chroma_db/` folder contains the persisted index. To avoid re-ingesting the documentation (which is time-consuming), copy or sync the `chroma_db/` directory between machines (for example using `rsync`, a shared drive, or a git-annex-like solution). Keeping a single shared `chroma_db/` avoids duplicate re-indexing when moving the project.


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

If your virtual environment is activated, this uses the environment's Streamlit installation. On Windows you can also run the equivalent PowerShell command from the venv:

```powershell
.\.venv\Scripts\streamlit.exe run data_engineering_copilot\ui\streamlit_app.py
```

The sidebar includes a `Refresh Documentation` button. It crawls the configured documentation sources and upserts new or updated chunks into ChromaDB. Ingestion requires internet access; answering after ingestion runs locally.

Runtime logs are written under `logs/` in the project workspace:

- `logs/application.log` captures CLI, Streamlit, ingestion, retrieval, vector store, and Ollama events for troubleshooting.
- `logs/ingestion_refresh.log` captures detailed UI refresh events and fetched documentation URLs.

## Architecture

This project intentionally does not use LangChain or LlamaIndex.

- `config`: source URLs and runtime settings
- `domain`: simple dataclasses shared by the app
- `infrastructure`: adapters for HTTP crawling, HTML parsing, embeddings, ChromaDB, and Ollama
- `services`: business workflows for ingestion and RAG answering
- `ui`: Streamlit interface

Local generation can take time on CPU. The timeout and generation limits are configured in `data_engineering_copilot/config/settings.py` as `ollama_timeout_seconds`, `ollama_num_ctx`, `ollama_num_predict`, `retrieval_top_k`, and `max_context_chars`.

If Ollama fails due to prompt or output length, the service automatically retries with reduced repository context and then with a larger output budget. You can tune this behavior with `ollama_retry_context_ratio`, `ollama_retry_extra_num_predict`, and `ollama_retry_max_num_predict` in the same settings file.

Default retry settings in `data_engineering_copilot/config/settings.py`:

```python
ollama_retry_context_ratio = 0.6
ollama_retry_extra_num_predict = 2048
ollama_retry_max_num_predict = 4096
```

---

CLI helpers to export/import the index

You can export the local `chroma_db/` to a zip archive and import it on another machine using the included CLI commands:

```bash
# Export to chroma_db_export.zip (defaults):
python main.py export-index

# Export to a specific path:
python main.py export-index --output /tmp/my_chroma.zip

# On the target machine, import the archive:
python main.py import-index /tmp/my_chroma.zip
```
