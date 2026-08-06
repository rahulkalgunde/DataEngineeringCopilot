"""Task 5 tests: SparkRenderedChunker — heading-bounded chunking, fallback, provenance."""

from __future__ import annotations

import asyncio

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.spark_metadata import SparkMetadata
from data_engineering_copilot.services.spark_rendered_chunker import SparkRenderedChunker

_COMMIT = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"


def _metadata(file_path: str = "docs/sql-data-sources-jdbc.md") -> SparkMetadata:
    return SparkMetadata(
        doc_type="guide",
        language="conceptual",
        spark_version="4.0.0",
        module="",
        source_commit=_COMMIT,
        file_path=file_path,
        license="Apache-2.0",
    )


def _parsed(text: str, metadata: SparkMetadata) -> ParsedDocument:
    return ParsedDocument(
        source_name="Apache Spark 4.0.0",
        title="JDBC To Other Databases",
        url="https://spark.apache.org/docs/4.0.0/sql-data-sources-jdbc.html",
        text=text,
        doc_type=metadata.doc_type,
        language=metadata.language,
        spark_version=metadata.spark_version,
        module=metadata.module,
        source_commit=metadata.source_commit,
        file_path=metadata.file_path,
        license=metadata.license,
    )


def test_chunks_by_heading_hierarchy() -> None:
    text = """
# JDBC To Other Databases

Spark SQL includes a data source that can read data from other databases using JDBC. This is preferred over JdbcRDD.

## Loading data from a JDBC source

```python
df = spark.read.jdbc(url, "table")
```

## Saving data to a JDBC source

Use the JDBC data source to write DataFrames back to a database table.
""".strip()
    metadata = _metadata()
    chunks = asyncio.run(SparkRenderedChunker().chunk(_parsed(text, metadata), metadata))
    assert chunks
    headers = [c.section_header for c in chunks]
    assert any("Loading data" in h for h in headers)
    assert any("Saving data" in h for h in headers)
    code_chunks = [c for c in chunks if "spark.read.jdbc" in c.text]
    assert code_chunks
    assert "```" in code_chunks[0].text


def test_preserves_code_with_section() -> None:
    text = """
# Example

```scala
val df = spark.read.format("jdbc").load()
```

This code reads a JDBC source. The DataFrame is then processed with Spark SQL.
""".strip()
    metadata = _metadata()
    chunks = asyncio.run(SparkRenderedChunker().chunk(_parsed(text, metadata), metadata))
    joined = "\n".join(c.text for c in chunks)
    assert 'format("jdbc")' in joined
    assert "processed with Spark SQL" in joined


def test_fallback_for_heading_free_page() -> None:
    text = (
        "Structured Streaming is a scalable fault-tolerant stream processing engine built on the Spark SQL "
        "engine. It processes data in micro-batches and provides exactly-once semantics. "
        "The programming model treats streams as unbounded tables with event-time handling."
    )
    metadata = _metadata("docs/streaming.txt")
    chunks = asyncio.run(SparkRenderedChunker().chunk(_parsed(text, metadata), metadata))
    assert chunks
    assert all(c.chunk_type == "text" for c in chunks)


def test_metadata_preserved_on_every_chunk() -> None:
    text = """
# Config

```python
spark.conf.set("spark.sql.shuffle.partitions", "200")
```

Every chunk must carry the source commit, path, and license for provenance.
""".strip()
    metadata = _metadata()
    chunks = asyncio.run(SparkRenderedChunker().chunk(_parsed(text, metadata), metadata))
    assert chunks
    for chunk in chunks:
        assert chunk.source_commit == _COMMIT
        assert chunk.file_path == "docs/sql-data-sources-jdbc.md"
        assert chunk.license == "Apache-2.0"
        assert chunk.url == "https://spark.apache.org/docs/4.0.0/sql-data-sources-jdbc.html"


def test_chunk_ids_unique_and_deterministic() -> None:
    text = """
# Window Functions

Window functions operate on a group of rows and return a value per row.

## Dense Rank

```sql
SELECT dense_rank() OVER (ORDER BY amount DESC) FROM sales
```
""".strip()
    metadata = _metadata()
    first = asyncio.run(SparkRenderedChunker().chunk(_parsed(text, metadata), metadata))
    second = asyncio.run(SparkRenderedChunker().chunk(_parsed(text, metadata), metadata))
    ids1 = [c.chunk_id for c in first]
    ids2 = [c.chunk_id for c in second]
    assert len(ids1) == len(set(ids1))
    assert ids1 == ids2
    assert all(c.chunk_id.startswith("spark-rendered-") for c in first)


def test_chunk_types() -> None:
    text = """
# Pure Code

```python
def f(x): return x + 1
```
""".strip()
    metadata = _metadata()
    chunks = asyncio.run(SparkRenderedChunker().chunk(_parsed(text, metadata), metadata))
    assert chunks
    assert all(c.chunk_type in ("code", "text", "mixed", "api") for c in chunks)


def test_empty_text_returns_empty() -> None:
    metadata = _metadata()
    chunks = asyncio.run(SparkRenderedChunker().chunk(_parsed("", metadata), metadata))
    assert chunks == []
