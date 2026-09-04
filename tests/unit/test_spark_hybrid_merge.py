"""Task 6 tests: hybrid native + rendered corpus merge and deduplication."""

from __future__ import annotations

import asyncio
from pathlib import Path

from data_engineering_copilot.config.settings import (
    SparkRenderedBuildConfig,
    SparkRenderedSourceConfig,
    SparkSourceConfig,
    SparkStreamConfig,
)
from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
from data_engineering_copilot.infrastructure.spark_rendered_builder import (
    RenderedFileRecord,
    RenderedManifest,
    load_rendered_manifest,
)
from data_engineering_copilot.infrastructure.spark_source_resolver import SparkFileRecord, SparkManifest
from data_engineering_copilot.services.chunker import deduplicate_chunks
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.spark_chunker import SparkChunker
from data_engineering_copilot.services.spark_index_builder import CoverageRecord, SparkIndexBuilder

_COMMIT = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"


def _config() -> SparkSourceConfig:
    return SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit=_COMMIT,
        license="Apache-2.0",
        streams=(
            SparkStreamConfig("guides", "guide", ("docs/**/*.md",), (), "conceptual", "header_aware"),
            SparkStreamConfig("api", "api_reference", ("python/pyspark/**/*.py",), (), "python", "api"),
            SparkStreamConfig("examples", "code_example", ("examples/**/*.py",), (), "mixed", "code"),
        ),
    )


def _rendered_config() -> SparkRenderedSourceConfig:
    return SparkRenderedSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit=_COMMIT,
        license="Apache-2.0",
        builds=(
            SparkRenderedBuildConfig(
                name="jekyll_docs",
                doc_type="guide",
                language="conceptual",
                working_dir="docs",
                command=("bundle", "exec", "jekyll", "build"),
                env=(),
                output_root="{output}",
                include=("**/*.html",),
                exclude=("index.html",),
                content_root_selector="div#content",
                excluded_selectors=(".left-menu-wrapper",),
                canonical_url="https://spark.apache.org/docs/4.0.0/{relpath}",
                renderer="jekyll",
            ),
            SparkRenderedBuildConfig(
                name="pyspark_api",
                doc_type="api_reference",
                language="python",
                working_dir="python/docs",
                command=("sphinx",),
                env=(),
                output_root="{output}/html",
                include=("reference/**/*.html",),
                exclude=(),
                content_root_selector="main",
                excluded_selectors=(),
                canonical_url="https://spark.apache.org/docs/4.0.0/api/python/{relpath}",
                renderer="sphinx",
            ),
        ),
    )


def _builder(
    tmp_path: Path,
    rendered_manifest: RenderedManifest | None = None,
) -> SparkIndexBuilder:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    return SparkIndexBuilder(
        config=_config(),
        resolver=None,  # type: ignore[arg-type]
        parser=NativeDocumentParser(),
        chunker=SparkChunker(header_chunker=HeaderAwareChunker()),
        store=store,
        generation="gen-1",
        embedder=None,  # type: ignore[arg-type]
        rendered_config=_rendered_config(),
        rendered_manifest=rendered_manifest,
    )


def _native_manifest(tmp_path: Path) -> tuple[SparkManifest, dict[str, Path]]:
    """Create native fixture files: a guide, an API module, an example."""
    files: list[SparkFileRecord] = []
    paths: dict[str, Path] = {}

    guide = tmp_path / "native" / "docs" / "window.md"
    guide.parent.mkdir(parents=True)
    guide.write_text(
        "# Window Functions\n\nUse `dense_rank` with a partition and ordering window.\n",
        encoding="utf-8",
    )
    paths["guide"] = guide
    files.append(
        SparkFileRecord(
            stream="guides",
            relative_path="docs/window.md",
            absolute_path=guide,
            doc_type="guide",
            language="conceptual",
            source_url=f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/docs/window.md",
        )
    )

    api = tmp_path / "native" / "python" / "pyspark" / "sql" / "functions.py"
    api.parent.mkdir(parents=True)
    api.write_text(
        'def filter(f):\n    """Filter rows using the given predicate."""\n    ...\n\n\n'
        'def dense_rank():\n    """Returns the rank of rows within a window partition."""\n    ...\n',
        encoding="utf-8",
    )
    paths["api"] = api
    files.append(
        SparkFileRecord(
            stream="api",
            relative_path="python/pyspark/sql/functions.py",
            absolute_path=api,
            doc_type="api_reference",
            language="python",
            source_url=f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/python/pyspark/sql/functions.py",
        )
    )

    example = tmp_path / "native" / "examples" / "src" / "main" / "python" / "nested_arrays.py"
    example.parent.mkdir(parents=True)
    example.write_text('# nested array example\n\nspark.sql("SELECT array(1, 2, 3)")\n', encoding="utf-8")
    paths["example"] = example
    files.append(
        SparkFileRecord(
            stream="examples",
            relative_path="examples/src/main/python/nested_arrays.py",
            absolute_path=example,
            doc_type="code_example",
            language="python",
            source_url=f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/examples/src/main/python/nested_arrays.py",
        )
    )

    manifest = SparkManifest(
        source_name="Apache Spark 4.0.0",
        ref="v4.0.0",
        commit=_COMMIT,
        root=tmp_path / "native",
        files=tuple(files),
        manifest_hash="fixture-hash",
    )
    return manifest, paths


def _guide_html(text: str) -> str:
    return f"""<html><head><title>Window Functions - Spark</title></head>
<body><div class="left-menu-wrapper"><ul><li>Sidebar</li></ul></div>
<div id="content">{text}</div></body></html>"""


def _sphinx_html(text: str) -> str:
    return f"<html><head><title>pyspark.sql.functions.filter</title></head><body><main>{text}</main></body></html>"


def _rendered_manifest(tmp_path: Path) -> RenderedManifest:
    """Create rendered fixture files: a guide page, a module page, a function page."""
    records: list[RenderedFileRecord] = []
    root = tmp_path / "rendered"

    jekyll = root / "jekyll_docs" / "output"
    jekyll.mkdir(parents=True)
    guide_html = jekyll / "window.html"
    guide_html.write_text(
        _guide_html(
            "<h1>Window Functions</h1>"
            "<p>Rendered window function guide content with dense_rank ranking and rolling aggregates.</p>"
        ),
        encoding="utf-8",
    )
    records.append(
        RenderedFileRecord(
            build="jekyll_docs",
            relative_path="window.html",
            absolute_path=guide_html,
            doc_type="guide",
            language="conceptual",
            canonical_url="https://spark.apache.org/docs/4.0.0/window.html",
        )
    )

    sphinx = root / "pyspark_api" / "output" / "html"
    module_html = sphinx / "reference" / "pyspark.sql" / "functions.html"
    module_html.parent.mkdir(parents=True)
    module_html.write_text(
        _sphinx_html(
            "<h1>pyspark.sql.functions</h1>"
            "<p>Rendered module page listing all functions with their signatures and descriptions.</p>"
        ),
        encoding="utf-8",
    )
    records.append(
        RenderedFileRecord(
            build="pyspark_api",
            relative_path="reference/pyspark.sql/functions.html",
            absolute_path=module_html,
            doc_type="api_reference",
            language="python",
            canonical_url="https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/functions.html",
        )
    )

    func_html = sphinx / "reference" / "pyspark.sql" / "api" / "pyspark.sql.functions.filter.html"
    func_html.parent.mkdir(parents=True)
    func_html.write_text(
        _sphinx_html(
            "<h1>pyspark.sql.functions.filter</h1>"
            "<p>Filters rows using the given predicate function over the input column values.</p>"
        ),
        encoding="utf-8",
    )
    records.append(
        RenderedFileRecord(
            build="pyspark_api",
            relative_path="reference/pyspark.sql/api/pyspark.sql.functions.filter.html",
            absolute_path=func_html,
            doc_type="api_reference",
            language="python",
            canonical_url="https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.filter.html",
        )
    )

    # A navigation-only page with no main content -> must be rejected.
    nav_html = sphinx / "reference" / "index.html"
    nav_html.write_text("<html><body><main>Home</main></body></html>", encoding="utf-8")
    records.append(
        RenderedFileRecord(
            build="pyspark_api",
            relative_path="reference/index.html",
            absolute_path=nav_html,
            doc_type="api_reference",
            language="python",
            canonical_url="https://spark.apache.org/docs/4.0.0/api/python/reference/index.html",
        )
    )

    return RenderedManifest(
        source_name="Apache Spark 4.0.0",
        ref="v4.0.0",
        commit=_COMMIT,
        root=root,
        files=tuple(records),
        manifest_hash="rendered-hash",
    )


def test_merge_prefers_rendered_and_keeps_native_provenance(tmp_path) -> None:
    manifest, _ = _native_manifest(tmp_path)
    builder = _builder(tmp_path, _rendered_manifest(tmp_path))
    merged = builder._merge_documents(manifest, SparkChunker(header_chunker=HeaderAwareChunker()))

    by_key = {doc.key: doc for doc in merged}
    guide = by_key["https://spark.apache.org/docs/4.0.0/window.html"]
    assert guide.representation == "rendered"
    assert "Rendered window function guide content" in guide.parsed.text
    assert guide.parsed.url == "https://spark.apache.org/docs/4.0.0/window.html"
    # Native provenance retained: file_path points at the repo markdown file.
    assert guide.metadata_relative_path == "docs/window.md"

    module = by_key["https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/functions.html"]
    assert module.representation == "rendered"
    assert module.metadata_relative_path == "python/pyspark/sql/functions.py"

    # Example stream has no rendered counterpart -> stays native.
    example = [d for d in merged if d.representation == "native" and "examples" in d.metadata_relative_path]
    assert example
    assert example[0].parsed.text


def test_merge_includes_rendered_only_function_page(tmp_path) -> None:
    manifest, _ = _native_manifest(tmp_path)
    builder = _builder(tmp_path, _rendered_manifest(tmp_path))
    merged = builder._merge_documents(manifest, SparkChunker(header_chunker=HeaderAwareChunker()))

    func = [
        d
        for d in merged
        if d.key
        == "https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.filter.html"
    ]
    assert len(func) == 1
    assert func[0].representation == "rendered"
    assert func[0].metadata_relative_path == "reference/pyspark.sql/api/pyspark.sql.functions.filter.html"


def test_merge_drops_navigation_only_rendered_page(tmp_path) -> None:
    manifest, _ = _native_manifest(tmp_path)
    builder = _builder(tmp_path, _rendered_manifest(tmp_path))
    merged = builder._merge_documents(manifest, SparkChunker(header_chunker=HeaderAwareChunker()))
    keys = {d.key for d in merged}
    assert "https://spark.apache.org/docs/4.0.0/api/python/reference/index.html" not in keys


def test_merge_without_rendered_keeps_all_native(tmp_path) -> None:
    manifest, _ = _native_manifest(tmp_path)
    builder = _builder(tmp_path, None)
    merged = builder._merge_documents(manifest, SparkChunker(header_chunker=HeaderAwareChunker()))
    assert len(merged) == 3
    assert all(d.representation == "native" for d in merged)


def test_native_canonical_url_mapping() -> None:
    builder = _builder(Path("/tmp/unused"))
    guide = SparkFileRecord(
        stream="guides",
        relative_path="docs/streaming/structured-streaming-programming-guide.md",
        absolute_path=Path("/tmp/unused"),
        doc_type="guide",
        language="conceptual",
        source_url="https://raw.githubusercontent.com/apache/spark/x/docs/streaming/structured-streaming-programming-guide.md",
    )
    assert builder._native_canonical_url(guide) == (
        "https://spark.apache.org/docs/4.0.0/streaming/structured-streaming-programming-guide.html"
    )

    api = SparkFileRecord(
        stream="api",
        relative_path="python/pyspark/sql/functions.py",
        absolute_path=Path("/tmp/unused"),
        doc_type="api_reference",
        language="python",
        source_url="https://raw.githubusercontent.com/apache/spark/x/python/pyspark/sql/functions.py",
    )
    assert builder._native_canonical_url(api) == (
        "https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/functions.html"
    )

    example = SparkFileRecord(
        stream="examples",
        relative_path="examples/src/main/python/nested_arrays.py",
        absolute_path=Path("/tmp/unused"),
        doc_type="code_example",
        language="python",
        source_url="https://raw.githubusercontent.com/apache/spark/x/examples/src/main/python/nested_arrays.py",
    )
    assert builder._native_canonical_url(example) == example.source_url


def test_dedup_by_content_hash_keeps_first() -> None:
    chunk_a = DocumentChunk(
        chunk_id="a",
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text="same content",
        content_hash="h1",
        doc_type="guide",
    )
    chunk_b = DocumentChunk(
        chunk_id="b",
        source_name="Apache Spark 4.0.0",
        title="T2",
        url="http://y",
        text="same content",
        content_hash="h1",
        doc_type="guide",
    )
    chunk_c = DocumentChunk(
        chunk_id="c",
        source_name="Apache Spark 4.0.0",
        title="T3",
        url="http://z",
        text="other content",
        content_hash="h2",
        doc_type="guide",
    )
    deduped = deduplicate_chunks([chunk_a, chunk_b, chunk_c])
    assert [c.chunk_id for c in deduped] == ["a", "c"]


def test_dedup_by_content_hash_normalizes_whitespace() -> None:
    """Identical content differing only in trailing whitespace (e.g. the ASF
    license header across files) must collapse to a single copy so parent
    content hashes stay unique per source chunk."""
    chunk_a = DocumentChunk(
        chunk_id="a",
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text="license header",
        content_hash="raw-hash-a",
        doc_type="guide",
    )
    chunk_b = DocumentChunk(
        chunk_id="b",
        source_name="Apache Spark 4.0.0",
        title="T2",
        url="http://y",
        text="license header  \n",  # trailing whitespace -> different raw hash
        content_hash="raw-hash-b",
        doc_type="guide",
    )
    deduped = deduplicate_chunks([chunk_a, chunk_b])
    assert [c.chunk_id for c in deduped] == ["a"]


def test_dedup_normalizes_case_and_whitespace() -> None:
    """Near-duplicates differing only in case and whitespace must collapse."""
    chunk_a = DocumentChunk(
        chunk_id="a",
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text="License Header Text",
        content_hash="raw-hash-a",
        doc_type="guide",
    )
    chunk_b = DocumentChunk(
        chunk_id="b",
        source_name="Apache Spark 4.0.0",
        title="T2",
        url="http://y",
        text="license  header   text\n",  # lowercase + extra whitespace
        content_hash="raw-hash-b",
        doc_type="guide",
    )
    deduped = deduplicate_chunks([chunk_a, chunk_b])
    assert [c.chunk_id for c in deduped] == ["a"]


def test_rendered_chunks_carry_representation(tmp_path) -> None:
    manifest, _ = _native_manifest(tmp_path)
    builder = _builder(tmp_path, _rendered_manifest(tmp_path))

    async def _chunk() -> tuple[list[DocumentChunk], list[CoverageRecord]]:
        return await builder._chunk_all(manifest)

    chunks, coverage = asyncio.run(_chunk())
    reps = {c.representation for c in chunks}
    assert "native" in reps
    assert "rendered" in reps
    for c in chunks:
        assert c.representation in ("native", "rendered")
    # Rendered module page replaces the native functions.py file: its chunks
    # must carry the repo path as provenance.
    rendered_module = [c for c in chunks if c.representation == "rendered" and "functions" in c.file_path]
    assert any(c.file_path == "python/pyspark/sql/functions.py" for c in rendered_module)


def test_load_rendered_manifest_reconstructs_paths(tmp_path) -> None:
    path = tmp_path / "rendered_manifest.json"
    path.write_text(
        '{"source_name":"Apache Spark 4.0.0","ref":"v4.0.0","commit":"' + _COMMIT + '",'
        '"manifest_hash":"rendered-hash",'
        '"files":[{"build":"jekyll_docs","relative_path":"window.html","doc_type":"guide",'
        '"language":"conceptual","canonical_url":"https://spark.apache.org/docs/4.0.0/window.html"}]}',
        encoding="utf-8",
    )
    loaded = load_rendered_manifest(path, tmp_path / "rendered", _rendered_config())
    assert loaded.manifest_hash == "rendered-hash"
    assert len(loaded.files) == 1
    assert loaded.files[0].absolute_path == (tmp_path / "rendered" / "jekyll_docs" / "output" / "window.html").resolve()
