"""Phase 3 tests: Spark source resolver — path matching, safe extraction, determinism."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from data_engineering_copilot.config.settings import SparkSourceConfig, SparkStreamConfig
from data_engineering_copilot.infrastructure.spark_source_resolver import SparkSourceResolver


def _config() -> SparkSourceConfig:
    return SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(
            SparkStreamConfig(
                name="guides",
                doc_type="guide",
                include=("docs/**/*.md",),
                exclude=("docs/api/**",),
                language="conceptual",
                chunking="header_aware",
            ),
            SparkStreamConfig(
                name="api",
                doc_type="api_reference",
                include=("python/pyspark/**/*.py",),
                exclude=("**/tests/**", "**/*_test.py", "**/worker/**"),
                language="python",
                chunking="api",
            ),
            SparkStreamConfig(
                name="examples",
                doc_type="code_example",
                include=("examples/src/main/**/*.py",),
                exclude=("**/data/**",),
                language="mixed",
                chunking="code",
            ),
        ),
    )


def _make_archive(payloads: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in payloads.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _write_archive(cache_dir: Path, payloads: dict[str, bytes], commit: str) -> Path:
    digest = __import__("hashlib").sha256(commit.encode("ascii")).hexdigest()[:16]
    root = cache_dir / f"v4.0.0-{digest}"
    root.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / ".spark_commit").write_text(commit, encoding="utf-8")
    return root


# ------------------------------------------------------------------
# Path matching / manifest building
# ------------------------------------------------------------------


def test_build_manifest_filters_by_streams(tmp_path) -> None:
    root = _write_archive(
        tmp_path / "cache",
        {
            "docs/sql-guide.md": b"# Spark SQL guide\ncontent\n",
            "docs/api/index.html": b"<html>noise</html>",
            "python/pyspark/sql/functions.py": b"def filter(col, f):\n    return col\n",
            "python/pyspark/sql/tests/test_functions.py": b"def test_filter():\n    pass\n",
            "python/pyspark/sql/worker/runner.py": b"def run():\n    pass\n",
            "examples/src/main/python/foo.py": b"print('hi')\n",
            "examples/src/main/data/asset.bin": b"binary",
        },
        "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
    )
    resolver = SparkSourceResolver(_config(), tmp_path / "cache")
    manifest = resolver.build_manifest(root)

    rel_paths = {f.relative_path for f in manifest.files}
    # docs/api excluded; data dir excluded; guides/api/examples included
    assert "docs/sql-guide.md" in rel_paths
    assert "docs/api/index.html" not in rel_paths
    assert "python/pyspark/sql/functions.py" in rel_paths
    assert "examples/src/main/python/foo.py" in rel_paths
    assert not any("data/" in p for p in rel_paths)
    # tests and worker internals excluded from api stream
    assert "python/pyspark/sql/tests/test_functions.py" not in rel_paths
    assert "python/pyspark/sql/worker/runner.py" not in rel_paths
    # Languages
    by_path = {f.relative_path: f for f in manifest.files}
    assert by_path["docs/sql-guide.md"].doc_type == "guide"
    assert by_path["python/pyspark/sql/functions.py"].doc_type == "api_reference"
    assert by_path["examples/src/main/python/foo.py"].doc_type == "code_example"
    assert by_path["examples/src/main/python/foo.py"].language == "python"


def test_manifest_is_deterministic(tmp_path) -> None:
    root = _write_archive(
        tmp_path / "cache",
        {
            "docs/a.md": b"# A\n",
            "python/pyspark/z.py": b"x = 1\n",
        },
        "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
    )
    resolver = SparkSourceResolver(_config(), tmp_path / "cache")
    m1 = resolver.build_manifest(root)
    m2 = resolver.build_manifest(root)
    assert m1.manifest_hash == m2.manifest_hash
    assert [f.relative_path for f in m1.files] == [f.relative_path for f in m2.files]


def test_reject_missing_root(tmp_path) -> None:
    resolver = SparkSourceResolver(_config(), tmp_path / "cache")
    with pytest.raises(ValueError, match="root"):
        resolver.build_manifest(tmp_path / "does-not-exist")


def test_materialize_reuses_cache(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    payloads = {"docs/a.md": b"# A\n"}
    commit = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"
    _write_archive(cache_dir, payloads, commit)
    resolver = SparkSourceResolver(_config(), cache_dir)

    # If the cache marker is valid, materialize() reuses it without downloading.
    called = {"download": False}

    def _fake_download(url, dest):
        called["download"] = True
        dest.write_bytes(_make_archive(payloads))

    monkeypatch.setattr(resolver, "_download", _fake_download)
    root = resolver.materialize()
    assert called["download"] is False
    assert root.exists()
    assert (root / ".spark_commit").read_text() == commit


def test_materialize_downloads_when_missing(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    commit = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"
    # GitHub commit archives have a single top-level directory
    # (e.g. apache-spark-<sha>/). Match that shape.
    payloads = {"apache-spark-123/docs/a.md": b"# A\n"}
    resolver = SparkSourceResolver(_config(), cache_dir)

    def _fake_download(url, dest):
        dest.write_bytes(_make_archive(payloads))

    monkeypatch.setattr(resolver, "_download", _fake_download)
    root = resolver.materialize()
    assert root.exists()
    assert (root / "docs/a.md").read_text() == "# A\n"
    assert (root / ".spark_commit").read_text() == commit


def test_safe_extract_rejects_traversal(tmp_path) -> None:
    resolver = SparkSourceResolver(_config(), tmp_path / "cache")
    evil = tarfile.TarInfo(name="../../evil.txt")
    evil.size = 5
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.addfile(evil, io.BytesIO(b"evil!"))
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(RuntimeError, match="Unsafe path"):
        resolver._safe_extract(archive, tmp_path / "out")


def test_source_url_uses_commit(tmp_path) -> None:
    root = _write_archive(
        tmp_path / "cache",
        {"docs/a.md": b"# A\n"},
        "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
    )
    resolver = SparkSourceResolver(_config(), tmp_path / "cache")
    manifest = resolver.build_manifest(root)
    assert manifest.files[0].source_url.startswith(
        "https://raw.githubusercontent.com/apache/spark/fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4/"
    )


def test_content_requires_filters_files_by_content(tmp_path) -> None:
    """A stream with ``content_requires`` only includes files containing every needle."""
    config = SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(
            SparkStreamConfig(
                name="sql_functions",
                doc_type="sql_function_ref",
                include=("sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/**/*.scala",),
                exclude=("**/codegen/**",),
                language="scala",
                chunking="code",
                content_requires=("ExpressionDescription",),
            ),
        ),
    )
    root = _write_archive(
        tmp_path / "cache",
        {
            "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/functions.scala": (
                b'@ExpressionDescription(\n  usage = "_FUNC_(x)")\ncase class Foo(x: Expression)\n'
            ),
            "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/helpers.scala": (
                b"object Helper {\n  val x = 1\n}\n"
            ),
            "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/codegen/Generated.scala": (
                b'@ExpressionDescription(\n  usage = "_FUNC_(x)")\ncase class Generated(x: Expression)\n'
            ),
        },
        "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
    )
    resolver = SparkSourceResolver(config, tmp_path / "cache")
    manifest = resolver.build_manifest(root)

    rel_paths = {f.relative_path for f in manifest.files}
    assert "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/functions.scala" in rel_paths
    assert "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/helpers.scala" not in rel_paths
    assert not any("codegen" in p for p in rel_paths)
    by_path = {f.relative_path: f for f in manifest.files}
    assert by_path[
        "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/functions.scala"
    ].doc_type == ("sql_function_ref")
    assert (
        by_path["sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/functions.scala"].language
        == "scala"
    )


def test_content_requires_requires_all_needles(tmp_path) -> None:
    config = SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(
            SparkStreamConfig(
                name="sql_functions",
                doc_type="sql_function_ref",
                include=("sql/**/*.scala",),
                exclude=(),
                language="scala",
                chunking="code",
                content_requires=("ExpressionDescription", "case class"),
            ),
        ),
    )
    root = _write_archive(
        tmp_path / "cache",
        {
            "sql/a.scala": b"@ExpressionDescription()\ncase class A(x: Expression)\n",
            "sql/b.scala": b"@ExpressionDescription()\nobject B { }\n",
        },
        "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
    )
    resolver = SparkSourceResolver(config, tmp_path / "cache")
    manifest = resolver.build_manifest(root)
    rel_paths = {f.relative_path for f in manifest.files}
    assert "sql/a.scala" in rel_paths
    assert "sql/b.scala" not in rel_paths
