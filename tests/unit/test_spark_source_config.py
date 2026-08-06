"""Phase 2 tests: Spark source configuration loading and validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data_engineering_copilot.config.settings import (
    SparkSourceConfig,
    SparkStreamConfig,
    load_spark_source_config,
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "spark_sources.json"
    path.write_text(json.dumps(payload))
    return path


_VALID_STREAMS = [
    {
        "name": "guides",
        "doc_type": "guide",
        "include": ["docs/**/*.md"],
        "exclude": ["docs/api/**"],
        "language": "conceptual",
        "chunking": "header_aware",
    },
    {
        "name": "api",
        "doc_type": "api_reference",
        "include": ["python/pyspark/**/*.py"],
        "exclude": [],
        "language": "python",
        "chunking": "api",
    },
    {
        "name": "examples",
        "doc_type": "code_example",
        "include": ["examples/src/main/**/*.py"],
        "exclude": ["**/data/**"],
        "language": "mixed",
        "chunking": "code",
    },
]


def _valid_config() -> dict:
    return {
        "name": "Apache Spark 4.0.0",
        "repository": "https://github.com/apache/spark.git",
        "ref": "v4.0.0",
        "commit": "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        "license": "Apache-2.0",
        "streams": copy.deepcopy(_VALID_STREAMS),
    }


def test_load_valid_config(tmp_path) -> None:
    path = _write_config(tmp_path, _valid_config())
    config = load_spark_source_config(path)
    assert isinstance(config, SparkSourceConfig)
    assert config.name == "Apache Spark 4.0.0"
    assert config.commit == "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"
    assert len(config.streams) == 3
    assert isinstance(config.streams[0], SparkStreamConfig)


def test_invalid_commit_raises(tmp_path) -> None:
    cfg = _valid_config()
    cfg["commit"] = "not-a-sha"
    with pytest.raises(ValueError, match="commit"):
        load_spark_source_config(_write_config(tmp_path, cfg))


def test_invalid_repository_raises(tmp_path) -> None:
    cfg = _valid_config()
    cfg["repository"] = "ftp://example.com/repo"
    with pytest.raises(ValueError, match="repository"):
        load_spark_source_config(_write_config(tmp_path, cfg))


def test_invalid_doc_type_raises(tmp_path) -> None:
    cfg = _valid_config()
    cfg["streams"][0]["doc_type"] = "invalid_type"
    with pytest.raises(ValueError, match="doc_type"):
        load_spark_source_config(_write_config(tmp_path, cfg))


def test_invalid_chunking_raises(tmp_path) -> None:
    cfg = _valid_config()
    cfg["streams"][1]["chunking"] = "invalid_chunking"
    with pytest.raises(ValueError, match="chunking"):
        load_spark_source_config(_write_config(tmp_path, cfg))


def test_duplicate_stream_names_raise(tmp_path) -> None:
    cfg = _valid_config()
    cfg["streams"][1]["name"] = "guides"
    with pytest.raises(ValueError, match="unique"):
        load_spark_source_config(_write_config(tmp_path, cfg))


def test_missing_required_string_raises(tmp_path) -> None:
    cfg = _valid_config()
    del cfg["license"]
    with pytest.raises(ValueError, match="license"):
        load_spark_source_config(_write_config(tmp_path, cfg))


def test_empty_streams_raise(tmp_path) -> None:
    cfg = _valid_config()
    cfg["streams"] = []
    with pytest.raises(ValueError, match="streams"):
        load_spark_source_config(_write_config(tmp_path, cfg))


def test_sql_function_ref_doc_type_accepted(tmp_path) -> None:
    cfg = _valid_config()
    cfg["streams"][1]["doc_type"] = "sql_function_ref"
    cfg["streams"][1]["chunking"] = "code"
    cfg["streams"][1]["language"] = "scala"
    cfg["streams"][1]["include"] = ["sql/catalyst/src/main/scala/**/*.scala"]
    cfg["streams"][1]["content_requires"] = ["ExpressionDescription"]
    config = load_spark_source_config(_write_config(tmp_path, cfg))
    stream = config.streams[1]
    assert stream.doc_type == "sql_function_ref"
    assert stream.content_requires == ("ExpressionDescription",)


def test_content_requires_must_be_string_list(tmp_path) -> None:
    cfg = _valid_config()
    cfg["streams"][1]["content_requires"] = ["ExpressionDescription", 42]
    with pytest.raises(ValueError, match="content_requires"):
        load_spark_source_config(_write_config(tmp_path, cfg))
