"""Phase 10 tests: spark activation/rollback CLI behavior and cache scoping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_engineering_copilot import cli
from data_engineering_copilot.domain.models import CacheScope
from data_engineering_copilot.services.query_cache import scope_fingerprint

_GEN = "spark-4.0.0-fa33ea00-abc123"


def _patch_settings(monkeypatch, tmp_path):
    from tests.conftest import make_settings

    index_state_dir = tmp_path / ".index_state"
    index_state_dir.mkdir(parents=True, exist_ok=True)
    settings = make_settings(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
        collection_name="data_engineering_docs",
        index_state_dir=index_state_dir,
    )
    monkeypatch.setattr(cli, "settings", settings)
    return settings


def test_activate_requires_validation_report(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    # No validation report exists -> must refuse with exit code 3.
    result = cli.spark_activate(_GEN)
    assert result == 3


def test_activate_with_corrupt_report(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    report_path = cli._validation_report_path(_GEN)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{not-json")
    result = cli.spark_activate(_GEN)
    assert result == 3


def test_activate_refuses_when_report_not_passed(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    report_path = cli._validation_report_path(_GEN)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generation": _GEN, "passed": False}))
    result = cli.spark_activate(_GEN)
    assert result == 3


def test_activate_requires_confirmation(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    report_path = cli._validation_report_path(_GEN)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generation": _GEN, "passed": True}))
    # FORCE not set, stdin answers "n" -> aborted (exit 0, alias unchanged).
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    result = cli.spark_activate(_GEN)
    assert result == 0
    assert not (cli.settings.index_state_dir / "active.json").exists()


def test_activate_forces_and_writes_state(monkeypatch, tmp_path) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    report_path = cli._validation_report_path(_GEN)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generation": _GEN, "passed": True}))
    monkeypatch.setenv("FORCE", "1")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "_qdrant_change_alias", lambda gen: None)
        result = cli.spark_activate(_GEN)

    assert result == 0
    active = json.loads((settings.index_state_dir / "active.json").read_text())
    assert active["generation"] == _GEN
    assert active["collection"] == cli._spark_generation_collection(_GEN)


def test_activate_failure_returns_5(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    report_path = cli._validation_report_path(_GEN)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generation": _GEN, "passed": True}))
    monkeypatch.setenv("FORCE", "1")

    def _fail(gen):
        raise RuntimeError("qdrant down")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "_qdrant_change_alias", _fail)
        result = cli.spark_activate(_GEN)
    assert result == 5


def test_rollback_requires_active_generation(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    # No active state -> rollback of any generation returns 4.
    result = cli.spark_rollback(_GEN)
    assert result == 4


def test_rollback_to_previous_generation(monkeypatch, tmp_path) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    history = settings.index_state_dir / "history.jsonl"
    history.write_text(
        json.dumps({"generation": "spark-4.0.0-old", "collection": "docs__old"})
        + "\n"
        + json.dumps({"generation": _GEN, "collection": cli._spark_generation_collection(_GEN)})
        + "\n"
    )
    settings.index_state_dir.joinpath("active.json").write_text(
        json.dumps({"generation": _GEN, "collection": cli._spark_generation_collection(_GEN)})
    )
    monkeypatch.setenv("FORCE", "1")

    target = {}

    def _alias(gen):
        target["gen"] = gen

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "_qdrant_change_alias", _alias)
        result = cli.spark_rollback(_GEN)

    assert result == 0
    assert target["gen"] == "spark-4.0.0-old"


# ------------------------------------------------------------------
# Cache generation scoping
# ------------------------------------------------------------------


_CONFIG_STREAMS = [
    {
        "name": "guides",
        "doc_type": "guide",
        "include": ["docs/**"],
        "exclude": ["docs/api/**"],
        "language": "markdown",
        "chunking": "header_aware",
    }
]


def _write_spark_config(tmp_path, streams) -> str:
    config_path = tmp_path / "spark_sources.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "apache-spark",
                "repository": "https://github.com/apache/spark.git",
                "ref": "v4.0.0",
                "commit": "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
                "license": "Apache-2.0",
                "streams": streams,
            }
        )
    )
    return str(config_path)


def _patch_spark_config(monkeypatch, tmp_path, streams) -> str:
    from tests.conftest import make_settings

    config_path = _write_spark_config(tmp_path, streams)
    settings = make_settings(spark_sources_path=Path(config_path))
    monkeypatch.setattr(cli, "settings", settings)
    return config_path


def test_default_generation_hashes_embedding_and_config(monkeypatch, tmp_path) -> None:
    _patch_spark_config(monkeypatch, tmp_path, _CONFIG_STREAMS)

    generation = cli._default_spark_generation()
    assert generation.startswith("spark-v4.0.0-fa33ea00-")


def test_default_generation_changes_when_config_changes(monkeypatch, tmp_path) -> None:
    _patch_spark_config(monkeypatch, tmp_path, _CONFIG_STREAMS)

    baseline = cli._default_spark_generation()

    streams_with_functions = _CONFIG_STREAMS + [
        {
            "name": "sql_functions",
            "doc_type": "sql_function_ref",
            "include": ["sql/catalyst/**/expressions/**/*.scala"],
            "exclude": [],
            "language": "scala",
            "chunking": "code",
            "content_requires": ["ExpressionDescription"],
        }
    ]
    _patch_spark_config(monkeypatch, tmp_path, streams_with_functions)

    updated = cli._default_spark_generation()
    assert updated != baseline
    assert updated.startswith("spark-v4.0.0-fa33ea00-")


def test_cache_scope_fingerprint_includes_generation() -> None:
    a = CacheScope(collection_name="docs", index_generation="spark-4.0.0")
    b = CacheScope(collection_name="docs", index_generation="spark-4.0.1")
    assert scope_fingerprint(a) != scope_fingerprint(b)


def test_cache_scope_legacy_empty_generation() -> None:
    a = CacheScope(collection_name="docs", index_generation="")
    b = CacheScope(collection_name="docs")
    assert scope_fingerprint(a) == scope_fingerprint(b)
