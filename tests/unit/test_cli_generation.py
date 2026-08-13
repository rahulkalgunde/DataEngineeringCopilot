"""Phase 2 tests: pinned generation (gen-*) CLI behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_engineering_copilot import cli

_GEN = "pinned-abc123def456"


def _patch_settings(monkeypatch, tmp_path):
    from tests.conftest import make_settings

    index_state_dir = tmp_path / ".index_state"
    index_state_dir.mkdir(parents=True, exist_ok=True)
    pinned_corpus_dir = tmp_path / "pinned_corpus"
    settings = make_settings(
        index_state_dir=index_state_dir,
        pinned_sources_path=tmp_path / "pinned_sources.json",
        pinned_cache_dir=tmp_path / "pinned_cache",
        pinned_corpus_dir=pinned_corpus_dir,
        spark_corpus_dir=tmp_path / "spark_corpus",
    )
    monkeypatch.setattr(cli, "settings", settings)
    return settings


def _write_pinned_config(tmp_path, sources=None) -> Path:
    if sources is None:
        sources = [
            {
                "type": "github",
                "name": "Apache Airflow Documentation",
                "slug": "airflow",
                "version": "3.3.1",
                "license": "Apache-2.0",
                "repository": "https://github.com/apache/airflow.git",
                "ref": "3.3.1",
                "commit": "3adbbe1c58e4532df1964cb7794805e763816ee8",
                "streams": [
                    {
                        "name": "core-docs",
                        "doc_type": "guide",
                        "include": ["airflow-core/docs/**/*.rst"],
                        "exclude": [],
                        "language": "rst",
                        "chunking": "header_aware",
                    }
                ],
            }
        ]
    config_path = tmp_path / "pinned_sources.json"
    config_path.write_text(json.dumps({"sources": sources}))
    return config_path


def test_gen_config_check_valid(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    _write_pinned_config(tmp_path)
    assert cli.gen_config_check() == 0


def test_gen_config_check_invalid(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    (tmp_path / "pinned_sources.json").write_text("{not-json")
    assert cli.gen_config_check() == 1


def test_default_generation_prefixes_pinned_and_tracks_config(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    _write_pinned_config(tmp_path)
    baseline = cli._default_generation()
    assert baseline.startswith("pinned-")

    _write_pinned_config(
        tmp_path,
        sources=[
            {
                "type": "github",
                "name": "Delta Lake Documentation",
                "slug": "delta",
                "version": "v4.3.1",
                "license": "Apache-2.0",
                "repository": "https://github.com/delta-io/delta.git",
                "ref": "v4.3.1",
                "commit": "54ce02692567fc3e5107a3ae69ebeccdffa7edfe",
                "streams": [
                    {
                        "name": "docs",
                        "doc_type": "guide",
                        "include": ["docs/src/content/docs/**/*.mdx"],
                        "exclude": [],
                        "language": "mdx",
                        "chunking": "header_aware",
                    }
                ],
            }
        ],
    )
    updated = cli._default_generation()
    assert updated.startswith("pinned-")
    assert updated != baseline


def test_gen_manifest_writes_combined_and_per_source(monkeypatch, tmp_path) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    _write_pinned_config(tmp_path)

    canned = [
        {
            "slug": "airflow",
            "type": "github",
            "name": "Apache Airflow Documentation",
            "commit": "3adbbe1c58e4532df1964cb7794805e763816ee8",
            "files": [
                {
                    "stream": "core-docs",
                    "relative_path": "airflow-core/docs/start.rst",
                    "doc_type": "guide",
                    "language": "rst",
                    "source_url": "https://raw.githubusercontent.com/apache/airflow/main/start.rst",
                }
            ],
        }
    ]
    monkeypatch.setattr(cli, "_resolve_pinned_sources", lambda: canned)

    result = cli.gen_manifest()
    assert result == 0
    artifact_root = settings.pinned_corpus_dir / cli._default_generation()
    assert (artifact_root / "manifest-airflow.json").is_file()
    assert (artifact_root / "manifest.json").is_file()
    combined = json.loads((artifact_root / "manifest.json").read_text())
    assert combined["files"] == [{"relative_path": "airflow-core/docs/start.rst"}]


def test_gen_manifest_resolver_error_returns_5(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    _write_pinned_config(tmp_path)

    def _boom():
        raise RuntimeError("materialize failed")

    monkeypatch.setattr(cli, "_resolve_pinned_sources", _boom)
    assert cli.gen_manifest() == 5


def test_gen_validate_rejects_bad_identifier(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    assert cli.gen_validate("bad id!") == 2


def test_gen_validate_requires_artifacts(monkeypatch, tmp_path) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    _write_pinned_config(tmp_path)
    (settings.pinned_corpus_dir / _GEN).mkdir(parents=True, exist_ok=True)
    assert cli.gen_validate(_GEN) == 3


def test_gen_activate_requires_validation_report(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    assert cli.gen_activate(_GEN) == 3


def test_gen_activate_forces_and_writes_state(monkeypatch, tmp_path) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    report_path = cli._validation_report_path(_GEN)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generation": _GEN, "passed": True}))
    monkeypatch.setenv("FORCE", "1")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "_qdrant_change_alias", lambda gen: None)
        result = cli.gen_activate(_GEN)

    assert result == 0
    active = json.loads((settings.index_state_dir / "active.json").read_text())
    assert active["generation"] == _GEN
    assert active["collection"] == cli._spark_generation_collection(_GEN)


def test_gen_activate_failure_returns_5(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    report_path = cli._validation_report_path(_GEN)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generation": _GEN, "passed": True}))
    monkeypatch.setenv("FORCE", "1")

    def _fail(gen):
        raise RuntimeError("qdrant down")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "_qdrant_change_alias", _fail)
        result = cli.gen_activate(_GEN)
    assert result == 5


def test_gen_rollback_requires_active_generation(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    assert cli.gen_rollback(_GEN) == 4


def test_gen_rollback_to_previous_generation(monkeypatch, tmp_path) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    history = settings.index_state_dir / "history.jsonl"
    history.write_text(
        json.dumps({"generation": "pinned-old", "collection": "data_engineering_docs__pinned-old"})
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
        result = cli.gen_rollback(_GEN)

    assert result == 0
    assert target["gen"] == "pinned-old"
