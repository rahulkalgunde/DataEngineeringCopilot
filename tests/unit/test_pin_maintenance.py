"""Phase 2 tests: generation maintenance classification + gen-stale CLI."""

from __future__ import annotations

import json

from data_engineering_copilot import cli
from data_engineering_copilot.services.pin_maintenance import (
    GenerationStatus,
    classify_generations,
    is_generation_collection,
    local_generation_collections,
)


def test_is_generation_collection() -> None:
    assert is_generation_collection("data_engineering_docs__pinned-x")
    assert not is_generation_collection("data_engineering_docs")
    assert not is_generation_collection("unrelated")


def test_classify_active_stale_orphan() -> None:
    collections = [
        "data_engineering_docs__pinned-a",
        "data_engineering_docs__pinned-b",
        "data_engineering_docs__pinned-c",
        "data_engineering_docs",
    ]
    local = {"data_engineering_docs__pinned-a", "data_engineering_docs__pinned-b"}

    statuses = classify_generations(collections, active_generation="pinned-a", local_collection_names=local)

    by_name = {s.name: s.state for s in statuses}
    assert by_name == {
        "data_engineering_docs__pinned-a": "active",
        "data_engineering_docs__pinned-b": "stale",
        "data_engineering_docs__pinned-c": "orphan",
    }


def test_classify_no_active_generation() -> None:
    statuses = classify_generations(
        ["data_engineering_docs__pinned-a"],
        active_generation=None,
        local_collection_names={"data_engineering_docs__pinned-a"},
    )
    assert statuses == [GenerationStatus(name="data_engineering_docs__pinned-a", state="stale")]


def test_local_generation_collections_scans_corpus_dirs(tmp_path) -> None:
    spark = tmp_path / "spark_corpus"
    pinned = tmp_path / "pinned_corpus"
    (spark / "spark-1").mkdir(parents=True)
    (spark / "spark-1" / "chunks.jsonl").write_text("{}")
    (spark / "spark-2").mkdir(parents=True)
    (pinned / "pinned-1").mkdir(parents=True)
    (pinned / "pinned-1" / "chunks.jsonl").write_text("{}")
    (pinned / "pinned-2").mkdir(parents=True)

    local = local_generation_collections([spark, pinned])

    assert local == {"data_engineering_docs__spark-1", "data_engineering_docs__pinned-1"}


def _patch_settings(monkeypatch, tmp_path):
    from tests.conftest import make_settings

    index_state_dir = tmp_path / ".index_state"
    index_state_dir.mkdir(parents=True, exist_ok=True)
    settings = make_settings(
        index_state_dir=index_state_dir,
        spark_corpus_dir=tmp_path / "spark_corpus",
        pinned_corpus_dir=tmp_path / "pinned_corpus",
    )
    monkeypatch.setattr(cli, "settings", settings)
    return settings


def test_gen_stale_reports_active_stale_and_orphan(monkeypatch, tmp_path, capsys) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    spark = settings.spark_corpus_dir
    (spark / "spark-1").mkdir(parents=True)
    (spark / "spark-1" / "chunks.jsonl").write_text("{}")
    settings.index_state_dir.joinpath("active.json").write_text(
        json.dumps({"generation": "spark-1", "collection": "data_engineering_docs__spark-1"})
    )
    monkeypatch.setattr(
        cli,
        "_list_qdrant_collections",
        lambda: ["data_engineering_docs__spark-1", "data_engineering_docs__pinned-9", "unrelated"],
    )

    result = cli.gen_stale()

    assert result == 0
    out = capsys.readouterr().out
    assert "✅ active" in out and "data_engineering_docs__spark-1" in out
    assert "⚪ orphan" in out and "data_engineering_docs__pinned-9" in out


def test_gen_stale_no_collections(monkeypatch, tmp_path, capsys) -> None:
    _patch_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_list_qdrant_collections", lambda: ["unrelated"])
    result = cli.gen_stale()
    assert result == 0
    assert "No generation collections" in capsys.readouterr().out


def test_gen_stale_collection_list_failure(monkeypatch, tmp_path) -> None:
    _patch_settings(monkeypatch, tmp_path)

    def _boom():
        raise TimeoutError("qdrant unreachable")

    monkeypatch.setattr(cli, "_list_qdrant_collections", _boom)
    assert cli.gen_stale() == 5


def test_gen_stale_reports_pinned_local_artifacts(monkeypatch, tmp_path) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    pinned = settings.pinned_corpus_dir
    (pinned / "pinned-2").mkdir(parents=True)
    (pinned / "pinned-2" / "chunks.jsonl").write_text("{}")
    monkeypatch.setattr(cli, "_list_qdrant_collections", lambda: ["data_engineering_docs__pinned-2"])
    assert cli.gen_stale() == 0
