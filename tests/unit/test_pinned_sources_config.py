"""Phase 2 tests: pinned multi-source configuration loading and validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data_engineering_copilot.config.settings import (
    PinnedSourceConfig,
    PinnedStreamConfig,
    load_pinned_sources,
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "pinned_sources.json"
    path.write_text(json.dumps(payload))
    return path


def _github_source() -> dict:
    return {
        "type": "github",
        "name": "Apache Airflow Documentation",
        "slug": "airflow",
        "version": "3.3.1",
        "license": "Apache-2.0",
        "repository": "https://github.com/apache/airflow.git",
        "ref": "3.3.1",
        "commit": "1111111111111111111111111111111111111111",
        "streams": [
            {
                "name": "docs",
                "doc_type": "guide",
                "include": ["docs/apache-airflow/**/*.rst"],
                "exclude": [],
                "language": "conceptual",
                "chunking": "header_aware",
            },
        ],
    }


def _url_index_source() -> dict:
    return {
        "type": "url_index",
        "name": "Claude Platform Docs",
        "slug": "claude-platform",
        "version": "unpinned",
        "index_url": "https://platform.claude.com/docs/llms.txt",
        "url_prefix": "https://platform.claude.com/docs/en/",
        "base_url": "https://platform.claude.com/docs",
        "cache_dir": "claude-platform",
        "doc_type": "guide",
    }


def _local_mirror_source() -> dict:
    return {
        "type": "local_mirror",
        "name": "Claude Platform Docs",
        "slug": "claude-platform",
        "version": "2026-08-17",
        "license": "MIT",
        "mirror_dir": "platform",
        "commit": "c2c813e171cb8d8c5f76bf1034aaf94304c267c8",
        "url_prefix": "https://platform.claude.com/docs/en/",
        "base_url": "https://platform.claude.com/docs",
        "doc_type": "guide",
    }


def test_load_valid_mixed_config(tmp_path) -> None:
    mirror = _local_mirror_source()
    mirror["name"] = "Claude Platform Docs (mirror)"
    mirror["slug"] = "claude-platform-mirror"
    path = _write_config(tmp_path, {"sources": [_github_source(), _url_index_source(), mirror]})
    sources = load_pinned_sources(path)
    assert len(sources) == 3

    github = sources[0]
    assert isinstance(github, PinnedSourceConfig)
    assert github.type == "github"
    assert github.slug == "airflow"
    assert github.ref == "3.3.1"
    assert github.commit == "1111111111111111111111111111111111111111"
    assert len(github.streams) == 1
    assert isinstance(github.streams[0], PinnedStreamConfig)

    url_index = sources[1]
    assert url_index.type == "url_index"
    assert url_index.index_url == "https://platform.claude.com/docs/llms.txt"
    assert url_index.cache_dir == "claude-platform"

    mirror = sources[2]
    assert mirror.type == "local_mirror"
    assert mirror.mirror_dir == "platform"
    assert mirror.commit == "c2c813e171cb8d8c5f76bf1034aaf94304c267c8"
    assert mirror.license == "MIT"


def test_local_mirror_requires_40hex_commit(tmp_path) -> None:
    src = _local_mirror_source()
    src["commit"] = "not-a-sha"
    with pytest.raises(ValueError, match="commit"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src]}))


def test_local_mirror_requires_mirror_dir_and_license(tmp_path) -> None:
    src = _local_mirror_source()
    del src["mirror_dir"]
    with pytest.raises(ValueError, match="mirror_dir"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src]}))
    src = _local_mirror_source()
    del src["license"]
    with pytest.raises(ValueError, match="license"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src]}))


def test_duplicate_slugs_raise(tmp_path) -> None:
    src = _github_source()
    dup = copy.deepcopy(src)
    dup["name"] = "Apache Airflow Documentation (mirror)"
    with pytest.raises(ValueError, match="slug"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src, dup]}))


def test_duplicate_names_raise(tmp_path) -> None:
    src = _github_source()
    other = _url_index_source()
    other["slug"] = "claude-alt"
    payload = {"sources": [src, other]}
    payload["sources"][1]["name"] = src["name"]
    with pytest.raises(ValueError, match="name"):
        load_pinned_sources(_write_config(tmp_path, payload))


def test_invalid_type_raises(tmp_path) -> None:
    src = _url_index_source()
    src["type"] = "crawler"
    with pytest.raises(ValueError, match="type"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src]}))


def test_invalid_commit_raises(tmp_path) -> None:
    src = _github_source()
    src["commit"] = "not-a-sha"
    with pytest.raises(ValueError, match="commit"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src]}))


def test_invalid_repository_raises(tmp_path) -> None:
    src = _github_source()
    src["repository"] = "ftp://example.com/repo"
    with pytest.raises(ValueError, match="repository"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src]}))


def test_invalid_stream_doc_type_raises(tmp_path) -> None:
    src = _github_source()
    src["streams"][0]["doc_type"] = "invalid_type"
    with pytest.raises(ValueError, match="doc_type"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src]}))


def test_url_index_missing_required_raises(tmp_path) -> None:
    src = _url_index_source()
    del src["index_url"]
    with pytest.raises(ValueError, match="index_url"):
        load_pinned_sources(_write_config(tmp_path, {"sources": [src]}))


def test_empty_sources_raise(tmp_path) -> None:
    with pytest.raises(ValueError, match="sources"):
        load_pinned_sources(_write_config(tmp_path, {"sources": []}))
