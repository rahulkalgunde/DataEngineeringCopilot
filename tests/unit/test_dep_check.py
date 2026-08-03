"""Tests for container dependency-freshness detection."""

import hashlib

import pytest

from data_engineering_copilot.infrastructure.dep_check import (
    FINGERPRINT_FILES,
    check_deps,
    deps_detail,
    fingerprint_ok,
)


def _write_fingerprint(dir_path, content):
    fp = dir_path / "image_deps_sha256.txt"
    fp.write_text(content, encoding="utf-8")
    return fp


def _live_hash(dir_path):
    parts = [(dir_path / name).read_bytes() for name in FINGERPRINT_FILES]
    return hashlib.sha256(b"".join(parts)).hexdigest()


class TestFingerprintOk:
    def test_returns_none_when_no_baked_file(self, tmp_path):
        live = tmp_path / "live"
        live.mkdir()
        assert fingerprint_ok(baked_path=str(tmp_path / "missing.txt"), live_dir=str(live)) is None

    def test_returns_none_when_live_files_missing(self, tmp_path):
        baked = tmp_path / "baked"
        baked.mkdir()
        _write_fingerprint(baked, "abc")
        empty = tmp_path / "empty"
        empty.mkdir()
        assert fingerprint_ok(baked_path=str(baked / "image_deps_sha256.txt"), live_dir=str(empty)) is None

    def test_returns_true_when_hashes_match(self, tmp_path):
        baked = tmp_path / "baked"
        baked.mkdir()
        live = tmp_path / "live"
        live.mkdir()
        (live / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (live / "uv.lock").write_text("lock v1\n", encoding="utf-8")
        _write_fingerprint(baked, _live_hash(live))
        assert fingerprint_ok(baked_path=str(baked / "image_deps_sha256.txt"), live_dir=str(live)) is True

    def test_returns_false_when_deps_changed(self, tmp_path):
        baked = tmp_path / "baked"
        baked.mkdir()
        live = tmp_path / "live"
        live.mkdir()
        (live / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (live / "uv.lock").write_text("lock v1\n", encoding="utf-8")
        _write_fingerprint(baked, _live_hash(live))
        (live / "pyproject.toml").write_text('[project]\ndependencies = ["new-pkg"]\n', encoding="utf-8")
        assert fingerprint_ok(baked_path=str(baked / "image_deps_sha256.txt"), live_dir=str(live)) is False


class TestDepsDetail:
    def test_indeterminate_when_not_in_container(self, tmp_path):
        result = deps_detail(baked_path=str(tmp_path / "missing.txt"), live_dir=str(tmp_path))
        assert result.ok is None
        assert result.in_container is False
        assert result.message is None

    def test_fresh_when_hashes_match(self, tmp_path):
        baked = tmp_path / "baked"
        baked.mkdir()
        live = tmp_path / "live"
        live.mkdir()
        (live / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (live / "uv.lock").write_text("lock v1\n", encoding="utf-8")
        _write_fingerprint(baked, _live_hash(live))
        result = deps_detail(baked_path=str(baked / "image_deps_sha256.txt"), live_dir=str(live))
        assert result.ok is True
        assert result.in_container is True
        assert result.baked_hash is not None
        assert result.live_hash is not None
        assert result.message is None

    def test_stale_when_hashes_differ(self, tmp_path):
        baked = tmp_path / "baked"
        baked.mkdir()
        live = tmp_path / "live"
        live.mkdir()
        (live / "pyproject.toml").write_text("a", encoding="utf-8")
        (live / "uv.lock").write_text("b", encoding="utf-8")
        _write_fingerprint(baked, "stale-hash")
        result = deps_detail(baked_path=str(baked / "image_deps_sha256.txt"), live_dir=str(live))
        assert result.ok is False
        assert result.in_container is True
        assert result.baked_hash == "stale-hash"
        assert result.live_hash is not None
        assert result.message is not None
        assert "STALE" in result.message


class TestCheckDeps:
    def test_returns_true_when_indeterminate(self, tmp_path):
        assert check_deps(baked_path=str(tmp_path / "missing.txt"), live_dir=str(tmp_path)) is True

    def test_returns_false_when_stale_and_not_fail_fast(self, tmp_path, caplog):
        baked = tmp_path / "baked"
        baked.mkdir()
        live = tmp_path / "live"
        live.mkdir()
        (live / "pyproject.toml").write_text("a", encoding="utf-8")
        (live / "uv.lock").write_text("b", encoding="utf-8")
        _write_fingerprint(baked, "not-the-live-hash")
        assert (
            check_deps(
                fail_fast=False,
                baked_path=str(baked / "image_deps_sha256.txt"),
                live_dir=str(live),
            )
            is False
        )
        assert "STALE" in caplog.text

    def test_stale_fail_fast_raises_systemexit(self, tmp_path, caplog):
        baked = tmp_path / "baked"
        baked.mkdir()
        live = tmp_path / "live"
        live.mkdir()
        (live / "pyproject.toml").write_text("a", encoding="utf-8")
        (live / "uv.lock").write_text("b", encoding="utf-8")
        _write_fingerprint(baked, "not-the-live-hash")
        with pytest.raises(SystemExit):
            check_deps(
                fail_fast=True,
                baked_path=str(baked / "image_deps_sha256.txt"),
                live_dir=str(live),
            )
        assert "STALE" in caplog.text
