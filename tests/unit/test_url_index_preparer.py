"""Tests for the local_mirror resolution path in UrlIndexPreparer."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from data_engineering_copilot.config.settings import PinnedSourceConfig
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.url_index_preparer import UrlIndexPreparer


@pytest.mark.asyncio
async def test_local_mirror_prepare_stamps_license_and_commit(tmp_path: Path) -> None:
    mirror_root = tmp_path / "mirror"
    site_dir = mirror_root / "platform"
    site_dir.mkdir(parents=True)
    (site_dir / "llms.txt").write_text(
        "- [Working with messages](https://platform.claude.com/docs/en/build-with-claude/working-with-messages.md)\n"
    )
    (site_dir / "build-with-claude").mkdir()
    (site_dir / "build-with-claude" / "working-with-messages.md").write_text(
        "# Working with messages\n\n" + ("Substantive body text. " * 40)
    )
    subprocess.run(["git", "init", "-q", str(mirror_root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(mirror_root), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(mirror_root), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "m"],
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "-C", str(mirror_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    config = PinnedSourceConfig(
        type="local_mirror",
        name="Claude Platform Docs",
        slug="claude-platform",
        version="2026-08-17",
        license="MIT",
        mirror_dir="platform",
        commit=head,
        url_prefix="https://platform.claude.com/docs/en/",
        base_url="https://platform.claude.com/docs",
        doc_type="guide",
    )
    chunker = HeaderAwareChunker(chunk_size_words=100, overlap_words=10, min_chunk_words=5)
    preparer = UrlIndexPreparer(
        config,
        tmp_path / "cache",
        "gen-1",
        mirror_root=mirror_root,
        header_chunker=chunker,
    )

    package = await preparer.prepare()
    assert package.commit == head
    assert package.chunks, "mirror content must produce chunks"
    for chunk in package.chunks:
        assert chunk.source_commit == head
        assert chunk.license == "MIT"
    assert package.coverage[0].status == "indexed"


@pytest.mark.asyncio
async def test_local_mirror_prepare_requires_mirror_root(tmp_path: Path) -> None:
    config = PinnedSourceConfig(
        type="local_mirror",
        name="Claude Platform Docs",
        slug="claude-platform",
        version="2026-08-17",
        license="MIT",
        mirror_dir="platform",
        commit="c" * 40,
        url_prefix="https://platform.claude.com/docs/en/",
        doc_type="guide",
    )
    preparer = UrlIndexPreparer(config, tmp_path / "cache", "gen-1")
    with pytest.raises(ValueError, match="mirror_root"):
        await preparer.prepare()
