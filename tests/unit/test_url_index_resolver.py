"""Phase 2 tests: config-driven ``llms.txt`` index resolver."""

from __future__ import annotations

import asyncio
import subprocess

import httpx
import pytest

from data_engineering_copilot.config.settings import PinnedSourceConfig
from data_engineering_copilot.services.url_index_resolver import (
    LocalMirrorResolver,
    UrlIndexManifest,
    UrlIndexResolver,
    _url_to_relpath,
)


def _config(cache_dir: str = "claude-platform") -> PinnedSourceConfig:
    return PinnedSourceConfig(
        type="url_index",
        name="Claude Platform Docs",
        slug="claude-platform",
        version="unpinned",
        index_url="https://platform.claude.com/docs/llms.txt",
        url_prefix="https://platform.claude.com/docs/en/",
        base_url="https://platform.claude.com/docs",
        cache_dir=cache_dir,
        doc_type="guide",
    )


def _llms_index() -> str:
    return "\n".join(
        [
            "## Docs",
            "- [Working with messages](https://platform.claude.com/docs/en/build-with-claude/working-with-messages.md)",
            "- [Tool use](https://platform.claude.com/docs/en/build-with-claude/tool-use.md)",
            "- [Raw page](https://platform.claude.com/docs/en/build-with-claude/example.html)",
            "- [External](https://github.com/anthropics/some-repo)",
        ]
    )


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/docs/llms.txt":
            return httpx.Response(200, text=_llms_index())
        if request.url.path.endswith(".md"):
            title = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                text=f"# {title}\n\nBody of {request.url.path}.\n",
                headers={"Content-Type": "text/markdown"},
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _run(coro):
    return asyncio.run(coro)


def test_url_to_relpath_prefix_and_path() -> None:
    assert (
        _url_to_relpath(
            "https://platform.claude.com/docs/en/build-with-claude/tool-use.md",
            "https://platform.claude.com/docs/en/",
        )
        == "build-with-claude/tool-use.md"
    )
    assert _url_to_relpath("https://example.com/x.md", "https://platform.claude.com/docs/en/") == "x.md"


def test_resolve_downloads_and_builds_manifest(tmp_path) -> None:
    resolver = UrlIndexResolver(_config(), tmp_path)
    client = httpx.AsyncClient(transport=_transport())

    async def _run_async() -> None:
        manifest = await resolver.resolve(client=client)
        assert manifest.source_name == "Claude Platform Docs"
        assert manifest.slug == "claude-platform"
        assert len(manifest.entries) == 2
        assert manifest.entries[0].relative_path == "build-with-claude/working-with-messages.md"
        cached = tmp_path / "claude-platform" / "build-with-claude" / "working-with-messages.md"
        assert cached.is_file()
        assert "# working-with-messages.md" in cached.read_text(encoding="utf-8")

    _run(_run_async())
    asyncio.run(client.aclose())


def test_resolve_is_idempotent_and_deterministic(tmp_path) -> None:
    resolver = UrlIndexResolver(_config(), tmp_path)

    async def _run_twice() -> tuple[object, object]:
        client = httpx.AsyncClient(transport=_transport())
        first = await resolver.resolve(client=client)
        second = await resolver.resolve(client=client)
        await client.aclose()
        return first.manifest_hash, second.manifest_hash

    first_hash, second_hash = _run(_run_twice())
    assert first_hash == second_hash


def test_resolve_skips_non_md_and_external_links(tmp_path) -> None:
    resolver = UrlIndexResolver(_config(), tmp_path)

    async def _run_async() -> UrlIndexManifest:
        client = httpx.AsyncClient(transport=_transport())
        manifest = await resolver.resolve(client=client)
        await client.aclose()
        return manifest

    manifest = _run(_run_async())
    relpaths = {e.relative_path for e in manifest.entries}
    assert relpaths == {"build-with-claude/working-with-messages.md", "build-with-claude/tool-use.md"}


def test_resolve_rejects_non_url_index_config(tmp_path) -> None:
    github = PinnedSourceConfig(
        type="github",
        name="x",
        slug="x",
        version="1",
        repository="https://github.com/a/b.git",
        ref="v1",
        commit="a" * 40,
        streams=(),
    )
    with pytest.raises(ValueError, match="url_index"):
        UrlIndexResolver(github, tmp_path)


def _mirror_config(commit: str = "c2c813e171cb8d8c5f76bf1034aaf94304c267c8") -> PinnedSourceConfig:
    return PinnedSourceConfig(
        type="local_mirror",
        name="Claude Platform Docs",
        slug="claude-platform",
        version="2026-08-17",
        license="MIT",
        mirror_dir="platform",
        commit=commit,
        url_prefix="https://platform.claude.com/docs/en/",
        base_url="https://platform.claude.com/docs",
        doc_type="guide",
    )


def _init_git(tmp_path) -> None:
    """Turn tmp_path into a git repo (LocalMirrorResolver validates HEAD)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    proc = subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "m"],
        check=True,
        capture_output=True,
    )
    _ = proc


def test_local_mirror_resolves_from_disk(tmp_path) -> None:
    root = tmp_path / "mirror" / "platform"
    root.mkdir(parents=True)
    (root / "llms.txt").write_text(
        "- [Working with messages](https://platform.claude.com/docs/en/build-with-claude/working-with-messages.md)\n"
        "- [Tool use](https://platform.claude.com/docs/en/build-with-claude/tool-use.md)\n"
    )
    (root / "build-with-claude").mkdir()
    (root / "build-with-claude" / "working-with-messages.md").write_text("# Messages\n")
    (root / "build-with-claude" / "tool-use.md").write_text("# Tool use\n")
    _init_git(tmp_path / "mirror")

    head = subprocess.run(
        ["git", "-C", str(tmp_path / "mirror"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    resolver = LocalMirrorResolver(_mirror_config(commit=head), tmp_path / "mirror")
    manifest = resolver.resolve()
    assert manifest.root == root
    assert len(manifest.entries) == 2
    assert manifest.entries[0].relative_path == "build-with-claude/working-with-messages.md"


def test_local_mirror_rejects_pinned_commit_mismatch(tmp_path) -> None:
    root = tmp_path / "mirror" / "platform"
    root.mkdir(parents=True)
    (root / "llms.txt").write_text("")
    _init_git(tmp_path / "mirror")

    resolver = LocalMirrorResolver(_mirror_config(commit="a" * 40), tmp_path / "mirror")
    with pytest.raises(ValueError, match="HEAD"):
        resolver.resolve()


def test_local_mirror_rejects_non_local_mirror_config(tmp_path) -> None:
    with pytest.raises(ValueError, match="local_mirror"):
        LocalMirrorResolver(_config(), tmp_path)
