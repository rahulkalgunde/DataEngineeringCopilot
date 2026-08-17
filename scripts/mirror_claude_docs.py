#!/usr/bin/env python3
"""Mirror the Claude Platform / Code docs into a local git repo.

Downloads each site's ``llms.txt`` index plus every linked ``.md`` file into
``data/claude_docs_mirror/<site>/`` and commits the snapshot so the pinned
``local_mirror`` sources in ``pinned_sources.json`` resolve reproducibly.

After running this script, paste the printed commit SHAs into the matching
``local_mirror`` source entries (``commit`` field) and re-run
``dec gen-manifest`` / ``dec gen-build``.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import httpx

from data_engineering_copilot.services.claude_docs_ingestion import (
    LLMS_DOC_SITES,
    fetch_llms_index,
    fetch_markdown_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIRROR_ROOT = PROJECT_ROOT / "data" / "claude_docs_mirror"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip()


def _commit_snapshot(site_dir: Path, site: str) -> str:
    """git init + commit the site snapshot, returning the HEAD SHA."""
    site_dir.mkdir(parents=True, exist_ok=True)
    _git(site_dir, "init", "-q")
    _git(site_dir, "add", ".")
    # `git commit` fails with nothing to commit; treat as no-op.
    subprocess.run(
        ["git", "-C", str(site_dir), "commit", "-q", "-m", f"snapshot: {site} {_now()}"],
        capture_output=True,
    )
    head = _git(site_dir, "rev-parse", "HEAD")
    return head or ""


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


async def _mirror_site(site: str) -> str:
    dest = MIRROR_ROOT / site
    dest.mkdir(parents=True, exist_ok=True)
    # Save the raw index so the local resolver can re-derive the entry list.
    index_url = LLMS_DOC_SITES[site]["llms_url"]
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(index_url, headers={"User-Agent": "DataEngineeringCopilot/1.0"})
        resp.raise_for_status()
        (dest / "llms.txt").write_text(resp.text, encoding="utf-8")
    entries = await fetch_llms_index(site)
    print(f"  {site}: {len(entries)} docs")
    await fetch_markdown_files(site, entries, dest)
    head = _commit_snapshot(dest, site)
    print(f"  {site}: committed {head or '(no changes)'} -> {dest}")
    return head


async def main() -> int:
    MIRROR_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Mirroring Claude docs -> {MIRROR_ROOT}")
    for site in LLMS_DOC_SITES:
        await _mirror_site(site)
    print("Done. Update `commit` in pinned_sources.json to the SHAs above.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
