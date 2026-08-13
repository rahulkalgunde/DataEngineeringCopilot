"""Generation maintenance helpers: classify built collections as active/stale/orphan.

``gen-stale`` uses this to surface generation collections that are built but no
longer the active index, or that have no local artifacts backing them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_GENERATION_PREFIX = "data_engineering_docs__"


@dataclass(frozen=True)
class GenerationStatus:
    """One generation collection classified for maintenance."""

    name: str
    state: str  # "active" | "stale" | "orphan"


def is_generation_collection(name: str) -> bool:
    """Return True for ``data_engineering_docs__<gen>`` collection names."""
    return name.startswith(_GENERATION_PREFIX)


def classify_generations(
    collection_names: list[str],
    active_generation: str | None,
    local_collection_names: set[str] | None = None,
) -> list[GenerationStatus]:
    """Classify generation collections present in Qdrant.

    - ``active``: the collection matching ``active_generation``;
    - ``stale``: a generation collection still backed by local artifacts
      (``chunks.jsonl`` under the corpus dirs, expressed as collection-name
      forms via ``local_collection_names``) that is not active;
    - ``orphan``: a generation collection with no local artifacts.

    Non-generation collections are ignored. ``local_collection_names`` default
    to empty so callers can omit the Qdrant-backed scan entirely.
    """
    active_name = f"{_GENERATION_PREFIX}{active_generation}" if active_generation else None
    local_names = local_collection_names or set()
    statuses: list[GenerationStatus] = []
    for name in sorted(collection_names):
        if not is_generation_collection(name):
            continue
        if name == active_name:
            state = "active"
        elif name in local_names:
            state = "stale"
        else:
            state = "orphan"
        statuses.append(GenerationStatus(name=name, state=state))
    return statuses


def local_generation_collections(corpus_dirs: Sequence[Path]) -> set[str]:
    """Collect ``data_engineering_docs__<gen>`` names backed by ``chunks.jsonl``.

    ``corpus_dirs`` are directories whose subdirectories are generation artifact
    roots. Only generations with a ``chunks.jsonl`` artifact count as present
    on disk.
    """
    local: set[str] = set()
    for corpus in corpus_dirs:
        if not corpus.is_dir():
            continue
        for gen_dir in sorted(corpus.iterdir()):
            if (gen_dir / "chunks.jsonl").is_file():
                local.add(f"{_GENERATION_PREFIX}{gen_dir.name}")
    return local
