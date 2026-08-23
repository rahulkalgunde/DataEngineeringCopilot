"""Frozen candidate pools: fetch once, then re-rank/re-fuse offline for free."""

from __future__ import annotations

import json
import pathlib


def save_pool(path: str | pathlib.Path, pools: dict[str, list[dict]]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pools, indent=2), encoding="utf-8")


def load_pool(path: str | pathlib.Path) -> dict[str, list[dict]]:
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def rank_from_pool(pool: list[dict], k: int) -> list[str]:
    ordered = sorted(pool, key=lambda c: c.get("fused_score", 0.0), reverse=True)
    return [c["url"] for c in ordered[:k]]
