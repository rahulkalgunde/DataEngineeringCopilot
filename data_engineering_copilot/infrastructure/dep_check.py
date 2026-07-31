"""Detect when a container is running with a stale dependency set.

The Dockerfile bakes ``/image_deps_sha256.txt`` = sha256 of ``pyproject.toml +
uv.lock`` at build time. At runtime (code is bind-mounted from the host) the
live ``/app/pyproject.toml`` + ``/app/uv.lock`` are re-hashed; any difference
means the image was not rebuilt after a dependency change.

Usage:
    API (warn only):  ``check_deps(fail_fast=False)`` at startup
    Worker (refuse):  ``check_deps(fail_fast=True)`` at import
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

IMAGE_FINGERPRINT_FILE = "/image_deps_sha256.txt"
LIVE_DIR = "/app"
FINGERPRINT_FILES = ("pyproject.toml", "uv.lock")

STALE_MESSAGE = (
    "Running a STALE image: dependencies (pyproject.toml/uv.lock) changed since "
    "this image was built. Rebuild with `make docker-dev` (or "
    "`docker compose --profile app up -d --build backend-api celery_worker`)."
)


def _read_baked(baked_path: str = IMAGE_FINGERPRINT_FILE) -> str | None:
    try:
        with open(baked_path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _live_fingerprint(live_dir: str = LIVE_DIR) -> str | None:
    parts: list[bytes] = []
    for name in FINGERPRINT_FILES:
        try:
            with open(os.path.join(live_dir, name), "rb") as f:
                parts.append(f.read())
        except OSError:
            return None
    return hashlib.sha256(b"".join(parts)).hexdigest()


def fingerprint_ok(baked_path: str = IMAGE_FINGERPRINT_FILE, live_dir: str = LIVE_DIR) -> bool | None:
    """Return True if deps match the baked image, False if stale.

    Returns None when the environment is not an image-built container
    (e.g. a plain host venv or test run), in which case callers should skip.
    """
    baked = _read_baked(baked_path)
    if baked is None:
        return None
    live = _live_fingerprint(live_dir)
    if live is None:
        return None
    return live == baked


def check_deps(
    fail_fast: bool = False,
    *,
    baked_path: str = IMAGE_FINGERPRINT_FILE,
    live_dir: str = LIVE_DIR,
) -> bool:
    """Verify dependency freshness; return True when OK or indeterminate.

    If stale and ``fail_fast`` is True, exits the process (worker refuses to
    start with mismatched dependencies). Otherwise logs an ERROR and returns False.
    """
    ok = fingerprint_ok(baked_path=baked_path, live_dir=live_dir)
    if ok is None:
        return True
    if not ok:
        logger.error("image_stale %s", STALE_MESSAGE)
        if fail_fast:
            raise SystemExit(STALE_MESSAGE)
    return ok
