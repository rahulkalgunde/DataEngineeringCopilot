"""Canonical URL content-key normalization for evaluation scoring and coverage.

Root-cause fix for an exact-URL matching defect: golden eval URLs are stored in
canonical public form (``spark.apache.org/docs/4.0.0/X.md``,
``docs.delta.io/latest/X.html``, ``airflow.apache.org/docs/.../X.html``) while
the indexed corpus stores the same documents under raw-GitHub commit URLs
(``raw.githubusercontent.com/apache/spark/<sha>/docs/X.md``). Exact-string and
``rstrip``-only matching silently scored correct retrievals as misses.

``url_content_key`` maps either form to a deterministic, host-independent
"content key" derived from the document's source-relative path, so both forms
collide for the same document. Every URL-matching site in evaluation (coverage,
retrieval scoring, fast-eval, benchmark) must route through this function on
both the expected and retrieved sides.

Unrecognized URLs fall back to a host-agnostic path normalization so plain
identifiers and synthetic test URLs keep a stable identity.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

# (canonical mount token, corpus mount token) per source.
#   canonical token : text stripped from a public URL before the doc rel path.
#   corpus token    : text stripped from a raw-GitHub URL before the doc rel path.
# The rel path after the matched token is identical for the same document.
_SOURCE_MOUNTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "spark": (("docs/4.0.0/", "/4.0.0/docs/", "docs/latest/"), ("docs/",)),
    "delta": (("docs/latest/", "latest/"), ("docs/src/content/docs/",)),
    "airflow": (
        ("docs/apache-airflow/stable/", "apache-airflow/stable/", "stable/"),
        ("airflow-core/docs/",),
    ),
    "claude_plat": (("docs/en/",), ("docs/en/",)),
    "claude_code": (("docs/en/",), ("docs/en/",)),
}

# Host/URL signatures used to detect the source when no source_name is supplied.
_SOURCE_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("spark", "spark.apache"),
    ("spark", "apache/spark"),
    ("delta", "delta.io"),
    ("delta", "delta-io"),
    ("airflow", "airflow"),
    ("claude_plat", "platform.claude"),
    ("claude_code", "code.claude"),
)


def detect_source(url: str) -> str | None:
    """Return the source key for ``url`` or ``None`` if unrecognized."""
    low = (url or "").lower()
    for key, sig in _SOURCE_SIGNATURES:
        if sig in low:
            return key
    return None


def _strip_tokens(path: str, tokens: tuple[str, ...]) -> str:
    """Return ``path`` starting just after the first matching token, else unchanged."""
    for tok in tokens:
        idx = path.find(tok)
        if idx != -1:
            return path[idx + len(tok) :]
    return path


def _clean_rel(rel: str) -> str:
    rel = unquote(rel)
    rel = re.sub(r"\.[a-z0-9]+$", "", rel)  # trailing file extension
    rel = re.sub(r"/index$", "", rel)  # index.md / index.html fold
    return rel.rstrip("/")


def _fallback_key(url: str) -> str:
    """Host-agnostic path normalization for unrecognized URLs / plain identifiers."""
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        return u  # plain identifier / synthetic test token: identity
    path = urlparse(u).path
    rel = _clean_rel(path.lstrip("/"))
    return rel or u


def url_content_key(url: str, source_name: str | None = None) -> str:
    """Return a deterministic content key for ``url``.

    Both the canonical public form and the indexed raw-GitHub form of the same
    document produce the same key. Unknown source names fall back to host-agnostic
    normalization (identity for plain identifiers).
    """
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        return u
    key = detect_source(u) or (source_name and detect_source(source_name))
    path = urlparse(u).path
    if key in _SOURCE_MOUNTS:
        canonical_tokens, corpus_tokens = _SOURCE_MOUNTS[key]
        is_raw = "raw.githubusercontent.com" in u or "/apache/" in u.lower() or "/delta-io/" in u.lower()
        tokens = corpus_tokens if is_raw else canonical_tokens
        rel = _clean_rel(_strip_tokens(path, tokens))
        return f"{key}::{rel}"
    return _fallback_key(u)


def same_document(url_a: str, url_b: str, source_name_a: str | None = None) -> bool:
    """True when two URLs reference the same indexed document."""
    return url_content_key(url_a, source_name_a) == url_content_key(url_b)


def normalize_urls(urls: list[str] | set[str]) -> set[str]:
    """Map a collection of URLs to their content keys (deduplicated)."""
    return {url_content_key(u) for u in urls}
