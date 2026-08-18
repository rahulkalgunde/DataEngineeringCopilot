"""Deterministic query-signal classification and RRF profile selection.

Technical queries (dotted identifiers, filesystem paths, version-qualified
lookups, SQL/code syntax) rely more on exact token matches than broad prose
queries, so they benefit from boosting the sparse (BM25) side of the hybrid
fusion. This module classifies a query into deterministic boolean signals and
picks one of two RRF profiles:

- ``equal_rrf``: the current behavior — dense and sparse prefetches are fused
  with equal weights.
- ``identifier_sparse_rrf``: Qdrant weighted RRF with sparse weight ``1.25``
  and dense weight ``1.0``. No raw-score alpha mixing is used.

The weighted profile is only applied when the benchmark gate passes
(identifier recall +>=0.05 with global recall/MRR thresholds satisfied);
until then ``equal_rrf`` remains the effective profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# RRF profile names (matched by ``AsyncQdrantVectorStore.query(rrf_profile=...)``).
RRF_EQUAL_PROFILE = "equal_rrf"
RRF_IDENTIFIER_SPARSE_PROFILE = "identifier_sparse_rrf"

# Weighted-RRF weights for the identifier-sparse profile. Order matches the
# prefetch order in the store: dense first, sparse second. Sparse gets a boost
# because technical queries hinge on exact token matches.
RRF_DENSE_WEIGHT = 1.0
RRF_SPARSE_WEIGHT = 1.25


@dataclass(frozen=True)
class QuerySignals:
    """Boolean signals describing how "technical" a query is."""

    identifier_heavy: bool
    path_heavy: bool
    version_qualified: bool
    code_like: bool


# Dotted / snake_case / camelCase identifiers, e.g. ``pyspark.sql.functions.col``,
# ``partitionBy``, ``data_frame``.
_DOTTED_IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b")
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+\b")

# Filesystem / module path, e.g. ``sql/catalyst/expressions`` or ``foo\\bar\\baz``.
_PATH_RE = re.compile(r"\b[\w.~-]+(?:/[\w.~-]+){1,}\b|\b[\w.\\-]+(?:\\[\w.\\-]+){1,}\b")

# Version numbers, e.g. ``4.0.0``, ``1.12``.
_VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")

# SQL / code syntax markers.
_SQL_KEYWORD_RE = re.compile(
    r"\b(select|from|where|group\s+by|order\s+by|join|case\s+when|insert\s+into|"
    r"update\s+|delete\s+from|create\s+(table|view)|having|limit|distinct)\b",
    re.IGNORECASE,
)
_CODE_SYNTAX_RE = re.compile(r"```|`[^`\n]+`|\b(def|class|import|return|print)\b|\b\w+\s*\(")


def classify_query_signals(query: str) -> QuerySignals:
    """Classify a query into deterministic technical signals.

    ``identifier_heavy`` fires on dotted identifiers, or several snake_case /
    camelCase tokens; ``path_heavy`` on path-like strings; ``version_qualified``
    on dotted version numbers; ``code_like`` on SQL keywords or code syntax.
    """
    dotted = _DOTTED_IDENTIFIER_RE.search(query) is not None
    snake = len(_SNAKE_CASE_RE.findall(query))
    camel = len(_CAMEL_CASE_RE.findall(query))
    identifier_heavy = dotted or snake >= 3 or camel >= 2
    path_heavy = _PATH_RE.search(query) is not None
    version_qualified = _VERSION_RE.search(query) is not None
    code_like = _SQL_KEYWORD_RE.search(query) is not None or _CODE_SYNTAX_RE.search(query) is not None
    return QuerySignals(
        identifier_heavy=identifier_heavy,
        path_heavy=path_heavy,
        version_qualified=version_qualified,
        code_like=code_like,
    )


def select_rrf_profile(signals: QuerySignals) -> str:
    """Pick the RRF profile for a query based on its signals.

    ``identifier_sparse_rrf`` is chosen only when any technical signal is set;
    otherwise ``equal_rrf`` is returned.
    """
    if signals.identifier_heavy or signals.path_heavy or signals.version_qualified or signals.code_like:
        return RRF_IDENTIFIER_SPARSE_PROFILE
    return RRF_EQUAL_PROFILE
