"""Deterministic query-signal classification and search mode selection.

Technical queries (dotted identifiers, filesystem paths, version-qualified
lookups, SQL/code syntax) rely more on exact token matches than broad prose
queries, so they benefit from different search mechanisms:

Search Modes:
- ``bm25_only``: Pure BM25 sparse vector search (exact token matching)
- ``dense_only``: Pure dense vector search (semantic similarity)
- ``hybrid_equal``: Dense + sparse with equal RRF weights (1.0, 1.0)
- ``hybrid_sparse_bias``: Dense + sparse with sparse bias (1.0, 1.25)
- ``hybrid_dense_bias``: Dense + sparse with dense bias (1.25, 1.0)

The mode is selected deterministically based on query intent and signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """Search mechanism to use for retrieval."""

    BM25_ONLY = "bm25_only"
    DENSE_ONLY = "dense_only"
    HYBRID_EQUAL = "hybrid_equal"
    HYBRID_SPARSE_BIAS = "hybrid_sparse_bias"
    HYBRID_DENSE_BIAS = "hybrid_dense_bias"


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


def select_search_mode(
    intent: str,
    signals: QuerySignals | str | None = None,
    *,
    query: str | None = None,
) -> SearchMode:
    """Select the search mechanism based on intent and query signals.

    Routing logic:
    - api_lookup, code_example -> BM25_ONLY (exact token matching for API/code)
    - debugging -> HYBRID_SPARSE_BIAS (error terms + semantic)
    - factual, how_to, synthesis -> DENSE_ONLY (semantic similarity)
    - comparative -> HYBRID_EQUAL (both modes matter)
    - configuration -> HYBRID_DENSE_BIAS (conceptual + exact terms)

    ADR-012: ``api_lookup`` must *always* return :attr:`SearchMode.BM25_ONLY`
    even when the rewrite drifts (e.g. ``dense_rank()`` → ``"dense ranking"``).
    Intent is the hard gate; signals are only a fallback for unknown intents.
    Callers MUST pass the **original** question (not the rewritten
    ``effective_query``) so drift cannot dilute the routing. ``query`` may be
    passed as a ``str`` (classified internally) or ``signals`` may be a
    :class:`QuerySignals` — both forms are accepted for backward compatibility.
    """
    # Resolve signals/query flexibly (TDD: tests call with ``query=`` kwarg,
    # production code calls with ``QuerySignals`` positional).
    effective_signals: QuerySignals | None = None
    if query is not None:
        effective_signals = classify_query_signals(query)
    elif isinstance(signals, str):
        effective_signals = classify_query_signals(signals)
    elif isinstance(signals, QuerySignals):
        effective_signals = signals
    else:
        effective_signals = None

    # Intent-based primary routing — hard gate, ignores signals/query drift
    if intent in ("api_lookup", "code_example"):
        return SearchMode.BM25_ONLY
    if intent == "debugging":
        return SearchMode.HYBRID_SPARSE_BIAS
    if intent in ("factual", "how_to", "synthesis"):
        return SearchMode.DENSE_ONLY
    if intent == "comparative":
        return SearchMode.HYBRID_EQUAL
    if intent == "configuration":
        return SearchMode.HYBRID_DENSE_BIAS

    # Fallback: signal-based routing (only when intent is unknown)
    if effective_signals is not None:
        if effective_signals.identifier_heavy or effective_signals.code_like:
            return SearchMode.HYBRID_SPARSE_BIAS
        if effective_signals.path_heavy or effective_signals.version_qualified:
            return SearchMode.HYBRID_DENSE_BIAS
    return SearchMode.HYBRID_EQUAL
