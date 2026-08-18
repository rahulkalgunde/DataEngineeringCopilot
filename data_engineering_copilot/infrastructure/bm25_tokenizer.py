"""Fast BM25 tokenizer for creating sparse vectors in hybrid search.

Produces Qdrant-compatible ``SparseVector`` representations without external
dependencies (no rank_bm25, no Elasticsearch).  The tokenizer is intentionally
lightweight: regex word extraction, stopword filtering, and standard IDF
weighting with BM25 k1/b normalization.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nltk.stem import PorterStemmer
from qdrant_client.http.models import SparseVector

_stemmer = PorterStemmer()

_WORD_RE = re.compile(r"[a-zA-Z0-9_\-]{2,}")
# Technical tokens: dotted/path/colon-separated identifiers (>=2 segments),
# e.g. ``spark.sql.functions``, ``data/engineering``, ``v3.4.1``.
_TECH_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+(?:[./:][a-zA-Z0-9_]+)+")
# URL schemes are stripped so ``https://spark.apache.org/docs`` is scanned as
# ``spark.apache.org/docs`` instead of merging the scheme into the host token.
_URL_SCHEME_RE = re.compile(r"(?:https?|ftp)://", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "in",
        "on",
        "at",
        "to",
        "of",
        "is",
        "are",
        "it",
        "as",
        "be",
        "by",
        "for",
        "with",
        "that",
        "this",
        "from",
        "which",
        "not",
        "we",
        "do",
        "if",
        "my",
        "has",
        "had",
        "was",
        "were",
        "can",
        "may",
        "its",
        "but",
    }
)


@dataclass(frozen=True)
class SparseToken:
    """A token with its numeric id and BM25 weight."""

    id: int
    weight: float


class BM25Tokenizer:
    """Lightweight BM25 tokenizer that produces Qdrant ``SparseVector`` objects.

    Usage::

        tok = BM25Tokenizer()
        tok.fit(corpus_texts)             # build vocab + IDF tables
        sv = tok.tokenize_query(query)    # → SparseVector for Qdrant

    After ``fit()`` the tokenizer is *frozen*: new words in queries that
    were not in the training corpus are silently dropped.  Before ``fit()``
    it can still ``tokenize()`` but uses a uniform weight of 1.0.

    Namespace mode (``namespace=True``) additionally preserves full dotted /
    path identifiers (``spark.sql.functions``, ``data/engineering``) alongside
    their stemmed components, so queries matching an exact technical name hit
    the identifier token instead of relying on bag-of-words overlap.
    """

    # Tokenizer identity persisted with the BM25 cache. ``legacy`` is the
    # pre-namespace word-only tokenizer; ``namespace-v1`` adds identifier
    # preservation. ``load()`` rejects any other stored version.
    LEGACY_TOKENIZER_VERSION = "legacy"
    TOKENIZER_VERSION = "namespace-v1"
    SUPPORTED_VERSIONS = frozenset({LEGACY_TOKENIZER_VERSION, TOKENIZER_VERSION})

    def __init__(self, k1: float = 1.2, b: float = 0.75, namespace: bool = False):
        self._k1 = k1
        self._b = b
        self._namespace = namespace
        self._version = self.TOKENIZER_VERSION if namespace else self.LEGACY_TOKENIZER_VERSION
        self._vocab: dict[str, int] = {}
        self._doc_freq: Counter[str] = Counter()
        self._corpus_size: int = 0
        self._avg_doc_len: float = 0.0
        self._frozen: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[SparseToken]:
        """Tokenize *text* into ``SparseToken`` objects.

        Uses IDF weights when fitted, uniform 1.0 otherwise.
        """
        tokens = self._extract_tokens(text)
        if not tokens:
            return []

        counts = Counter(tokens)
        doc_len = len(tokens)

        idf_cache: dict[str, float] = {}
        if self._frozen and self._corpus_size > 0:
            for t in set(tokens):
                df = self._doc_freq.get(t, 1)
                idf_cache[t] = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1)

        result: list[SparseToken] = []
        seen_ids: set[int] = set()

        for token, count in counts.items():
            tid = self._vocab.get(token)
            if tid is None:
                if not self._frozen:
                    tid = len(self._vocab)
                    self._vocab[token] = tid
                else:
                    continue

            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            weight = 1.0
            if self._frozen and doc_len > 0 and self._avg_doc_len > 0:
                idf = idf_cache.get(token, 1.0)
                tf = count / doc_len
                norm_factor = 1 - self._b + self._b * doc_len / self._avg_doc_len
                weight = idf * (tf * (self._k1 + 1)) / (tf + self._k1 * norm_factor)

            result.append(SparseToken(id=tid, weight=weight))

        return result

    def tokenize_query(self, text: str) -> SparseVector:
        """Tokenize a query into a Qdrant ``SparseVector``.

        Returns ``SparseVector(indices=[], values=[])`` when no tokens are
        produced, which Qdrant treats as a zero-match vector.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        indices = [t.id for t in tokens]
        values = [t.weight for t in tokens]
        return SparseVector(indices=indices, values=values)

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary and IDF tables from a training corpus.

        Accumulates across multiple calls: each call adds new documents
        to the existing stats (doc_freq, corpus_size, vocab).  After the
        first call the tokenizer is *frozen* (new words in queries are
        dropped), but ``fit()`` itself can be called repeatedly to refine
        IDF statistics as more data arrives.

        With ``Modifier.IDF`` on the Qdrant side, ``tokenize()`` now
        returns raw term-frequency weights — IDF is handled server-side.
        """
        token_lists = [self._extract_tokens(t) for t in texts if t.strip()]
        if not token_lists:
            return

        # Doc frequency: each unique token per document (accumulates)
        for token_list in token_lists:
            for t in set(token_list):
                self._doc_freq[t] += 1

        # Corpus statistics (accumulate)
        all_tokens = [t for tl in token_lists for t in tl]
        new_corpus_size = len(token_lists)
        self._corpus_size += new_corpus_size
        if self._corpus_size > 0:
            self._avg_doc_len = (
                self._avg_doc_len * (self._corpus_size - new_corpus_size) + len(all_tokens)
            ) / self._corpus_size

        # Build vocab — assign sequential IDs (accumulates)
        for token_list in token_lists:
            for t in token_list:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)

        self._frozen = True

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def version(self) -> str:
        """Persisted tokenizer version (``legacy`` or ``namespace-v1``)."""
        return self._version

    @property
    def namespace_enabled(self) -> bool:
        return self._namespace

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        data = {
            "tokenizer_version": self._version,
            "namespace": self._namespace,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
            "corpus_size": self._corpus_size,
            "avg_doc_len": self._avg_doc_len,
            "k1": self._k1,
            "b": self._b,
            "frozen": self._frozen,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> BM25Tokenizer:
        data = json.loads(path.read_text())
        version = str(data.get("tokenizer_version", cls.LEGACY_TOKENIZER_VERSION))
        if version not in cls.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported BM25 tokenizer version {version!r} (supported: {sorted(cls.SUPPORTED_VERSIONS)})"
            )
        namespace = bool(data.get("namespace", version == cls.TOKENIZER_VERSION))
        tok = cls(k1=data["k1"], b=data["b"], namespace=namespace)
        tok._vocab = data["vocab"]
        tok._doc_freq = Counter(data["doc_freq"])
        tok._corpus_size = data["corpus_size"]
        tok._avg_doc_len = data["avg_doc_len"]
        tok._frozen = data["frozen"]
        return tok

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_word_tokens(text: str) -> list[str]:
        """Legacy word-only extraction: lowercase regex + stopwords + stem."""
        return [_stemmer.stem(t.lower()) for t in _WORD_RE.findall(text) if t.lower() not in _STOPWORDS]

    @staticmethod
    def _component_tokens(identifier: str) -> list[str]:
        """Stemmed, stopword-filtered segments of a technical identifier."""
        out: list[str] = []
        for segment in re.split(r"[./:]+", identifier):
            lowered = segment.lower()
            if len(segment) >= 2 and lowered not in _STOPWORDS:
                stemmed = _stemmer.stem(lowered)
                if stemmed not in out:
                    out.append(stemmed)
        return out

    def _extract_namespace_tokens(self, text: str) -> list[str]:
        """Dual extraction: full identifiers + stemmed components + prose.

        Full dotted/path identifiers are emitted lowercased and unstemmed (so
        ``spark.sql.functions`` stays whole); every segment is also emitted
        through the legacy word pipeline so prose queries ("spark sql
        functions") still match.  URL schemes are stripped first.
        """
        masked = _URL_SCHEME_RE.sub(" ", text)
        tech_matches = list(_TECH_TOKEN_RE.finditer(masked))
        if not tech_matches:
            return self._extract_word_tokens(text)

        tokens: list[str] = []
        for match in tech_matches:
            tokens.append(match.group(0).lower())
        cursor = 0
        for match in tech_matches:
            tokens.extend(self._component_tokens(match.group(0)))
            tokens.extend(self._extract_word_tokens(masked[cursor : match.start()]))
            cursor = match.end()
        tokens.extend(self._extract_word_tokens(masked[cursor:]))
        return tokens

    def _extract_tokens(self, text: str) -> list[str]:
        if self._namespace:
            return self._extract_namespace_tokens(text)
        return self._extract_word_tokens(text)
