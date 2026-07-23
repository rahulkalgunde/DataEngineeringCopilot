"""Fast BM25 tokenizer for creating sparse vectors in hybrid search.

Produces Qdrant-compatible ``SparseVector`` representations without external
dependencies (no rank_bm25, no Elasticsearch).  The tokenizer is intentionally
lightweight: regex word extraction, stopword filtering, and standard IDF
weighting with BM25 k1/b normalization.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from nltk.stem import PorterStemmer
from qdrant_client.http.models import SparseVector

_stemmer = PorterStemmer()

_WORD_RE = re.compile(r"[a-zA-Z0-9_\-]{2,}")
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
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self._k1 = k1
        self._b = b
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

        After calling ``fit()`` the tokenizer is frozen.
        """
        token_lists = [self._extract_tokens(t) for t in texts if t.strip()]
        if not token_lists:
            return

        # Doc frequency: each unique token per document
        for token_list in token_lists:
            for t in set(token_list):
                self._doc_freq[t] += 1

        # Corpus statistics
        all_tokens = [t for tl in token_lists for t in tl]
        self._corpus_size = len(token_lists)
        self._avg_doc_len = len(all_tokens) / self._corpus_size if self._corpus_size else 0

        # Build vocab — assign sequential IDs
        for token_list in token_lists:
            for t in token_list:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)

        self._frozen = True

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tokens(text: str) -> list[str]:
        """Lowercase regex extraction + stopword removal + stemming."""
        return [_stemmer.stem(t.lower()) for t in _WORD_RE.findall(text) if t.lower() not in _STOPWORDS]
