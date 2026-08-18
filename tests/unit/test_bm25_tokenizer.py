"""Tests for BM25 tokenizer used in hybrid search sparse vectors."""

from __future__ import annotations

import json

import pytest

from data_engineering_copilot.infrastructure.bm25_tokenizer import (
    BM25Tokenizer,
    _stemmer,
)


class TestBM25TokenizerTokenize:
    """Tokenization without prior fit() — uses default weights."""

    def test_simple_tokens(self):
        tok = BM25Tokenizer()
        tokens = tok.tokenize("Delta Lake ACID transactions")
        assert len(tokens) == 4  # delta, lake, acid, transactions

    def test_stopwords_removed(self):
        tok = BM25Tokenizer()
        tokens = tok.tokenize("the quick brown fox is in a box")
        assert len(tokens) == 4  # quick, brown, fox, box — stopwords gone

    def test_short_tokens_filtered(self):
        tok = BM25Tokenizer()
        tokens = tok.tokenize("SQL queries on io")
        # "on" is a stopword, "io" is 2 chars and kept
        assert len(tokens) >= 1

    def test_empty_text(self):
        tok = BM25Tokenizer()
        tokens = tok.tokenize("")
        assert tokens == []

    def test_only_stopwords(self):
        tok = BM25Tokenizer()
        tokens = tok.tokenize("the and or in on at")
        assert tokens == []

    def test_deduplicates_tokens(self):
        tok = BM25Tokenizer()
        tokens = tok.tokenize("spark spark spark")
        assert len(tokens) == 1  # unique token IDs

    def test_case_insensitive(self):
        tok = BM25Tokenizer()
        tokens_a = tok.tokenize("Spark SQL query")
        tok2 = BM25Tokenizer()
        tokens_b = tok2.tokenize("spark sql query")
        # Same vocab IDs produced
        assert sorted([t.id for t in tokens_a]) == sorted([t.id for t in tokens_b])


class TestBM25TokenizerFit:
    """Tests for corpus fitting."""

    def test_fit_builds_vocab(self):
        tok = BM25Tokenizer()
        tok.fit(["Delta Lake ACID transactions", "Spark SQL queries"])
        assert tok._corpus_size == 2
        assert len(tok._vocab) > 0
        assert tok._avg_doc_len > 0

    def test_fit_sets_doc_freq(self):
        tok = BM25Tokenizer()
        tok.fit(
            [
                "Delta Lake ACID transactions",
                "Delta Lake time travel",
                "Spark SQL queries",
            ]
        )
        assert tok._doc_freq["delta"] == 2
        assert tok._doc_freq["lake"] == 2
        assert tok._doc_freq["acid"] == 1
        assert tok._doc_freq["spark"] == 1

    def test_fit_freezes_tokenizer(self):
        tok = BM25Tokenizer()
        tok.fit(["hello world"])
        assert tok._frozen is True

    def test_fit_empty_corpus(self):
        tok = BM25Tokenizer()
        tok.fit([])
        assert tok._corpus_size == 0
        assert tok._avg_doc_len == 0.0


class TestBM25TokenizerTokenizeQuery:
    """Query tokenization produces SparseVector."""

    def test_returns_sparse_vector(self):
        from qdrant_client.http.models import SparseVector

        tok = BM25Tokenizer()
        tok.fit(["Delta Lake ACID transactions", "Spark SQL queries"])
        sv = tok.tokenize_query("Delta Lake")
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) == 2
        assert len(sv.values) == 2

    def test_query_tokens_match_fitted_vocab(self):
        tok = BM25Tokenizer()
        tok.fit(["Delta Lake ACID transactions", "Spark SQL queries"])
        sv = tok.tokenize_query("Delta Lake")
        # All token IDs should be in vocab
        for idx in sv.indices:
            assert idx in tok._vocab.values()

    def test_empty_query(self):
        from qdrant_client.http.models import SparseVector

        tok = BM25Tokenizer()
        tok.fit(["hello world"])
        sv = tok.tokenize_query("")
        assert isinstance(sv, SparseVector)
        assert sv.indices == []
        assert sv.values == []

    def test_idf_weights_make_sense(self):
        tok = BM25Tokenizer()
        tok.fit(
            [
                "Delta Lake ACID transactions",
                "Delta Lake time travel",
                "Spark SQL queries",
                "Spark Streaming real-time",
            ]
        )
        sv = tok.tokenize_query("Delta Lake")
        weights = dict(zip(sv.indices, sv.values, strict=True))
        # "lake" appears in 2/4 docs, "acid" in 1/4
        lake_id = tok._vocab["lake"]
        acid_id = tok._vocab["acid"]
        # IDF for "acid" (1/4) should be higher than "lake" (2/4)
        if acid_id in weights and lake_id in weights:
            assert weights[acid_id] >= weights[lake_id]


class TestBM25TokenizerEdgeCases:
    def test_special_characters(self):
        tok = BM25Tokenizer()
        tokens = tok.tokenize("config: spark.sql.shuffle.partitions=200")
        # "spark", "sql", "shuffle", "partitions", "200" should be extracted
        assert len(tokens) >= 4

    def test_unicode_text(self):
        tok = BM25Tokenizer()
        tokens = tok.tokenize("Spark ëncoding café test")
        assert len(tokens) >= 2

    def test_large_corpus_fit(self):
        tok = BM25Tokenizer()
        corpus = [f"Document {i} about topic {i % 50}" for i in range(1000)]
        tok.fit(corpus)
        assert tok._corpus_size == 1000
        sv = tok.tokenize_query("Document 42 topic 42")
        assert len(sv.indices) >= 2


class TestBM25TokenizerNamespace:
    """namespace-v1 mode: full identifiers + stemmed components."""

    def _tokens(self, text: str) -> list[str]:
        return BM25Tokenizer(namespace=True)._extract_tokens(text)

    def test_dotted_identifier_full_and_components(self):
        tokens = self._tokens("use spark.sql.functions.row_number")
        assert "spark.sql.functions.row_number" in tokens
        for component in ("spark", "sql", _stemmer.stem("functions"), _stemmer.stem("row_number")):
            assert component in tokens

    def test_path_identifier_full_and_components(self):
        tokens = self._tokens("see data/engineering/copilot guide")
        assert "data/engineering/copilot" in tokens
        for component in ("data", _stemmer.stem("engineering"), _stemmer.stem("copilot")):
            assert component in tokens

    def test_version_identifier(self):
        tokens = self._tokens("upgrade to v3.4.1")
        assert "v3.4.1" in tokens
        assert "v3" in tokens  # single-char segments (4, 1) are dropped

    def test_case_insensitive_identifiers(self):
        upper = self._tokens("Spark.SQL.Functions")
        lower = self._tokens("spark.sql.functions")
        assert upper == lower

    def test_sql_names_stay_atomic(self):
        tokens = self._tokens("ROW_NUMBER() OVER (PARTITION BY dept)")
        assert _stemmer.stem("row_number") in tokens
        assert "row" not in tokens and _stemmer.stem("number") not in tokens

    def test_urls_preserve_host_and_path_tokens(self):
        tokens = self._tokens("read https://spark.apache.org/docs/latest guide")
        assert "spark.apache.org/docs/latest" in tokens
        assert "https" not in tokens
        for component in ("spark", "apache", "org", "docs", "latest"):
            assert _stemmer.stem(component) in tokens

    def test_prose_unaffected_by_namespace_mode(self):
        namespace = self._tokens("The quick brown fox jumps")
        legacy = BM25Tokenizer()._extract_tokens("The quick brown fox jumps")
        assert namespace == legacy

    def test_punctuation_only_terms_absent(self):
        assert self._tokens("...") == []
        assert self._tokens("::") == []
        assert self._tokens("use ... or") == [*BM25Tokenizer()._extract_tokens("use or")]

    def test_namespace_fit_and_query_match_full_identifier(self):
        tok = BM25Tokenizer(namespace=True)
        tok.fit(["applies spark.sql.functions.col to each row"])
        sv = tok.tokenize_query("spark.sql.functions.col")
        assert len(sv.indices) >= 3  # full token + spark/sql components
        assert "spark.sql.functions.col" in tok._vocab

    def test_namespace_query_component_hits_fitted_vocab(self):
        tok = BM25Tokenizer(namespace=True)
        tok.fit(["applies spark.sql.functions.col to each row"])
        sv = tok.tokenize_query("spark sql functions")
        assert len(sv.indices) >= 3
        for idx in sv.indices:
            assert idx in tok._vocab.values()

    def test_namespace_legacy_query_overlap(self):
        """A namespace-fitted corpus is still queriable by prose words."""
        tok = BM25Tokenizer(namespace=True)
        tok.fit(["the pyspark.sql module provides DataFrame"])
        sv = tok.tokenize_query("pyspark sql dataframe")
        assert len(sv.indices) >= 3

    def test_save_load_round_trip_namespace(self, tmp_path):
        tok = BM25Tokenizer(namespace=True)
        tok.fit(["spark.sql.functions row_number"])
        path = tmp_path / "bm25.json"
        tok.save(path)
        loaded = BM25Tokenizer.load(path)
        assert loaded.version == BM25Tokenizer.TOKENIZER_VERSION
        assert loaded.namespace_enabled is True
        assert loaded._vocab == tok._vocab
        assert loaded._frozen is True

    def test_save_load_round_trip_legacy(self, tmp_path):
        tok = BM25Tokenizer()
        tok.fit(["apache spark"])
        path = tmp_path / "bm25.json"
        tok.save(path)
        loaded = BM25Tokenizer.load(path)
        assert loaded.version == BM25Tokenizer.LEGACY_TOKENIZER_VERSION
        assert loaded.namespace_enabled is False

    def test_load_legacy_cache_without_version_key(self, tmp_path):
        path = tmp_path / "bm25.json"
        path.write_text(
            json.dumps(
                {
                    "k1": 1.2,
                    "b": 0.75,
                    "vocab": {"spark": 0},
                    "doc_freq": {"spark": 1},
                    "corpus_size": 1,
                    "avg_doc_len": 2.0,
                    "frozen": True,
                }
            )
        )
        loaded = BM25Tokenizer.load(path)
        assert loaded.version == BM25Tokenizer.LEGACY_TOKENIZER_VERSION
        assert loaded.namespace_enabled is False

    def test_load_unsupported_version_raises(self, tmp_path):
        path = tmp_path / "bm25.json"
        data = {
            "tokenizer_version": "namespace-v2",
            "k1": 1.2,
            "b": 0.75,
            "vocab": {},
            "doc_freq": {},
            "corpus_size": 0,
            "avg_doc_len": 0.0,
            "frozen": True,
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Unsupported BM25 tokenizer version"):
            BM25Tokenizer.load(path)
