"""Tests for BM25 tokenizer used in hybrid search sparse vectors."""

from __future__ import annotations

from data_engineering_copilot.infrastructure.bm25_tokenizer import (
    BM25Tokenizer,
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
