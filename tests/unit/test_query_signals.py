"""Tests for deterministic query-signal classification and RRF profile selection."""

from __future__ import annotations

from data_engineering_copilot.services.query_signals import (
    RRF_EQUAL_PROFILE,
    RRF_IDENTIFIER_SPARSE_PROFILE,
    classify_query_signals,
    select_rrf_profile,
)


class TestClassifyQuerySignals:
    def test_prose_query_has_no_signals(self):
        signals = classify_query_signals("how does spark memory management work")
        assert signals.identifier_heavy is False
        assert signals.path_heavy is False
        assert signals.version_qualified is False
        assert signals.code_like is False

    def test_dotted_identifier(self):
        signals = classify_query_signals("what does pyspark.sql.functions.col do")
        assert signals.identifier_heavy is True
        assert signals.path_heavy is False
        assert signals.version_qualified is False

    def test_snake_case_heavy(self):
        signals = classify_query_signals("partition_by order_by data_frame column_names")
        assert signals.identifier_heavy is True

    def test_camel_case(self):
        signals = classify_query_signals("DataFrame groupBy partitionBy")
        assert signals.identifier_heavy is True

    def test_path(self):
        signals = classify_query_signals("where is the sql/catalyst/expressions source")
        assert signals.path_heavy is True

    def test_windows_path(self):
        signals = classify_query_signals("path in the docs\\user_guide\\sql folder")
        assert signals.path_heavy is True

    def test_version_qualified(self):
        signals = classify_query_signals("how does spark 4.0 handle shuffling")
        assert signals.version_qualified is True

    def test_sql_syntax(self):
        signals = classify_query_signals("select * from table where column = 1 group by x")
        assert signals.code_like is True

    def test_code_syntax(self):
        signals = classify_query_signals("def load_data(): return spark.read.parquet")
        assert signals.code_like is True

    def test_backtick_code(self):
        signals = classify_query_signals("what does `df.groupBy('x').count()` return")
        assert signals.code_like is True

    def test_mixed_technical_query(self):
        signals = classify_query_signals("pyspark.sql filter window partitionBy 4.0")
        assert signals.identifier_heavy is True
        assert signals.version_qualified is True


class TestSelectRrfProfile:
    def test_technical_signals_select_identifier_sparse(self):
        for query in (
            "what does pyspark.sql.functions.col do",
            "where is the sql/catalyst/expressions source",
            "how does spark 4.0 handle shuffling",
            "select * from table where x = 1",
            "def load(): return x",
        ):
            assert select_rrf_profile(classify_query_signals(query)) == RRF_IDENTIFIER_SPARSE_PROFILE

    def test_prose_selects_equal(self):
        for query in (
            "how does spark memory management work",
            "what is the difference between wide and narrow transformations",
        ):
            assert select_rrf_profile(classify_query_signals(query)) == RRF_EQUAL_PROFILE

    def test_only_two_profiles_used(self):
        assert {select_rrf_profile(classify_query_signals(q)) for q in ("prose", "a.b.c", "x/y", "spark 4.0")} == {
            RRF_EQUAL_PROFILE,
            RRF_IDENTIFIER_SPARSE_PROFILE,
        }
