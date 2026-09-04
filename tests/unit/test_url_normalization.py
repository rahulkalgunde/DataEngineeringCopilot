"""Contract + behavioral tests for the shared URL content-key normalizer.

Pins the public API of ``evaluation.url_normalization`` and locks the
root-cause fix: canonical public URLs (``spark.apache.org/docs/4.0.0/X.md``)
and their indexed raw-GitHub forms (``raw.githubusercontent.com/apache/spark/
<sha>/docs/X.md``) must collide to a single content key, so evaluation scoring
measures content presence, not URL host spelling.
"""

from __future__ import annotations

from data_engineering_copilot.evaluation.url_normalization import (
    detect_source,
    normalize_urls,
    same_document,
    url_content_key,
)


class TestPublicApi:
    """Pin function signatures and return types (contract)."""

    def test_url_content_key_signature(self) -> None:
        assert url_content_key.__code__.co_argcount == 2
        assert url_content_key.__code__.co_varnames[:2] == ("url", "source_name")

    def test_same_document_defaults_source_none(self) -> None:
        assert same_document.__code__.co_argcount == 3

    def test_normalize_urls_returns_set(self) -> None:
        result = normalize_urls(["a", "b", "a"])
        assert isinstance(result, set)
        assert result == {"a", "b"}


class TestSameDocument:
    """Canonical public vs indexed raw-GitHub forms must collide per source."""

    def test_spark_canonical_vs_raw(self) -> None:
        canonical = "https://spark.apache.org/docs/4.0.0/sql-performance-tuning.md"
        raw = "https://raw.githubusercontent.com/apache/spark/fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4/docs/sql-performance-tuning.md"
        assert url_content_key(canonical) == url_content_key(raw)
        assert same_document(canonical, raw)
        assert url_content_key(canonical) == "spark::sql-performance-tuning"

    def test_spark_versionless_canonical(self) -> None:
        raw = "https://raw.githubusercontent.com/apache/spark/fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4/docs/ml-pipeline.md"
        assert same_document("https://spark.apache.org/docs/4.0.0/ml-pipeline.md", raw)

    def test_delta_canonical_vs_raw(self) -> None:
        canonical = "https://docs.delta.io/latest/best-practices.html"
        raw = "https://raw.githubusercontent.com/delta-io/delta/88c008d55b38fb8f49a9a80982004cb5b06a2b59/docs/src/content/docs/best-practices.mdx"
        assert url_content_key(canonical) == url_content_key(raw)
        assert url_content_key(canonical) == "delta::best-practices"

    def test_airflow_canonical_vs_raw(self) -> None:
        canonical = "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html"
        raw = "https://raw.githubusercontent.com/apache/airflow/3adbbe1c58e4532df1964cb7794805e763816ee8/airflow-core/docs/core-concepts/dags.rst"
        assert same_document(canonical, raw)
        assert url_content_key(canonical) == "airflow::core-concepts/dags"

    def test_non_hex_sha_corpus_still_matches(self) -> None:
        """The test corpus uses a short 'abc' sha — must still normalize."""
        raw = "https://raw.githubusercontent.com/apache/spark/abc/docs/window.md"
        assert same_document("https://spark.apache.org/docs/4.0.0/window.html", raw)

    def test_claude_same_host(self) -> None:
        assert url_content_key("https://platform.claude.com/docs/en/api/messages.md") == ("claude_plat::api/messages")

    def test_index_page_folds(self) -> None:
        canonical = "https://docs.delta.io/latest/optimizations-oss.html"
        raw = "https://raw.githubusercontent.com/delta-io/delta/88c008d55b38fb8f49a9a80982004cb5b06a2b59/docs/src/content/docs/optimizations-oss/index.mdx"
        assert same_document(canonical, raw)


class TestDistinctDocuments:
    """Unrelated documents must NOT collide under the content key."""

    def test_spark_delta_do_not_collide(self) -> None:
        spark = url_content_key("https://spark.apache.org/docs/4.0.0/best-practices.html")
        delta = url_content_key("https://docs.delta.io/latest/best-practices.html")
        assert spark != delta

    def test_different_spark_pages_do_not_collide(self) -> None:
        a = url_content_key("https://spark.apache.org/docs/4.0.0/ml-pipeline.md")
        b = url_content_key("https://spark.apache.org/docs/4.0.0/ml-tuning.md")
        assert a != b

    def test_plain_identifiers_keep_identity(self) -> None:
        assert url_content_key("u1") == "u1"
        assert url_content_key("a") != url_content_key("b")


class TestDetection:
    def test_detect_source(self) -> None:
        assert detect_source("https://spark.apache.org/docs/4.0.0/x.md") == "spark"
        assert detect_source("https://docs.delta.io/latest/x.html") == "delta"
        assert detect_source("https://unknown.example/x.md") is None

    def test_empty_url(self) -> None:
        assert url_content_key("") == ""
