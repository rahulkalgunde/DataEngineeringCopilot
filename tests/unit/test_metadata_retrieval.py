"""Phase 8 tests: metadata-aware retrieval, constraint extraction, and merging."""

from __future__ import annotations

from data_engineering_copilot.domain.models import DocumentChunk, RetrievalFilters, RetrievedChunk
from data_engineering_copilot.services.async_rag import merge_retrieval_results
from data_engineering_copilot.services.query_rewriting import extract_retrieval_constraints


def _chunk(chunk_id: str, text: str, doc_type: str = "", language: str = "", module: str = "") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name="src",
        title="T",
        url=f"http://x/{chunk_id}",
        text=text,
        doc_type=doc_type,
        language=language,
        module=module,
    )


def _retrieved(chunk: DocumentChunk, confidence: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, distance=1.0 - confidence, confidence=confidence)


# ------------------------------------------------------------------
# RetrievalFilters
# ------------------------------------------------------------------


def test_retrieval_filters_default_is_empty() -> None:
    filters = RetrievalFilters()
    assert filters.is_empty is True


def test_retrieval_filters_non_empty() -> None:
    filters = RetrievalFilters(modules=("pyspark.sql.functions",))
    assert filters.is_empty is False


# ------------------------------------------------------------------
# extract_retrieval_constraints
# ------------------------------------------------------------------


def test_extract_dotted_module() -> None:
    filters = extract_retrieval_constraints("How does pyspark.sql.functions.filter work?")
    assert "pyspark.sql.functions.filter" in filters.modules
    assert "pyspark.sql.functions" in filters.preferred_modules


def test_extract_version() -> None:
    filters = extract_retrieval_constraints("Spark 4.0.0 window functions")
    assert "4.0.0" in filters.versions


def test_extract_module_term_is_soft_preference() -> None:
    filters = extract_retrieval_constraints("filter transform aggregate array of structs")
    # Term-derived modules are soft preferences, never hard filters.
    assert "pyspark.sql.functions" in filters.preferred_modules
    assert filters.modules == ()


def test_extract_dense_rank_is_soft_preference() -> None:
    filters = extract_retrieval_constraints("calculate dense_rank per category")
    assert "pyspark.sql.functions" in filters.preferred_modules
    assert filters.modules == ()


def test_extract_no_latest_version() -> None:
    filters = extract_retrieval_constraints("Spark latest window functions")
    assert "latest" not in " ".join(filters.versions)


def test_extract_empty_query() -> None:
    filters = extract_retrieval_constraints("")
    assert filters.is_empty


# ------------------------------------------------------------------
# merge_retrieval_results
# ------------------------------------------------------------------


def test_merge_ranks_original_query_first() -> None:
    c1 = _chunk("c1", "window dense_rank")
    c2 = _chunk("c2", "spark sql")
    original = [_retrieved(c1, 0.4)]
    expansion = [_retrieved(c2, 0.9)]
    merged = merge_retrieval_results([original, expansion], "window dense_rank")
    # c1 is in the original query set so gets the bonus and ranks first.
    assert merged[0].chunk.chunk_id == "c1"


def test_merge_deduplicates_across_queries() -> None:
    c1 = _chunk("c1", "window functions")
    original = [_retrieved(c1, 0.5)]
    expansion = [_retrieved(c1, 0.7)]
    merged = merge_retrieval_results([original, expansion], "window")
    assert len(merged) == 1
    assert merged[0].chunk.chunk_id == "c1"


def test_merge_boosts_original_query_results() -> None:
    # The original-query result receives the generic rank-fusion bonus.
    filter_chunk = DocumentChunk(
        chunk_id="filter",
        source_name="src",
        title="builtin",
        url="http://x/builtin.py",
        text='def filter(\n    col: "ColumnOrName",\n) -> Column:\n    """Returns an array of elements for which a predicate holds in a given array."""',
    )
    inline_chunk = DocumentChunk(
        chunk_id="inline",
        source_name="src",
        title="builtin",
        url="http://x/builtin.py",
        text='def inline(col: "ColumnOrName") -> Column:\n    """Explodes an array of structs into a table."""',
    )
    # The original query is the first result list, so its chunk wins the tie.
    original = [_retrieved(filter_chunk, 0.5), _retrieved(inline_chunk, 0.9)]
    merged = merge_retrieval_results([original], "filter array of structs where discount > 0.20")
    assert merged[0].chunk.chunk_id == "filter"


def test_merge_empty_results() -> None:
    assert merge_retrieval_results([], "q") == []


def test_merge_single_result() -> None:
    c1 = _chunk("c1", "hello world")
    merged = merge_retrieval_results([[_retrieved(c1)]], "hello")
    assert len(merged) == 1


# ------------------------------------------------------------------
# Metadata filter building (via store static helper)
# ------------------------------------------------------------------


def test_build_query_filter_none() -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    assert AsyncQdrantVectorStore._build_query_filter(None) is None


def test_build_query_filter_empty() -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    assert AsyncQdrantVectorStore._build_query_filter(RetrievalFilters()) is None


def test_build_query_filter_conditions() -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    filters = RetrievalFilters(doc_types=("guide",), languages=("python",))
    conditions = AsyncQdrantVectorStore._build_query_filter(filters)
    assert conditions is not None
    assert len(conditions) == 2


# ------------------------------------------------------------------
# RewrittenQuery carries filters
# ------------------------------------------------------------------


def test_rewritten_query_has_filters_default() -> None:
    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    rewriter = QueryRewriter(llm_client=None, enabled=False)
    rewritten = rewriter.rewrite("How does pyspark.sql.Window work in Spark 4.0.0?")
    assert rewritten.filters is not None
    assert "pyspark.sql.window" in rewritten.filters.modules


def test_spark_retrieval_variants_window() -> None:
    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    variants = QueryRewriter._spark_retrieval_variants(
        "calculate 7-day rolling total spend and dense_rank per category"
    )
    assert any("RANGE BETWEEN" in v for v in variants)
    assert any("dense_rank" in v for v in variants)


def test_spark_retrieval_variants_array() -> None:
    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    variants = QueryRewriter._spark_retrieval_variants(
        "filter items array of structs where discount > 0.20 then aggregate net_total"
    )
    assert any("filter" in v and "transform" in v for v in variants)
    assert any("without explode" in v for v in variants)
    assert any("net_total" in v for v in variants)
