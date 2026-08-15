"""Phase 6 tests: stream-specific Spark chunking."""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.spark_chunker import chunk_spark_document
from data_engineering_copilot.services.spark_metadata import SparkMetadata


def _metadata(doc_type: str, language: str = "conceptual") -> SparkMetadata:
    return SparkMetadata(
        doc_type=doc_type,
        language=language,
        spark_version="4.0.0",
        module="pyspark.sql.functions" if language == "python" else "",
        source_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        file_path=f"docs/{doc_type}.md",
        license="Apache-2.0",
        deployment_mode="yarn" if doc_type == "guide" else "",
    )


def _document(text: str, title: str = "Doc") -> ParsedDocument:
    return ParsedDocument(
        source_name="Apache Spark 4.0.0",
        title=title,
        url="https://raw.githubusercontent.com/apache/spark/fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4/docs/x.md",
        text=text,
    )


def test_guide_chunking_preserves_hierarchy() -> None:
    md = """# Window Functions

## Ranking

Use dense_rank.

## Frames

Use rangeBetween.
"""
    doc = _document(md, "Window Functions")
    chunks = chunk_spark_document(doc, _metadata("guide"))
    assert len(chunks) >= 3
    assert chunks[0].chunk_type == "text"
    assert chunks[0].doc_type == "guide"
    assert chunks[0].source_commit == "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"


def test_guide_metadata_propagated() -> None:
    md = "# T\nbody\n"
    chunks = chunk_spark_document(_document(md), _metadata("guide"))
    assert chunks
    assert chunks[0].language == "conceptual"
    assert chunks[0].spark_version == "4.0.0"
    assert chunks[0].license == "Apache-2.0"
    assert chunks[0].deployment_mode == "yarn"


def test_api_chunk_propagates_empty_deployment_mode() -> None:
    doc = _document("def foo():\n    return 1\n", "Functions")
    chunks = chunk_spark_document(doc, _metadata("api_reference", language="python"))
    assert chunks
    assert chunks[0].deployment_mode == ""


def test_api_chunking_groups_by_function() -> None:
    code = """def filter(col, f):
    return col

def transform(col, f):
    return col
"""
    doc = _document(code, "functions")
    chunks = chunk_spark_document(doc, _metadata("api_reference", language="python"))
    assert len(chunks) == 2
    assert all(c.chunk_type == "api" for c in chunks)
    assert "def filter" in chunks[0].text
    assert chunks[0].module == "pyspark.sql.functions"


def test_code_example_chunking() -> None:
    code = """from pyspark.sql import functions as F

filtered = orders.filter(F.col("discount") <= 0.20)
result = filtered.agg(F.sum("amount"))
"""
    doc = _document(code, "nested_arrays")
    chunks = chunk_spark_document(doc, _metadata("code_example", language="python"))
    assert chunks
    assert all(c.chunk_type == "code" for c in chunks)
    assert chunks[0].doc_type == "code_example"


def test_empty_document_returns_empty() -> None:
    chunks = chunk_spark_document(_document("   "), _metadata("guide"))
    assert chunks == []


def test_unsupported_doc_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported doc_type"):
        chunk_spark_document(_document("body"), _metadata("bogus"))


def test_deterministic_chunk_ids() -> None:
    doc = _document("def a():\n    pass\ndef b():\n    pass\n", "x")
    chunks1 = chunk_spark_document(doc, _metadata("code_example", language="python"))
    chunks2 = chunk_spark_document(doc, _metadata("code_example", language="python"))
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_short_code_accepted() -> None:
    doc = _document("x = 1\n", "tiny")
    chunks = chunk_spark_document(doc, _metadata("code_example", language="python"))
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "x = 1"


# ---------------------------------------------------------------------------
# sql_function_ref chunking (@ExpressionDescription sources)
# ---------------------------------------------------------------------------

_REGISTRY = """
  expression[ArrayTransform]("transform"),
  expression[ArrayFilter]("filter"),
  expression[ArrayAggregate]("aggregate"),
  expression[ArrayAggregate]("reduce", setAlias = true, Some("3.4.0")),
  expression[ArrayJoin]("array_join"),
"""


def _sql_metadata() -> SparkMetadata:
    return SparkMetadata(
        doc_type="sql_function_ref",
        language="scala",
        spark_version="4.0.0",
        module="",
        source_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        file_path="sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/higherOrderFunctions.scala",
        license="Apache-2.0",
    )


def _sql_document(text: str) -> ParsedDocument:
    return ParsedDocument(
        source_name="Apache Spark 4.0.0",
        title="higherOrderFunctions (scala)",
        url="https://raw.githubusercontent.com/apache/spark/fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4/"
        "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/higherOrderFunctions.scala",
        text=text,
        doc_type="sql_function_ref",
        language="scala",
        spark_version="4.0.0",
        module="",
        source_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        file_path="sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/higherOrderFunctions.scala",
        license="Apache-2.0",
    )


def test_sql_function_chunking_replaces_func_token() -> None:
    text = """
@ExpressionDescription(
  usage = "_FUNC_(expr, func) - Transforms elements in an array.",
  examples = \"""
    Examples:
      > SELECT _FUNC_(array(1, 2, 3), x -> x + 1);
       [2,3,4]
  \""",
  since = "2.4.0",
  group = "lambda_funcs")
case class ArrayTransform(
    argument: Expression,
    function: Expression)
  extends ArrayBasedSimpleHigherOrderFunction with CodegenFallback {
}
"""
    chunks = chunk_spark_document(_sql_document(text), _sql_metadata(), _REGISTRY)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "api"
    assert chunks[0].doc_type == "sql_function_ref"
    assert "_FUNC_" not in chunks[0].text
    assert "transform(expr, func) - Transforms elements" in chunks[0].text
    assert "SELECT transform(array(1, 2, 3)" in chunks[0].text
    assert chunks[0].section_header == "transform (ArrayTransform)"


def test_sql_function_aliases_emit_extra_chunk() -> None:
    text = """
@ExpressionDescription(
  usage = "_FUNC_(expr, start, merge, finish) - Applies a binary operator.",
  examples = \"""
    Examples:
      > SELECT _FUNC_(array(1, 2, 3), 0, (acc, x) -> acc + x);
       6
  \""",
  since = "2.4.0",
  group = "lambda_funcs")
case class ArrayAggregate(
    argument: Expression,
    function: Expression)
  extends ArrayBasedSimpleHigherOrderFunction with CodegenFallback {
}
"""
    chunks = chunk_spark_document(_sql_document(text), _sql_metadata(), _REGISTRY)
    assert len(chunks) == 2
    assert {c.section_header for c in chunks} == {"aggregate (ArrayAggregate)", "reduce (ArrayAggregate)"}
    assert any("aggregate(expr, start, merge, finish)" in c.text for c in chunks)
    assert any("reduce(expr, start, merge, finish)" in c.text for c in chunks)


def test_sql_function_aliases_without_func_token_emit_single_chunk() -> None:
    text = """
@ExpressionDescription(
  usage = "expr1 % expr2, or mod(expr1, expr2) - Returns the remainder after `expr1`/`expr2`.",
  examples = \"""
    Examples:
      > SELECT 2 % 1.8;
       0.2
  \""",
  since = "1.4.0",
  group = "math_funcs")
case class Remainder(
    left: Expression,
    right: Expression)
  extends BinaryArithmetic {
}
"""
    chunks = chunk_spark_document(_sql_document(text), _sql_metadata(), _REGISTRY)
    assert len(chunks) == 1
    assert chunks[0].section_header == "Remainder"
    assert "mod" in chunks[0].text


def test_sql_function_unresolved_keeps_func_literal() -> None:
    text = """
@ExpressionDescription(
  usage = "_FUNC_(x) - A mystery function.",
  examples = \"""
    Examples:
      > SELECT _FUNC_(1);
       1
  \""")
case class MysteryFunction(argument: Expression)
"""
    chunks = chunk_spark_document(_sql_document(text), _sql_metadata(), _REGISTRY)
    assert len(chunks) == 1
    assert "_FUNC_" in chunks[0].text
    assert chunks[0].section_header == "MysteryFunction"


def test_sql_function_without_registry_keeps_literal() -> None:
    text = """
@ExpressionDescription(
  usage = "_FUNC_(expr) - Sorts the input array.")
case class ArraySort(argument: Expression)
"""
    chunks = chunk_spark_document(_sql_document(text), _sql_metadata(), None)
    assert len(chunks) == 1
    assert "_FUNC_" in chunks[0].text


def test_sql_function_annotation_free_falls_back_to_blank_lines() -> None:
    text = "package org.apache.spark\n\nobject Helper {\n  val x = 1\n}\n"
    chunks = chunk_spark_document(_sql_document(text), _sql_metadata(), _REGISTRY)
    assert chunks
    assert all("_FUNC_" not in c.text for c in chunks)


def test_sql_function_metadata_propagated() -> None:
    text = """
@ExpressionDescription(
  usage = "_FUNC_(expr, func) - Transforms elements in an array.")
case class ArrayTransform(argument: Expression)
"""
    chunks = chunk_spark_document(_sql_document(text), _sql_metadata(), _REGISTRY)
    assert chunks
    chunk = chunks[0]
    assert chunk.doc_type == "sql_function_ref"
    assert chunk.language == "scala"
    assert chunk.source_commit == "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"
    assert chunk.license == "Apache-2.0"
    assert chunk.spark_version == "4.0.0"


def test_sql_function_deterministic_chunk_ids() -> None:
    text = """
@ExpressionDescription(
  usage = "_FUNC_(expr, func) - Transforms elements in an array.")
case class ArrayTransform(argument: Expression)
"""
    chunks1 = chunk_spark_document(_sql_document(text), _sql_metadata(), _REGISTRY)
    chunks2 = chunk_spark_document(_sql_document(text), _sql_metadata(), _REGISTRY)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_parse_function_registry_merges_builder_and_internal() -> None:
    from data_engineering_copilot.services.spark_chunker import parse_function_registry

    registry = """
  expression[Abs]("abs"),
  expressionBuilder("explode", ExplodeExpressionBuilder),
  registerInternalExpression[TimestampAdd]("timestampadd")
"""
    names = parse_function_registry(registry)
    assert names["Abs"] == ("abs",)
    assert names["ExplodeExpressionBuilder"] == ("explode",)
    assert names["TimestampAdd"] == ("timestampadd",)
