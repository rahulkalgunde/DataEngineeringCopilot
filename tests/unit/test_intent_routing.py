"""Unit tests for query intent routing (api_lookup, code_example)."""

from data_engineering_copilot.services.query_rewriting import QueryRewriter


def _rewriter() -> QueryRewriter:
    rw = QueryRewriter.__new__(QueryRewriter)
    rw._enabled = True
    rw._llm_client = None
    rw._hyde_enabled = False
    return rw


class TestIntentRouting:
    def test_api_lookup_exact_method(self):
        rw = _rewriter()
        assert rw.classify_intent("spark.read.parquet()") == "api_lookup"

    def test_api_lookup_class_method(self):
        rw = _rewriter()
        assert rw.classify_intent("DataFrame.groupBy(*cols)") == "api_lookup"

    def test_api_lookup_spark_sql(self):
        rw = _rewriter()
        assert rw.classify_intent('spark.sql("SELECT * FROM t")') == "api_lookup"

    def test_code_example_pyspark(self):
        rw = _rewriter()
        assert rw.classify_intent("show me a pyspark example") == "code_example"

    def test_code_example_write_code(self):
        rw = _rewriter()
        assert rw.classify_intent("code for writing parquet") == "code_example"

    def test_how_to_still_works(self):
        rw = _rewriter()
        assert rw.classify_intent("how to use spark") == "how_to"

    def test_debugging_still_works(self):
        rw = _rewriter()
        assert rw.classify_intent("why is spark slow") == "debugging"

    def test_comparative_still_works(self):
        rw = _rewriter()
        assert rw.classify_intent("compare spark vs flink") == "comparative"

    def test_factual_fallback(self):
        rw = _rewriter()
        assert rw.classify_intent("what is apache spark") == "factual"


class TestDecomposition:
    def test_api_lookup_decomposition(self):
        rw = _rewriter()
        steps = rw.decompose("spark.read.parquet(path)", intent="api_lookup")
        assert len(steps) >= 2
        assert any("signature" in s.lower() for s in steps)

    def test_code_example_decomposition(self):
        rw = _rewriter()
        steps = rw.decompose("write a pyspark example", intent="code_example")
        assert len(steps) >= 2
        assert any("code example" in s.lower() for s in steps)

    def test_how_to_decomposition(self):
        rw = _rewriter()
        steps = rw.decompose("how to install spark", intent="how_to")
        assert len(steps) >= 2

    def test_debugging_decomposition(self):
        rw = _rewriter()
        steps = rw.decompose("why is my spark job failing", intent="debugging")
        assert len(steps) >= 2
