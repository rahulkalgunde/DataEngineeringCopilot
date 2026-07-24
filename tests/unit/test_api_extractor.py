"""Unit tests for ApiDocExtractor."""

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.services.api_extractor import ApiDocExtractor


def _chunk(text: str, **kwargs) -> DocumentChunk:
    defaults = dict(chunk_id="c1", source_name="Spark", title="API", url="http://x", text=text)
    defaults.update(kwargs)
    return DocumentChunk(**defaults)


class TestApiDocExtractor:
    def test_detects_method_call(self):
        chunk = _chunk("Use DataFrame.groupBy(*cols) to group data.")
        result = ApiDocExtractor(enabled=True).extract([chunk])
        assert result[0].chunk_type == "api"
        assert "groupBy" in result[0].text

    def test_detects_pyspark_chain(self):
        chunk = _chunk("spark.read.parquet('path') reads parquet files.")
        result = ApiDocExtractor(enabled=True).extract([chunk])
        assert result[0].chunk_type == "api"
        assert "parquet" in result[0].text

    def test_detects_def_statement(self):
        chunk = _chunk("def create_dataframe(data, schema):")
        result = ApiDocExtractor(enabled=True).extract([chunk])
        assert result[0].chunk_type == "api"
        assert "create_dataframe" in result[0].text

    def test_detects_params(self):
        chunk = _chunk("DataFrame.groupBy(*cols)\n:param cols: Column names")
        result = ApiDocExtractor(enabled=True).extract([chunk])
        assert "cols" in result[0].text

    def test_no_api_content_unchanged(self):
        chunk = _chunk("This is just regular text about Apache Spark.", chunk_type="text")
        result = ApiDocExtractor(enabled=True).extract([chunk])
        assert result[0].chunk_type == "text"
        assert result[0].text == chunk.text

    def test_already_tagged_unchanged(self):
        chunk = _chunk("API content", chunk_type="api")
        result = ApiDocExtractor(enabled=True).extract([chunk])
        assert result[0].chunk_type == "api"
        assert result[0].text == chunk.text

    def test_disabled_returns_unchanged(self):
        chunk = _chunk("DataFrame.groupBy(*cols)")
        result = ApiDocExtractor(enabled=False).extract([chunk])
        assert result[0].chunk_type == "text"
