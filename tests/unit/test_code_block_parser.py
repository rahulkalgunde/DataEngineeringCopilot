"""Unit tests for CodeBlockParser."""

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.services.code_block_parser import CodeBlockParser


def _chunk(text: str, **kwargs) -> DocumentChunk:
    defaults = dict(chunk_id="c1", source_name="Spark", title="Code", url="http://x", text=text)
    defaults.update(kwargs)
    return DocumentChunk(**defaults)


class TestCodeBlockParser:
    def test_non_code_chunks_unchanged(self):
        chunk = _chunk("Regular text about Spark.", chunk_type="text")
        result = CodeBlockParser(enabled=True).extract([chunk])
        assert len(result) == 1
        assert result[0].chunk_type == "text"

    def test_already_code_unchanged(self):
        chunk = _chunk("```python\ndef foo(): pass\n```", chunk_type="code")
        result = CodeBlockParser(enabled=True).extract([chunk])
        assert len(result) == 1
        assert result[0].chunk_type == "code"

    def test_code_heavy_chunk_detected(self):
        code_text = (
            "## Example\n\n"
            "```python\n"
            "from pyspark.sql import SparkSession\n\n"
            "def create_session(app_name):\n"
            "    spark = SparkSession.builder.appName(app_name).getOrCreate()\n"
            "    return spark\n\n"
            "def process_data(df):\n"
            "    return df.filter(df.age > 21)\n"
            "```\n"
        )
        chunk = _chunk(code_text, section_header="Example")
        result = CodeBlockParser(enabled=True).extract([chunk])
        assert result[0].chunk_type == "code"
        assert "Source:" in result[0].text

    def test_scope_prefix_includes_section(self):
        code_text = "```python\ndef foo(): pass\n```"
        chunk = _chunk(code_text, section_header="My Section")
        result = CodeBlockParser(enabled=True).extract([chunk])
        assert "My Section" in result[0].text

    def test_disabled_returns_unchanged(self):
        code_text = "```python\ndef foo(): pass\n```"
        chunk = _chunk(code_text)
        result = CodeBlockParser(enabled=False).extract([chunk])
        assert result[0].chunk_type == "text"

    def test_ast_splitting(self):
        code = (
            "```python\n"
            "def create_session(app_name):\n"
            "    return None\n\n"
            "class MyPipeline:\n"
            "    def run(self): pass\n\n"
            "def process_data(df):\n"
            "    return df\n"
            "```"
        )
        chunk = _chunk(code, chunk_type="text")
        result = CodeBlockParser(enabled=True, max_code_lines=5).extract([chunk])
        # Should split into separate function/class chunks
        assert len(result) >= 2
        assert all(c.chunk_type == "code" for c in result)
