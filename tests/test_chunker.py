from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.chunker import DocumentChunker, ChunkingStrategy


def test_chunker_preserves_required_metadata():
    document = ParsedDocument(
        source_name="Apache Spark Documentation",
        title="Spark SQL",
        url="https://spark.apache.org/docs/latest/sql-programming-guide.html",
        text=" ".join(f"word{i}" for i in range(30)),
    )

    chunks = DocumentChunker(
        chunk_size_words=10,
        overlap_words=2,
        strategy=ChunkingStrategy.FIXED_SIZE,
        min_chunk_words=5,
    ).chunk(document)

    assert len(chunks) == 4
    assert chunks[0].source_name == "Apache Spark Documentation"
    assert chunks[0].title == "Spark SQL"
    assert chunks[0].url == document.url
    assert chunks[0].chunk_id.startswith("apache-spark-documentation:")

