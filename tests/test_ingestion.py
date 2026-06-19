import pytest

from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument, RawDocument
from data_engineering_copilot.services.ingestion import IngestionService


class RecordingCrawler:
    def __init__(self) -> None:
        self.source_names: list[str] = []

    def crawl(self, source: DocumentationSource, max_pages: int, on_event=None):
        self.source_names.append(source.name)
        yield RawDocument(source_name=source.name, url=source.start_urls[0], html="<html></html>")


class SkippingParser:
    def parse(self, raw: RawDocument):
        return None


class UnusedChunker:
    def chunk(self, document):
        raise AssertionError("chunker should not be called for skipped documents")


class UnusedEmbeddings:
    def embed_texts(self, texts):
        raise AssertionError("embeddings should not be called for skipped documents")


class UnusedVectorStore:
    def upsert_chunks(self, chunks, vectors) -> None:
        raise AssertionError("vector store should not be called for skipped documents")


class BatchRecordingEmbeddings:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_texts(self, texts):
        self.batches.append(list(texts))
        return [[0.0] * 384 for _ in texts]


class BatchRecordingVectorStore:
    def __init__(self) -> None:
        self.upserted_chunks: list[list[str]] = []

    def upsert_chunks(self, chunks, vectors) -> None:
        self.upserted_chunks.append([chunk.chunk_id for chunk in chunks])


class SimpleParser:
    def parse(self, raw: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            source_name=raw.source_name,
            title="Test Page",
            url=raw.url,
            text="This is a test document with enough words to be chunked.",
        )


class SimpleChunker:
    def chunk(self, document: ParsedDocument):
        return [
            DocumentChunk(
                chunk_id=f"test:{document.url}",
                source_name=document.source_name,
                title=document.title,
                url=document.url,
                text=document.text,
            )
        ]


def build_service(
    crawler: RecordingCrawler,
    parser=None,
    chunker=None,
    embeddings=None,
    vector_store=None,
) -> IngestionService:
    settings = AppSettings(
        sources=(
            DocumentationSource(
                name="Apache Spark Documentation",
                start_urls=("https://spark.apache.org/docs/latest/",),
                allowed_domains=("spark.apache.org",),
                url_prefixes=("https://spark.apache.org/docs/latest/",),
            ),
            DocumentationSource(
                name="Delta Lake Documentation",
                start_urls=("https://docs.delta.io/latest/",),
                allowed_domains=("docs.delta.io",),
                url_prefixes=("https://docs.delta.io/latest/",),
            ),
        )
    )
    return IngestionService(
        settings=settings,
        crawler=crawler,
        parser=parser or SkippingParser(),
        chunker=chunker or UnusedChunker(),
        embeddings=embeddings or UnusedEmbeddings(),
        vector_store=vector_store or UnusedVectorStore(),
    )


def test_ingest_only_selected_sources():
    crawler = RecordingCrawler()
    service = build_service(crawler)

    total_chunks = service.ingest(source_names=("Delta Lake Documentation",))

    assert total_chunks == 0
    assert crawler.source_names == ["Delta Lake Documentation"]


def test_ingest_batches_chunks_and_flushes_at_end():
    class TwoPageCrawler(RecordingCrawler):
        def crawl(self, source, max_pages, on_event=None):
            self.source_names.append(source.name)
            yield RawDocument(source_name=source.name, url=source.start_urls[0], html="<html></html>")

    crawler = TwoPageCrawler()
    embeddings = BatchRecordingEmbeddings()
    vector_store = BatchRecordingVectorStore()
    service = build_service(
        crawler,
        parser=SimpleParser(),
        chunker=SimpleChunker(),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    total_chunks = service.ingest(source_names=("Apache Spark Documentation",))

    assert total_chunks == 1
    assert len(embeddings.batches) == 1
    assert len(vector_store.upserted_chunks) == 1
    assert vector_store.upserted_chunks[0] == ["test:https://spark.apache.org/docs/latest/"]


def test_ingest_rejects_unknown_source_name():
    crawler = RecordingCrawler()
    service = build_service(crawler)

    with pytest.raises(ValueError, match="Unknown documentation source"):
        service.ingest(source_names=("Missing Docs",))

    assert crawler.source_names == []
