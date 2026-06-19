import pytest

from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
from data_engineering_copilot.domain.models import RawDocument
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


def build_service(crawler: RecordingCrawler) -> IngestionService:
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
        parser=SkippingParser(),
        chunker=UnusedChunker(),
        embeddings=UnusedEmbeddings(),
        vector_store=UnusedVectorStore(),
    )


def test_ingest_only_selected_sources():
    crawler = RecordingCrawler()
    service = build_service(crawler)

    total_chunks = service.ingest(source_names=("Delta Lake Documentation",))

    assert total_chunks == 0
    assert crawler.source_names == ["Delta Lake Documentation"]


def test_ingest_rejects_unknown_source_name():
    crawler = RecordingCrawler()
    service = build_service(crawler)

    with pytest.raises(ValueError, match="Unknown documentation source"):
        service.ingest(source_names=("Missing Docs",))

    assert crawler.source_names == []
