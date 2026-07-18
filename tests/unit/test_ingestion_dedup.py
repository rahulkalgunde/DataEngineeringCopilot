"""Unit tests for content-hash-based duplicate page detection in ingestion."""

import hashlib

from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
from data_engineering_copilot.domain.models import DocumentChunk, IngestionEvent, ParsedDocument, RawDocument
from data_engineering_copilot.services.ingestion import IngestionService

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class SinglePageCrawler:
    """Yields one page per source and records the source name."""

    def __init__(self) -> None:
        self.source_names: list[str] = []

    def crawl(self, source: DocumentationSource, max_pages: int, on_event=None):
        self.source_names.append(source.name)
        yield RawDocument(
            source_name=source.name,
            url=source.start_urls[0],
            html="<html><body><p>Sample content for chunking.</p></body></html>",
        )


class FixedTextParser:
    """Returns a ParsedDocument with configurable text."""

    def __init__(self, text: str = "Default test document text for parsing and chunking.") -> None:
        self._text = text

    def parse(self, raw: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            source_name=raw.source_name,
            title="Test Page",
            url=raw.url,
            text=self._text,
        )


class SimpleChunker:
    """Produces a single chunk per document with deterministic chunk_id."""

    def chunk(self, document: ParsedDocument):
        return [
            DocumentChunk(
                chunk_id=f"test:{document.url}:0001",
                source_name=document.source_name,
                title=document.title,
                url=document.url,
                text=document.text,
            )
        ]


class RecordingEmbeddings:
    """Records every batch of texts passed to embed_texts."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_texts(self, texts):
        self.batches.append(list(texts))
        return [[0.0] * 768 for _ in texts]


class RecordingVectorStore:
    """Records upserted chunks and supports URL-based content_hash lookup."""

    def __init__(self, existing_url_hashes: dict[str, str] | None = None) -> None:
        """
        Parameters
        ----------
        existing_url_hashes:
            Dictionary mapping URL -> content_hash that simulates what's already
            stored in the vector DB.
        """
        self.upserted_chunks: list[list[str]] = []
        self._url_hashes = dict(existing_url_hashes or {})
        self.deleted_urls: list[str] = []

    def upsert_chunks(self, chunks, vectors) -> None:
        self.upserted_chunks.append([chunk.chunk_id for chunk in chunks])

    def get_content_hash_for_url(self, url: str) -> str | None:
        """Simulate querying the vector store for a page's stored content hash."""
        return self._url_hashes.get(url)

    def set_content_hash(self, url: str, content_hash: str) -> None:
        self._url_hashes[url] = content_hash

    def delete_by_url(self, url: str) -> None:
        """Simulate deleting all chunks for a given URL."""
        self.deleted_urls.append(url)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_service(
    crawler=None,
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
        )
    )
    return IngestionService(
        settings=settings,
        crawler=crawler or SinglePageCrawler(),
        parser=parser or FixedTextParser(),
        chunker=chunker or SimpleChunker(),
        embeddings=embeddings or RecordingEmbeddings(),
        vector_store=vector_store or RecordingVectorStore(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embeds_and_upserts_when_no_existing_chunks_for_url():
    """When the vector store has no prior chunks for a URL, embedding should proceed."""
    crawler = SinglePageCrawler()
    embeddings = RecordingEmbeddings()
    vector_store = RecordingVectorStore(existing_url_hashes={})  # empty DB

    service = build_service(
        crawler=crawler,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    total = service.ingest(source_names=("Apache Spark Documentation",))

    assert total == 1
    assert len(embeddings.batches) == 1, "Should embed since URL is new"
    assert len(vector_store.upserted_chunks) == 1


def test_embeds_and_upserts_when_content_hash_changed():
    """When the URL exists but content_hash differs from stored, embedding should proceed."""
    old_text = "Old documentation text that has changed."
    new_text = "New documentation text that is different now."

    old_hash = _compute_hash(old_text)
    vector_store = RecordingVectorStore(existing_url_hashes={"https://spark.apache.org/docs/latest/": old_hash})
    embeddings = RecordingEmbeddings()

    service = build_service(
        parser=FixedTextParser(text=new_text),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    total = service.ingest(source_names=("Apache Spark Documentation",))

    assert total == 1
    assert len(embeddings.batches) == 1, "Should embed because content changed"
    assert len(vector_store.upserted_chunks) == 1


def test_skips_embedding_when_content_hash_unchanged():
    """When the URL exists and content_hash matches, embedding and upsert should be skipped."""
    text = "This documentation page has not changed."
    content_hash = _compute_hash(text)

    vector_store = RecordingVectorStore(existing_url_hashes={"https://spark.apache.org/docs/latest/": content_hash})
    embeddings = RecordingEmbeddings()

    service = build_service(
        parser=FixedTextParser(text=text),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    total = service.ingest(source_names=("Apache Spark Documentation",))

    assert total == 0, "No new chunks should be indexed for duplicate content"
    assert len(embeddings.batches) == 0, "Embedding should be skipped"
    assert len(vector_store.upserted_chunks) == 0, "Upsert should be skipped"


def test_emits_page_skipped_duplicate_event():
    """When a page is skipped as duplicate, a 'page_skipped_duplicate' event should fire."""
    text = "Duplicate page content."
    content_hash = _compute_hash(text)

    vector_store = RecordingVectorStore(existing_url_hashes={"https://spark.apache.org/docs/latest/": content_hash})
    events: list[IngestionEvent] = []

    service = build_service(
        parser=FixedTextParser(text=text),
        vector_store=vector_store,
    )

    service.ingest(
        source_names=("Apache Spark Documentation",),
        on_event=events.append,
    )

    skipped_events = [e for e in events if e.event_type == "page_skipped_duplicate"]
    assert len(skipped_events) == 1
    assert skipped_events[0].url == "https://spark.apache.org/docs/latest/"


def test_mixed_pages_some_new_some_duplicate():
    """Multiple pages: new pages embedded, duplicates skipped."""

    text_a = "Unique page A content."
    text_b = "Unique page B content."
    text_dup = "Page that already exists."
    dup_hash = _compute_hash(text_dup)

    class MultiPageCrawler:
        def __init__(self) -> None:
            self.source_names: list[str] = []

        def crawl(self, source, max_pages, on_event=None):
            self.source_names.append(source.name)
            urls = [
                "https://spark.apache.org/docs/latest/a",
                "https://spark.apache.org/docs/latest/b",
                "https://spark.apache.org/docs/latest/dup",
            ]
            texts = [text_a, text_b, text_dup]
            for url, t in zip(urls, texts, strict=False):
                yield RawDocument(
                    source_name=source.name,
                    url=url,
                    html=f"<html><body><p>{t}</p></body></html>",
                )

    vector_store = RecordingVectorStore(existing_url_hashes={"https://spark.apache.org/docs/latest/dup": dup_hash})
    embeddings = RecordingEmbeddings()

    class PerUrlParser:
        """Returns different text per URL to simulate real parser behavior."""

        _url_texts = {
            "https://spark.apache.org/docs/latest/a": text_a,
            "https://spark.apache.org/docs/latest/b": text_b,
            "https://spark.apache.org/docs/latest/dup": text_dup,
        }

        def parse(self, raw: RawDocument) -> ParsedDocument:
            return ParsedDocument(
                source_name=raw.source_name,
                title="Test",
                url=raw.url,
                text=self._url_texts.get(raw.url, raw.url),
            )

    service = build_service(
        crawler=MultiPageCrawler(),
        parser=PerUrlParser(),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    total = service.ingest(source_names=("Apache Spark Documentation",))

    assert total == 2, "Only the two new pages should be indexed"
    assert len(embeddings.batches) > 0
    # Verify two pages were embedded (could be batched together)
    all_embedded_texts = [t for batch in embeddings.batches for t in batch]
    assert len(all_embedded_texts) == 2
    assert text_a in all_embedded_texts
    assert text_b in all_embedded_texts
    assert text_dup not in all_embedded_texts

    # Verify upserted chunks count
    all_upserted = [cid for batch in vector_store.upserted_chunks for cid in batch]
    assert len(all_upserted) == 2


def test_null_safe_when_vector_store_has_no_get_content_hash_method():
    """Graceful degradation: if vector store lacks get_content_hash_for_url, proceed normally."""

    class MinimalVectorStore:
        def upsert_chunks(self, chunks, vectors) -> None:
            pass

    crawler = SinglePageCrawler()
    embeddings = RecordingEmbeddings()

    service = build_service(
        crawler=crawler,
        embeddings=embeddings,
        vector_store=MinimalVectorStore(),
    )

    total = service.ingest(source_names=("Apache Spark Documentation",))

    assert total == 1
    assert len(embeddings.batches) == 1, "Should fall back to embedding when method missing"


def test_computes_consistent_content_hash():
    """content_hash computation must be deterministic and based on parsed text."""
    text = "Hello world"
    h1 = _compute_hash(text)
    h2 = _compute_hash(text)
    assert h1 == h2
    assert h1 != _compute_hash("Different text")


# ---------------------------------------------------------------------------
# Ghost chunk deletion tests
# ---------------------------------------------------------------------------


def test_deletes_ghost_chunks_when_page_shrinks():
    """When a page's content changes (hash mismatch) and it now produces fewer chunks,
    the ingestion service must delete old chunks for that URL before upserting new ones."""
    old_text = "This is the old version with many many many many content words that will generate more chunks."
    new_text = "Short new version."

    old_hash = _compute_hash(old_text)

    class ShrinkingChunker:
        """Simulates a page that used to produce 10 chunks but now produces only 3."""

        def __init__(self, old_chunk_count: int = 10, new_chunk_count: int = 3) -> None:
            self.old_chunk_count = old_chunk_count
            self.new_chunk_count = new_chunk_count

        def chunk(self, document: ParsedDocument):
            """Produce new_chunk_count chunks (simulating the shrunk page)."""
            return [
                DocumentChunk(
                    chunk_id=f"shrinking-source:{document.url}:{i:04d}",
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=f"{document.text} chunk {i}",
                )
                for i in range(1, self.new_chunk_count + 1)
            ]

    vector_store = RecordingVectorStore(existing_url_hashes={"https://spark.apache.org/docs/latest/": old_hash})
    embeddings = RecordingEmbeddings()

    service = build_service(
        parser=FixedTextParser(text=new_text),
        chunker=ShrinkingChunker(old_chunk_count=10, new_chunk_count=3),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    total = service.ingest(source_names=("Apache Spark Documentation",))

    # Verify that delete_by_url was called for the shrinking page
    assert "https://spark.apache.org/docs/latest/" in vector_store.deleted_urls, (
        "Ghost chunks should be deleted before upserting new chunks when content changes"
    )

    # Verify that the new (fewer) chunks were still embedded and upserted
    assert total == 3, "All 3 new chunks should be indexed"
    assert len(embeddings.batches) == 1, "New chunks should be embedded"
    assert len(vector_store.upserted_chunks) == 1, "New chunks should be upserted"
    assert len(vector_store.upserted_chunks[0]) == 3, "Exactly 3 new chunks upserted"


def test_deletes_ghost_chunks_when_page_content_changes_but_chunk_count_same():
    """When a page's content changes but produces the same number of chunks,
    ghost chunks should still be deleted to ensure clean overwrite."""
    old_text = "Old version of the page with different wording."
    new_text = "New version of the page with updated wording."

    old_hash = _compute_hash(old_text)

    class ThreeChunker:
        def chunk(self, document: ParsedDocument):
            return [
                DocumentChunk(
                    chunk_id=f"source:{document.url}:{i:04d}",
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=f"{document.text} {i}",
                )
                for i in range(1, 4)
            ]

    vector_store = RecordingVectorStore(existing_url_hashes={"https://spark.apache.org/docs/latest/": old_hash})
    embeddings = RecordingEmbeddings()

    service = build_service(
        parser=FixedTextParser(text=new_text),
        chunker=ThreeChunker(),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    total = service.ingest(source_names=("Apache Spark Documentation",))

    assert "https://spark.apache.org/docs/latest/" in vector_store.deleted_urls, (
        "delete_by_url should be called even when chunk count is same, "
        "to handle potential chunk ID distribution changes"
    )
    assert total == 3


def test_no_delete_when_page_unchanged():
    """When a page's content hash is unchanged, delete_by_url must NOT be called."""
    text = "Stable documentation page."
    content_hash = _compute_hash(text)

    vector_store = RecordingVectorStore(existing_url_hashes={"https://spark.apache.org/docs/latest/": content_hash})
    embeddings = RecordingEmbeddings()

    service = build_service(
        parser=FixedTextParser(text=text),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    service.ingest(source_names=("Apache Spark Documentation",))

    assert len(vector_store.deleted_urls) == 0, "No delete should occur when page content is unchanged"
    assert len(embeddings.batches) == 0, "No embedding for unchanged page"


def test_no_delete_when_url_is_new():
    """When a URL has no stored hash (first-time ingestion), delete_by_url must NOT be called."""
    vector_store = RecordingVectorStore(existing_url_hashes={})
    embeddings = RecordingEmbeddings()

    service = build_service(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    service.ingest(source_names=("Apache Spark Documentation",))

    assert len(vector_store.deleted_urls) == 0, "No delete should occur for a brand-new URL"
    assert len(embeddings.batches) == 1, "New page should be embedded"
    assert len(vector_store.upserted_chunks) == 1


def test_content_hash_stamped_on_chunks():
    """Every chunk produced during ingestion must carry the content_hash in its payload."""
    text = "Page content for hash stamping."
    computed_hash = _compute_hash(text)

    class CaptureChunker:
        def chunk(self, document: ParsedDocument):
            return [
                DocumentChunk(
                    chunk_id=f"capture:{document.url}:{i:04d}",
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=f"{document.text} part {i}",
                )
                for i in range(1, 3)
            ]

    vector_store = RecordingVectorStore(existing_url_hashes={})
    embeddings = RecordingEmbeddings()

    # We need to capture the actual chunk objects from upsert
    orig_upsert = vector_store.upsert_chunks
    captured_chunks: list[DocumentChunk] = []

    def capturing_upsert(chunks, vectors):
        captured_chunks.extend(list(chunks))
        orig_upsert(chunks, vectors)

    vector_store.upsert_chunks = capturing_upsert

    service = build_service(
        parser=FixedTextParser(text=text),
        chunker=CaptureChunker(),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    service.ingest(source_names=("Apache Spark Documentation",))

    assert len(captured_chunks) == 2, "Both chunks should be upserted"
    for chunk in captured_chunks:
        assert chunk.content_hash == computed_hash, f"Chunk {chunk.chunk_id} must carry the content_hash"


def test_null_safe_when_vector_store_lacks_delete_by_url():
    """Graceful degradation: if the vector store doesn't have delete_by_url, ingestion proceeds."""

    class MinimalVectorStore:
        def upsert_chunks(self, chunks, vectors) -> None:
            pass

        def get_content_hash_for_url(self, url: str) -> str | None:
            return None

    _compute_hash("old content that differs")

    vector_store = MinimalVectorStore()
    embeddings = RecordingEmbeddings()

    service = build_service(
        parser=FixedTextParser(text="new different content"),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    # Should not raise an exception even though delete_by_url is missing
    total = service.ingest(source_names=("Apache Spark Documentation",))
    assert total == 1
