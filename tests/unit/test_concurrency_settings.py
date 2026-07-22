"""Tests for configurable concurrency settings validation."""

from __future__ import annotations

from data_engineering_copilot.config.settings import AppSettings, DocumentationSource


def _make_settings(**overrides):
    defaults = dict(
        max_pages_per_source=10,
        ingestion_batch_chunk_size=2,
        processing_concurrency=4,
        parse_concurrency=4,
        chunk_concurrency=4,
        embed_concurrency=4,
        store_concurrency=2,
        embedding_batch_size=32,
        crawl_thread_pool_size=4,
        crawl_async_concurrency=20,
        crawl_async_max_concurrency=40,
        crawl_async_per_domain_concurrency=3,
        sources=(
            DocumentationSource(
                name="test",
                start_urls=("https://example.com",),
                allowed_domains=("example.com",),
            ),
        ),
    )
    defaults.update(overrides)
    return AppSettings(**defaults)


class TestConcurrencySettingsExist:
    def test_parse_concurrency(self):
        s = _make_settings(parse_concurrency=8)
        assert s.parse_concurrency == 8

    def test_chunk_concurrency(self):
        s = _make_settings(chunk_concurrency=6)
        assert s.chunk_concurrency == 6

    def test_embed_concurrency(self):
        s = _make_settings(embed_concurrency=3)
        assert s.embed_concurrency == 3

    def test_store_concurrency(self):
        s = _make_settings(store_concurrency=2)
        assert s.store_concurrency == 2

    def test_crawl_async_concurrency(self):
        s = _make_settings(crawl_async_concurrency=10)
        assert s.crawl_async_concurrency == 10

    def test_crawl_thread_pool_size(self):
        s = _make_settings(crawl_thread_pool_size=8)
        assert s.crawl_thread_pool_size == 8

    def test_processing_concurrency(self):
        s = _make_settings(processing_concurrency=6)
        assert s.processing_concurrency == 6


class TestConcurrencySettingsDefaults:
    def test_parse_concurrency_default(self):
        s = _make_settings()
        assert s.parse_concurrency == 4

    def test_chunk_concurrency_default(self):
        s = _make_settings()
        assert s.chunk_concurrency == 4

    def test_embed_concurrency_default(self):
        s = _make_settings()
        assert s.embed_concurrency == 4

    def test_store_concurrency_default(self):
        s = _make_settings()
        assert s.store_concurrency == 2

    def test_crawl_async_concurrency_default(self):
        s = _make_settings()
        assert s.crawl_async_concurrency == 20
