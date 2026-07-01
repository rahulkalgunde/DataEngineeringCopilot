"""Async ingestion task using Crawl4AI and Qdrant.

The task is executed by Celery.  It receives a list of URLs, crawls them
asynchronously with ``AsyncWebCrawler`` (which handles rate‑limiting,
user‑agent rotation and markdown generation), embeds the resulting text
with the project's ``SentenceTransformerEmbeddings`` and upserts the
chunks into the Qdrant vector store.
"""

from __future__ import annotations

import asyncio
from typing import List

from celery import Celery
from crawl4ai import AsyncWebCrawler

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker

# Celery app – broker URL comes from the Docker‑Compose Redis service
app = Celery("tasks", broker=settings.redis_url)

async def _run_async_crawl(urls: List[str]):
    """Crawl a list of URLs concurrently and return the raw Crawl4AI results."""
    async with AsyncWebCrawler(verbose=True) as crawler:
        tasks = [crawler.arun(url=url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

@app.task
def execute_background_ingestion(urls: List[str]):
    """Celery entry point.

    The function runs the async crawler, parses the markdown, chunks the
    text, embeds the chunks and finally upserts them into Qdrant.
    """
    # Run the async crawler in the current event loop
    loop = asyncio.get_event_loop()
    raw_docs = loop.run_until_complete(_run_async_crawl(urls))

    # Initialise services
    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model_name,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    chunker = DocumentChunker(
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
    )

    processed = 0
    for doc in raw_docs:
        if not getattr(doc, "success", False):
            continue

        # ``doc.markdown`` contains the generated markdown text
        text = doc.markdown
        # Chunk the markdown into DocumentChunk objects
        chunks = chunker.chunk(
            # The chunker expects a ``ParsedDocument``‑like object; we can
            # reuse the ``DocumentChunk`` dataclass for the minimal fields.
            # Here we construct a lightweight object on‑the‑fly.
            type("TmpDoc", (), {
                "source_name": "crawl4ai",
                "title": doc.title or doc.url,
                "url": doc.url,
                "text": text,
            })()
        )
        # Embed and upsert
        embeddings = embedder.embed_texts([c.text for c in chunks])
        vector_store.upsert_chunks(chunks, embeddings)
        processed += 1

    return {"status": "INGESTION_COMPLETED", "processed_count": processed}