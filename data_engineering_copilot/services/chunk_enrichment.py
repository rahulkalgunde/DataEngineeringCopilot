"""Lightweight chunk enrichment: source type detection, entity extraction, quality scoring."""

from __future__ import annotations

import logging
import re

from data_engineering_copilot.domain.models import DocumentChunk

logger = logging.getLogger(__name__)

# --- Source type heuristics ---

_API_PATTERNS = [
    re.compile(r"\b(def |class |import |from \S+ import|return |raise |yield )\b"),
    re.compile(r"\b(function|method|constructor|async def|lambda)\b", re.IGNORECASE),
    re.compile(r"\bAPI\b"),
]

_TUTORIAL_PATTERNS = [
    re.compile(r"\b(step \d|first,|second,|next,|then,|finally,|to get started)\b", re.IGNORECASE),
    re.compile(r"\b(install|run |execute |configure|setup|getting started)\b", re.IGNORECASE),
]

_CONFIG_PATTERNS = [
    re.compile(r"\b(configur|setting|property|properties|\.yml|\.yaml|\.env|\.toml)\b", re.IGNORECASE),
    re.compile(r"\b(timeout|max_retries|batch_size|log_level|debug)\b", re.IGNORECASE),
]

_URL_SOURCE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/api[/\-_]", re.IGNORECASE), "api_reference"),
    (re.compile(r"/tutorial|/guide|/howto|/how-to|/getting-started", re.IGNORECASE), "tutorial"),
    (re.compile(r"/config|/settings|/properties", re.IGNORECASE), "configuration"),
    (re.compile(r"/concept|/overview|/intro|/architecture", re.IGNORECASE), "concept"),
]

# --- Entity extraction (lightweight regex) ---

# Domain-specific entity patterns for data engineering docs
_ENTITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("spark", re.compile(r"\b(?:Apache )?Spark\b")),
    ("delta_lake", re.compile(r"\bDelta Lake\b")),
    ("databricks", re.compile(r"\bDatabricks\b")),
    ("kafka", re.compile(r"\b(?:Apache )?Kafka\b")),
    ("flink", re.compile(r"\b(?:Apache )?Flink\b")),
    ("hadoop", re.compile(r"\b(?:Apache )?Hadoop\b")),
    ("hive", re.compile(r"\b(?:Apache )?Hive\b")),
    ("parquet", re.compile(r"\bParquet\b")),
    ("avro", re.compile(r"\bAvro\b")),
    ("hudi", re.compile(r"\bApache Hudi\b")),
    ("iceberg", re.compile(r"\bApache Iceberg\b")),
    ("dbt", re.compile(r"\b[db]t\b")),
    ("airflow", re.compile(r"\b(?:Apache )?Airflow\b")),
    ("spark_sql", re.compile(r"\bSpark SQL\b")),
    ("pyspark", re.compile(r"\bPySpark\b")),
    ("dataframe", re.compile(r"\bDataFrame\b")),
    ("etl", re.compile(r"\bETL\b")),
    ("elt", re.compile(r"\bELT\b")),
]


def _score_content_quality(text: str) -> float:
    """Score chunk text quality from 0.0 (poor) to 1.0 (excellent).

    Heuristics: word count, code presence, heading structure, sentence count,
    alphanumeric ratio.
    """
    if not text or not text.strip():
        return 0.0

    score = 0.0
    words = text.split()
    word_count = len(words)

    # Word count: 20+ words is good, 50+ is great
    if word_count >= 50:
        score += 0.3
    elif word_count >= 20:
        score += 0.2
    elif word_count >= 8:
        score += 0.1

    # Code presence (indicates API/technical content)
    has_code = bool(re.search(r"[=({}\[\];]|def |class |import |#.*\n", text))
    if has_code:
        score += 0.2

    # Heading or section structure
    has_headings = bool(re.search(r"^#{1,4}\s|^\*\*[A-Z]", text, re.MULTILINE))
    if has_headings:
        score += 0.15

    # Sentence count (structured prose)
    sentence_count = len(re.split(r"[.!?]+", text))
    if sentence_count >= 3:
        score += 0.15
    elif sentence_count >= 2:
        score += 0.1

    # Alphanumeric ratio
    alnum_count = sum(1 for c in text if c.isalnum())
    if len(text) > 0 and alnum_count / len(text) > 0.7:
        score += 0.1

    # Penalize very short or single-word text
    if word_count < 3:
        score = max(0.05, score * 0.3)

    return min(1.0, score)


def _detect_source_type(text: str, url: str) -> str:
    """Classify chunk source type from content + URL heuristics."""
    # URL hints first (most reliable)
    for pattern, source_type in _URL_SOURCE_HINTS:
        if pattern.search(url):
            return source_type

    # Content-based heuristics
    api_hits = sum(1 for p in _API_PATTERNS if p.search(text))
    tutorial_hits = sum(1 for p in _TUTORIAL_PATTERNS if p.search(text))
    config_hits = sum(1 for p in _CONFIG_PATTERNS if p.search(text))

    scores = {
        "api_reference": api_hits,
        "tutorial": tutorial_hits,
        "configuration": config_hits,
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] > 0:
        return best

    return "concept"


def _extract_entities(text: str) -> tuple[str, ...]:
    """Extract domain-specific entity names from text using regex patterns."""
    found: set[str] = set()
    for name, pattern in _ENTITY_PATTERNS:
        if pattern.search(text):
            found.add(name)
    return tuple(sorted(found))


def enrich_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Return new DocumentChunk instances with enriched metadata fields populated."""
    enriched: list[DocumentChunk] = []
    for chunk in chunks:
        quality = _score_content_quality(chunk.text)
        source_type = _detect_source_type(chunk.text, chunk.url)
        entities = _extract_entities(chunk.text)
        enriched.append(
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                source_name=chunk.source_name,
                title=chunk.title,
                url=chunk.url,
                text=chunk.text,
                content_hash=chunk.content_hash,
                extracted_entities=entities,
                source_type=source_type,
                content_quality_score=quality,
            )
        )
    return enriched
