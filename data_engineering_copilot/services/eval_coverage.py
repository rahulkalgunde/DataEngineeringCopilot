"""Corpus-coverage validation for evaluation datasets.

Verifies that every in-scope ``recall`` row's expected URLs resolve to indexed
chunks in a generation and that its expected terms are actually present in the
corpus. This is the guard that prevents eval rows from silently targeting
content that is not in the active index (e.g. a source dropped from a pinned
generation).

Reading the generation's ``chunks.jsonl`` (plus ``coverage.json`` when present)
builds an in-memory url -> text index. 15k chunks is well within memory.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_generation_root(generation: str, data_root: Path | None = None) -> Path | None:
    """Locate a generation's corpus directory under ``data/``.

    Checks ``pinned_corpus/<gen>`` then ``spark_corpus/<gen>`` (and
    ``data/spark_corpus/<gen>``-style legacy layouts). Returns ``None`` when no
    corpus directory with a ``chunks.jsonl`` is found.
    """
    data_root = data_root or (Path(__file__).resolve().parents[2] / "data")
    candidates = [
        data_root / "pinned_corpus" / generation,
        data_root / "spark_corpus" / generation,
        data_root / "spark_corpus" / f"spark-v{generation}",
    ]
    for cand in candidates:
        if (cand / "chunks.jsonl").exists():
            return cand
    return None


class CoverageValidator:
    """Validate eval rows against a generation's indexed corpus."""

    def __init__(self, generation_root: Path) -> None:
        self._generation_root = generation_root
        self._urls: set[str] = set()
        self._url_source: dict[str, str] = {}
        self._source_names: set[str] = set()
        self._corpus_text: str = ""
        self._source_text: dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        chunks_path = self._generation_root / "chunks.jsonl"
        texts: list[str] = []
        source_parts: dict[str, list[str]] = {}
        if chunks_path.exists():
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    url = (d.get("url") or "").strip().rstrip("/")
                    src = str(d.get("source_name") or "")
                    text = d.get("text") or ""
                    if url:
                        self._urls.add(url)
                        self._url_source[url] = src
                    if src:
                        self._source_names.add(src)
                        source_parts.setdefault(src, []).append(text)
                    texts.append(text)
        self._corpus_text = " ".join(texts).lower()
        self._source_text = {src: " ".join(parts).lower() for src, parts in source_parts.items()}
        self._loaded = True

    @property
    def indexed_url_count(self) -> int:
        self._load()
        return len(self._urls)

    @property
    def source_names(self) -> set[str]:
        self._load()
        return set(self._source_names)

    @staticmethod
    def _norm(url: str) -> str:
        return url.strip().rstrip("/")

    def _url_path_match(self, url: str) -> bool:
        """Flexible URL matching by path components.

        Matches if the last 2-3 path components of the URL exist in the corpus.
        Handles differences like raw.githubusercontent.com vs spark.apache.org.
        """
        self._load()
        norm_url = self._norm(url)
        if norm_url in self._urls:
            return True

        # Extract path components after domain
        try:
            from urllib.parse import urlparse

            parsed = urlparse(norm_url)
            path = parsed.path.strip("/")
            parts = path.split("/")

            # Try matching last 3, then last 2 path components
            for n in [3, 2]:
                if len(parts) >= n:
                    suffix = "/".join(parts[-n:])
                    for corpus_url in self._urls:
                        if suffix in corpus_url:
                            return True
        except Exception:
            pass
        return False

    def url_covered(self, url: str) -> bool:
        """Whether ``url`` corresponds to an indexed chunk."""
        return self._url_path_match(url)

    def term_present(self, term: str, *, source: str | None = None) -> bool:
        """Whether ``term`` occurs anywhere in the corpus (optionally a source)."""
        self._load()
        haystack = self._source_text.get(source, "") if source else self._corpus_text
        return term.lower() in haystack

    def terms_covered(
        self,
        terms: list[str],
        urls: list[str] | None = None,
        source: str | None = None,
    ) -> tuple[float, list[str]]:
        """Return ``(recall, missing_terms)`` against the url-scoped (or whole) corpus."""
        self._load()
        if urls:
            haystack = " ".join(
                text for url in urls for text in [self._source_text.get(self._url_source.get(self._norm(url), ""), "")]
            )
        elif source:
            haystack = self._source_text.get(source, "")
        else:
            haystack = self._corpus_text
        if not haystack:
            return 0.0, list(terms)
        missing = [t for t in terms if t.lower() not in haystack]
        recall = (len(terms) - len(missing)) / len(terms) if terms else 1.0
        return recall, missing

    def validate_row(self, row: dict) -> dict:
        """Coverage verdict for one row.

        Returns a dict with ``id``, ``status`` (``"pass"``/``"fail"``),
        ``out_of_scope``, ``missing_urls``, ``missing_terms`` and the
        url-scoped ``term_recall``.
        """
        self._load()
        row_id = row.get("id", "")
        oos = bool(row.get("out_of_scope", False))
        source = row.get("source_name") or ""
        expected_urls = [str(u) for u in row.get("expected_urls") or []]
        expected_terms = [str(t) for t in row.get("expected_terms") or []]

        if oos:
            return {
                "id": row_id,
                "status": "pass",
                "out_of_scope": True,
                "missing_urls": [],
                "missing_terms": [],
                "term_recall": 1.0,
            }

        missing_urls = [u for u in expected_urls if not self.url_covered(u)]
        # Terms must be found somewhere in the corpus (hard fail) and ideally
        # within the expected-url chunks (soft signal via term_recall).
        globally_missing = [t for t in expected_terms if not self.term_present(t, source=source)]
        term_recall, _ = self.terms_covered(expected_terms, expected_urls, source=source)
        status = "pass" if not missing_urls and not globally_missing else "fail"
        return {
            "id": row_id,
            "status": status,
            "out_of_scope": False,
            "missing_urls": missing_urls,
            "missing_terms": globally_missing,
            "term_recall": round(term_recall, 3),
        }

    def validate_rows(self, rows: list[dict]) -> list[dict]:
        return [self.validate_row(row) for row in rows]

    def report(self, rows: list[dict]) -> dict:
        """Aggregate a coverage report across rows."""
        verdicts = self.validate_rows(rows)
        by_source: Counter = Counter()
        by_status: Counter = Counter()
        for v in verdicts:
            by_status[v["status"]] += 1
            # source attribution from the original row (not present in verdict)
        # Re-run source attribution from rows to keep the report readable.
        for row, v in zip(rows, verdicts, strict=False):
            src = row.get("source_name") or ("out_of_scope" if v["out_of_scope"] else "unknown")
            by_source[src] += 1
        return {
            "rows": len(rows),
            "pass": by_status["pass"],
            "fail": by_status["fail"],
            "by_source": dict(by_source),
            "failures": [v for v in verdicts if v["status"] == "fail"],
        }
