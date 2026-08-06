"""Spark-specific HTML main-content extraction.

Rendered Spark documentation comes from two very different generators:

- Jekyll guide pages (``docs/*.html``) that place content in ``div#content``
  and navigation in ``div.left-menu-wrapper`` (not a ``<nav>`` tag).
- PySpark Sphinx pages (``reference/**/*.html``) that place content in
  ``<main>``/``<article>`` with breadcrumb and page-TOC ``<nav>`` elements.

``html_to_markdown`` (the generic crawler parser) cannot be used directly: its
``<body>`` fallback would drag the Jekyll sidebar into every page. This parser
selects ``<main>`` first, then ``<article>``, then the configured content root,
and removes navigation containers before converting with ``markdownify``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from data_engineering_copilot.utils.text import normalize_whitespace

# Minimum normalized words for a rendered page to be considered content.
_MIN_CONTENT_WORDS = 10

# Tags always stripped from any candidate content root.
_STRIP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside")

# Headings that add no retrieval value and are dropped after conversion.
_SIG_HEADING_RE = re.compile(r"^\s*#(?:#)*\s*\[\s*source\s*\]")

# ``[source]`` links point into Sphinx ``_modules`` source listings which we
# deliberately do not ingest; the raw ``[source]`` text adds no value.
_SOURCE_LINK_RE = re.compile(r"\[\s*source\s*\]")


@dataclass(frozen=True)
class RenderedParseResult:
    """Extracted main content from a single rendered HTML page."""

    title: str
    text: str
    word_count: int
    content_hash: str
    canonical_url: str
    source_path: str
    headings: tuple[tuple[str, int], ...]  # (heading text, level)


class SparkHtmlParser:
    """Extract main content from a rendered Spark HTML page."""

    def __init__(
        self,
        content_root_selector: str = "",
        excluded_selectors: tuple[str, ...] = (),
        min_words: int = _MIN_CONTENT_WORDS,
    ) -> None:
        self._content_root_selector = content_root_selector
        self._excluded_selectors = excluded_selectors
        self._min_words = min_words

    def parse(
        self,
        html: str,
        canonical_url: str,
        source_path: str,
    ) -> RenderedParseResult | None:
        """Parse *html* and return extracted content, or ``None`` when too short.

        ``None`` means the page is navigation-only or otherwise too short to be
        retrieval content — never a valid empty document.
        """
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        content = self._select_content(soup)
        if content is None:
            return None

        for selector in self._excluded_selectors:
            for tag in content.select(selector):
                tag.decompose()

        title = self._title(soup, content, canonical_url)
        markdown_text = md(
            str(content),
            heading_style="ATX",
            strip=["img", "script", "style", "noscript", "nav", "header", "footer", "aside"],
        )
        markdown_text = self._clean_markdown(markdown_text)

        word_count = len(markdown_text.split())
        if word_count < self._min_words:
            return None

        content_hash = hashlib.sha256(markdown_text.encode("utf-8")).hexdigest()
        headings = self._extract_headings(markdown_text)

        return RenderedParseResult(
            title=title,
            text=markdown_text,
            word_count=word_count,
            content_hash=content_hash,
            canonical_url=canonical_url,
            source_path=source_path,
            headings=headings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_content(self, soup: BeautifulSoup):
        main = soup.find("main")
        if main is not None:
            return main
        article = soup.find("article")
        if article is not None:
            return article
        if self._content_root_selector:
            roots = soup.select(self._content_root_selector)
            if roots:
                return roots[0]
        return None

    @staticmethod
    def _title(soup: BeautifulSoup, content, fallback: str) -> str:
        heading = soup.find("h1")
        if heading is not None:
            return normalize_whitespace(heading.get_text(" "))
        if soup.title and soup.title.string:
            title = normalize_whitespace(soup.title.string)
            for marker in (" - Spark ", " — Spark "):
                if marker in title:
                    title = title.split(marker, 1)[0]
            return title
        stem = Path(fallback).stem.replace("-", " ").replace("_", " ").strip()
        return stem or fallback

    @staticmethod
    def _clean_markdown(text: str) -> str:
        text = _SOURCE_LINK_RE.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        # Remove markdownify artifact lines for pure anchor links.
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("[#]") and not _SIG_HEADING_RE.match(line)
        )
        return text.strip()

    @staticmethod
    def _extract_headings(text: str) -> tuple[tuple[str, int], ...]:
        headings: list[tuple[str, int]] = []
        for line in text.splitlines():
            m = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if m:
                headings.append((m.group(2).strip(), len(m.group(1))))
        return tuple(headings)
