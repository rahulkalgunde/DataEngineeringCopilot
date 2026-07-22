from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from data_engineering_copilot.domain.models import ParsedDocument, RawDocument
from data_engineering_copilot.utils.text import normalize_whitespace


def html_to_markdown(html: str, min_words: int = 40) -> str | None:
    """Convert documentation HTML to clean Markdown for RAG ingestion.

    Returns None if the result has fewer than *min_words* words.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    content = soup.find("main") or soup.find("article") or soup.find("body") or soup

    markdown_text = md(
        str(content),
        heading_style="ATX",
        strip=["img", "script", "style", "noscript", "nav", "footer", "header", "aside"],
    )

    markdown_text = _clean_markdown(markdown_text)

    word_count = len(markdown_text.split())
    if word_count < min_words:
        return None

    return markdown_text


def _clean_markdown(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class MarkdownParser:
    """Parser that converts HTML to Markdown via ``html_to_markdown()``.

    Implements ``ParserProtocol`` — returns a ``ParsedDocument`` with
    Markdown-formatted text instead of plain text.
    """

    def parse(self, raw: RawDocument) -> ParsedDocument | None:
        soup = BeautifulSoup(raw.html, "html.parser")
        title = self._title(soup, raw.url)
        markdown_text = html_to_markdown(raw.html)
        if markdown_text is None:
            return None
        return ParsedDocument(
            source_name=raw.source_name,
            title=title,
            url=raw.url,
            text=markdown_text,
        )

    @staticmethod
    def _title(soup: BeautifulSoup, fallback: str) -> str:
        heading = soup.find("h1")
        if heading:
            return normalize_whitespace(heading.get_text(" "))
        if soup.title and soup.title.string:
            return normalize_whitespace(soup.title.string)
        return fallback
