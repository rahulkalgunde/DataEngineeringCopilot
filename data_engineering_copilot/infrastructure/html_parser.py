from __future__ import annotations

import logging

from bs4 import BeautifulSoup

from data_engineering_copilot.domain.models import ParsedDocument, RawDocument
from data_engineering_copilot.utils.text import normalize_whitespace

logger = logging.getLogger(__name__)


class DocumentationHtmlParser:
    def parse(self, raw: RawDocument) -> ParsedDocument | None:
        soup = BeautifulSoup(raw.html, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = self._title(soup, raw.url)
        content = soup.find("main") or soup.find("article") or soup.body or soup
        text = normalize_whitespace(content.get_text(" "))

        if len(text.split()) < 40:
            logger.info("HTML parser skipped short page source=%s url=%s title=%r", raw.source_name, raw.url, title)
            return None

        logger.info(
            "HTML parser extracted document source=%s url=%s title=%r words=%s",
            raw.source_name,
            raw.url,
            title,
            len(text.split()),
        )
        return ParsedDocument(
            source_name=raw.source_name,
            title=title,
            url=raw.url,
            text=text,
        )

    def _title(self, soup: BeautifulSoup, fallback: str) -> str:
        heading = soup.find("h1")
        if heading:
            return normalize_whitespace(heading.get_text(" "))
        if soup.title and soup.title.string:
            return normalize_whitespace(soup.title.string)
        return fallback
