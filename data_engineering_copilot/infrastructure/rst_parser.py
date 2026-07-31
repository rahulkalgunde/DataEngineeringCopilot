from __future__ import annotations

from bs4 import BeautifulSoup
from docutils.core import publish_parts

from data_engineering_copilot.domain.models import ParsedDocument, RawDocument
from data_engineering_copilot.infrastructure.html_to_markdown import html_to_markdown
from data_engineering_copilot.utils.text import normalize_whitespace

_RST_URL_SUFFIXES = (".rst", ".rst.txt")


class RstParser:
    def parse(self, raw: RawDocument) -> ParsedDocument | None:
        url_lower = raw.url.lower()
        if not any(url_lower.endswith(p) for p in _RST_URL_SUFFIXES):
            raise ValueError(f"RstParser does not support: {raw.url}")

        html = self._rst_to_html(raw.html)
        if html is None:
            return None

        soup = BeautifulSoup(html, "html.parser")

        title = self._title(soup, raw.url)

        markdown_text = html_to_markdown(html)
        if markdown_text is None:
            return None

        return ParsedDocument(
            source_name=raw.source_name,
            title=title,
            url=raw.url,
            text=markdown_text,
        )

    def _rst_to_html(self, rst_text: str) -> str | None:
        try:
            parts = publish_parts(
                source=rst_text,
                writer_name="html5",
                settings_overrides={
                    "report_level": 2,
                    "halt_level": 4,
                    "warning_stream": None,
                },
            )
            body = parts.get("html_body", "")
            if not body or not body.strip():
                return None
            return body
        except Exception:
            return None

    @staticmethod
    def _title(soup: BeautifulSoup, fallback: str) -> str:
        heading = soup.find("h1")
        if heading:
            return normalize_whitespace(heading.get_text(" "))
        if soup.title and soup.title.string:
            return normalize_whitespace(soup.title.string)
        return fallback
