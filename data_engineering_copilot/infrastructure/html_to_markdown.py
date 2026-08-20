from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md

from data_engineering_copilot.domain.models import ParsedDocument, RawDocument
from data_engineering_copilot.utils.text import normalize_whitespace

# Placeholder tokens injected before markdownify so table content survives the
# HTML->Markdown pass; replaced with pipe-table Markdown afterwards. Kept free
# of underscores/hyphens-adjacent emphasis so markdownify leaves them verbatim.
_TABLE_PLACEHOLDER_PREFIX = "MARKDOWNTABLEPLACEHOLDER"
_TABLE_PLACEHOLDER_SUFFIX = "END"


def html_to_markdown(html: str, min_words: int = 40) -> str | None:
    """Convert documentation HTML to clean Markdown for RAG ingestion.

    Returns None if the result has fewer than *min_words* words.

    Tables are preserved as GitHub-style pipe-tables (``markdownify`` drops
    them), and sidebar/navigation stripping is handled by callers that pass
    pre-cleaned HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Preserve <table> elements as pipe-tables. markdownify drops tables, so we
    # serialize each one to Markdown up-front, swap in a placeholder that
    # survives the markdownify pass, then restore the Markdown afterwards.
    table_restorations: list[tuple[str, str]] = []
    for index, table in enumerate(soup.find_all("table")):
        placeholder = f"{_TABLE_PLACEHOLDER_PREFIX}{index}{_TABLE_PLACEHOLDER_SUFFIX}"
        table.replace_with(soup.new_string(placeholder))
        table_restorations.append((placeholder, _table_to_markdown(table)))

    content = soup.find("main") or soup.find("article") or soup.find("body") or soup

    markdown_text = md(
        str(content),
        heading_style="ATX",
        strip=["img", "script", "style", "noscript", "nav", "footer", "header", "aside"],
        code_language_callback=_code_language,
    )

    markdown_text = _clean_markdown(markdown_text)

    for placeholder, table_md in table_restorations:
        markdown_text = markdown_text.replace(placeholder, table_md)

    word_count = len(markdown_text.split())
    if word_count < min_words:
        return None

    return markdown_text


def _table_to_markdown(table: Tag) -> str:
    """Render an HTML ``<table>`` as a GitHub-style pipe-table."""
    rows: list[list[str]] = []
    header_cells: list[str] = []

    header = table.find("thead")
    if header is not None:
        header_cells = [_cell_text(th) for th in header.find_all(["th", "td"])]

    body = table.find("tbody") or table
    data_rows = body.find_all("tr")
    for tr in data_rows:
        cells = tr.find_all(["td", "th"])
        # Skip the header row if we already captured a <thead>.
        if header is not None and cells == header.find_all(["td", "th"]):
            continue
        rows.append([_cell_text(c) for c in cells])

    if not header_cells and rows:
        header_cells = rows[0]
        rows = rows[1:]

    if not header_cells:
        return ""

    def _fmt(cells: list[str]) -> str:
        return "| " + " | ".join(_escape_pipe(c) for c in cells) + " |"

    lines = [_fmt(header_cells), _fmt(["---"] * len(header_cells))]
    lines.extend(_fmt(r) for r in rows)
    return "\n".join(lines)


def _cell_text(cell: Tag) -> str:
    return normalize_whitespace(cell.get_text(" ")).strip()


def _escape_pipe(text: str) -> str:
    return text.replace("|", "\\|")


def _code_language(el: Tag) -> str:
    """Return the ``language-*`` class of a ``<pre><code>`` block.

    ``markdownify`` otherwise emits an unlabeled fence, which drops the syntax
    hint the HTML carries. Unknown or absent languages fall back to empty.
    """
    code = el.find("code")
    if code is not None:
        classes = code.get("class") or []
        for cls in classes:
            if cls.startswith("language-"):
                return cls[len("language-") :]
    return ""


_FENCE_RE = re.compile(r"(?ms)^(`{3,}|~{3,}).*?^\1\s*$")


def _clean_markdown(text: str) -> str:
    # Collapse runs of blank lines and of spaces/tabs, but never inside fenced
    # code blocks: indentation and blank lines there are semantically loaded.
    parts: list[str] = []
    cursor = 0
    for match in _FENCE_RE.finditer(text):
        parts.append(_collapse(text[cursor : match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(_collapse(text[cursor:]))
    return "".join(parts).strip()


def _collapse(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


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
