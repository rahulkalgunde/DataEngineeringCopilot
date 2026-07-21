from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as md


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
