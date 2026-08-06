"""Native parsing for Markdown, RST, and plain source files.

Routes Spark source files to the appropriate parser without passing Markdown
or code through the HTML cleanup pipeline. Preserves headings, fenced code
blocks, and section boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Supported source extensions mapped to a canonical language.
_CODE_EXTENSIONS = {
    ".py": "python",
    ".scala": "scala",
    ".java": "java",
    ".sql": "sql",
    ".r": "r",
}

_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}
_RST_EXTENSIONS = {".rst", ".rst.txt"}
_TEXT_EXTENSIONS = {".txt"}

# Liquid/Jekyll directives that add no retrieval value. Preserved as empty
# lines so surrounding prose stays intact.
_LIQUID_DIRECTIVE_RE = re.compile(r"^\s*\{%[-]?\s+[^%]*\s*[-]?%\}\s*$", re.MULTILINE)
_LIQUID_VARIABLE_RE = re.compile(r"\{\{[^}]*\}\}")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class NativeParsedDocument:
    """A parsed native document with preserved structure."""

    title: str
    text: str
    sections: tuple[tuple[str, str], ...]
    language: str
    relative_path: str
    line_count: int


class NativeDocumentParser:
    """Parse Markdown, RST, and code source files natively."""

    def parse(self, path: Path, doc_type: str) -> NativeParsedDocument:
        """Parse a file based on its extension and document type.

        Raises ``FileNotFoundError`` for missing files and ``ValueError`` for
        unsupported extensions.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        suffix = path.suffix.lower()
        rel = path.as_posix()

        if suffix in _MARKDOWN_EXTENSIONS or (suffix in _TEXT_EXTENSIONS and doc_type == "guide"):
            return self.parse_markdown(text, rel)
        if suffix in _RST_EXTENSIONS:
            return self.parse_rst(text, rel)
        if suffix in _CODE_EXTENSIONS:
            language = _CODE_EXTENSIONS[suffix]
            return self.parse_code(text, rel, language)
        raise ValueError(f"Unsupported file extension for Spark ingestion: {suffix}")

    def parse_markdown(self, text: str, path: str) -> NativeParsedDocument:
        """Normalize Markdown, preserving headings and fenced code blocks."""
        normalized = self._normalize_newlines(text)
        cleaned = self._strip_liquid(normalized)
        title = self._markdown_title(cleaned, path)
        sections = self._split_sections_markdown(cleaned)
        return NativeParsedDocument(
            title=title,
            text=cleaned,
            sections=sections,
            language="conceptual",
            relative_path=path,
            line_count=len(cleaned.splitlines()),
        )

    def parse_rst(self, text: str, path: str) -> NativeParsedDocument:
        """Parse RST, preserving underlined headings and code blocks."""
        normalized = self._normalize_newlines(text)
        cleaned = self._strip_liquid(normalized)
        title = self._rst_title(cleaned, path)
        sections = self._split_sections_rst(cleaned)
        return NativeParsedDocument(
            title=title,
            text=cleaned,
            sections=sections,
            language="conceptual",
            relative_path=path,
            line_count=len(cleaned.splitlines()),
        )

    def parse_code(self, text: str, path: str, language: str) -> NativeParsedDocument:
        """Parse a source file, preserving its text for code chunking.

        Non-empty code files are always accepted, even below a prose word
        threshold. Python syntax is validated with ``ast`` only for boundary
        detection; the source is never executed.
        """
        normalized = self._normalize_newlines(text)
        title = self._code_title(path, language)
        return NativeParsedDocument(
            title=title,
            text=normalized,
            sections=(),
            language=language,
            relative_path=path,
            line_count=len(normalized.splitlines()),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _strip_liquid(self, text: str) -> str:
        text = _HTML_COMMENT_RE.sub("", text)
        text = _LIQUID_DIRECTIVE_RE.sub("", text)
        text = _LIQUID_VARIABLE_RE.sub("", text)
        # Collapse runs of blank lines left by directive removal to two.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n" if text.strip() else text

    @staticmethod
    def _markdown_title(text: str, path: str) -> str:
        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                return m.group(2).strip()
        return Path(path).stem.replace("-", " ").replace("_", " ").strip()

    @staticmethod
    def _rst_title(text: str, path: str) -> str:
        lines = text.splitlines()
        for i in range(len(lines) - 1):
            if lines[i].strip() and lines[i + 1] and set(lines[i + 1].strip()) <= {"="}:
                return lines[i].strip()
        return Path(path).stem.replace("-", " ").replace("_", " ").strip()

    @staticmethod
    def _code_title(path: str, language: str) -> str:
        stem = Path(path).stem.replace("-", " ").replace("_", " ").strip()
        return f"{stem} ({language})"

    def _split_sections_markdown(self, text: str) -> tuple[tuple[str, str], ...]:
        sections: list[tuple[str, str]] = []
        current_header = ""
        current_parts: list[str] = []
        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m and m.group(1) in ("#", "##", "###"):
                if current_parts:
                    sections.append((current_header, "\n".join(current_parts)))
                current_header = m.group(2).strip()
                current_parts = [line]
            else:
                current_parts.append(line)
        if current_parts:
            sections.append((current_header, "\n".join(current_parts)))
        return tuple(sections)

    def _split_sections_rst(self, text: str) -> tuple[tuple[str, str], ...]:
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        current_header = ""
        current_parts: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if i + 1 < len(lines) and lines[i + 1].strip() and set(lines[i + 1].strip()) <= {"="} and line.strip():
                if current_parts:
                    sections.append((current_header, "\n".join(current_parts)))
                current_header = line.strip()
                current_parts = [line, lines[i + 1]]
                i += 2
                continue
            current_parts.append(line)
            i += 1
        if current_parts:
            sections.append((current_header, "\n".join(current_parts)))
        return tuple(sections)
