"""RST → Markdown extraction via docutils (Task 6.1, H8).

The chunker operates on Markdown, but the Spark/Delta guide corpus is RST.
A regex that maps underline headings to ``#`` is blind to the real RST
grammar: overline+underline styles, ``.. code-block::`` / ``.. literalinclude::``
directives, literal blocks, and definition lists. This module parses with the
real RST parser (``docutils.core.publish_doctree``) and walks the doctree so
section nesting resolves to proper heading levels, code becomes fenced blocks,
tables flatten to pipe rows, and includes are inlined. Hermetic, offline, 0 LLM.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_INCLUDE_RE = re.compile(
    r"^\.\. (literalinclude|include)::\s+(\S+?)(?:\s+(.*?))?\s*$\n(?:[ \t]+:.*$\n?)*",
    re.MULTILINE,
)
_FENCE_LANG_RE = re.compile(r":language:\s*(\S+)")
_MAX_INCLUDE_DEPTH = 3

# Leading Apache Software Foundation site-license preamble. Stripped BEFORE
# parsing: as an RST comment at column 0 (``.. Licensed…``) docutils already
# drops it, but the Airflow/Docs-style variant indents the ``..`` block, which
# doctree models as a *block quote ``> ``* that would ride into every chunk.
_LICENSE_HINT = "Licensed to the Apache Software Foundation"
_LICENSE_TAIL = "under the License."


def _strip_site_license_preamble(text: str) -> str:
    """Drop a leading ASF license preamble, comment- or quote-encoded.

    Only removes a block that sits before the real content (every non-blank
    line is a ``..`` comment marker or indented/quote body) and that contains
    the Apache boilerplate start and end; anything else is left untouched.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _LICENSE_HINT in ln), None)
    if start is None:
        return text
    end = next((i for i in range(start, min(start + 60, len(lines))) if _LICENSE_TAIL in lines[i]), None)
    if end is None:
        return text
    header = lines[: end + 1]
    if any(ln.strip() and not ln.startswith("..") and not ln[0].isspace() for ln in header):
        return text
    out = lines[end + 1 :]
    while out and not out[0].strip():
        out.pop(0)
    return "\n".join(out)


# JSX/MDX wrappers used by the Claude / Delta mirror sources: open/close and
# self-closing tags (props included) are stripped; inner Markdown is kept.
_JSX_TAGS = (
    "Tabs",
    "Tab",
    "TabList",
    "TabListContainer",
    "Accordion",
    "AccordionItem",
    "Code",
    "Step",
    "Steps",
    "Video",
    "Tip",
    "Hint",
    "Hidden",
    "Banner",
    "Callout",
    "Note",
    "Warning",
    "Caution",
    "Details",
    "Summary",
    "WideContent",
    "Aside",
)
_JSX_TAG_RE = re.compile(
    r"</(?:" + "|".join(_JSX_TAGS) + r")(?:\s+[^>]*)?>" r"|<(?:" + "|".join(_JSX_TAGS) + r")(?:\s+[^>]*)?/?>"
)


def _expand_includes(text: str, source_path: str | None, depth: int = 0) -> str:
    """Inline ``.. include::`` / ``.. literalinclude::`` directives.

    Reference pages sprinkled with includes otherwise produce chunk bodies full
    of directive boilerplate instead of the actual content.
    """
    if depth >= _MAX_INCLUDE_DEPTH:
        return text
    base = Path(source_path).parent if source_path else Path.cwd()

    def _expand(m: re.Match) -> str:
        kind, target = m.group(1), m.group(2)
        opts = m.group(0)
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = base / candidate
        if not candidate.exists():
            logger.warning("rst include unresolved: %s via %s", target, source_path)
            return f"<!-- literalinclude: {target} not resolvable -->"
        body = candidate.read_text(encoding="utf-8", errors="replace")
        if kind == "literalinclude":
            lang = _FENCE_LANG_RE.search(opts)
            fence = lang.group(1) if lang else "text"
            indented = "\n".join("   " + line for line in body.splitlines())
            return f".. code-block:: {fence}\n\n{indented}\n"
        return _expand_includes(body, str(candidate), depth + 1)

    return _INCLUDE_RE.sub(_expand, text)


class _MarkdownWriter:
    """Walk a docutils doctree and emit Markdown (block-level fidelity only)."""

    def __init__(self) -> None:
        self.out: list[str] = []
        self._doc_title_emitted = False

    def _blank(self) -> None:
        if self.out and self.out[-1] != "":
            self.out.append("")

    def write_children(self, node) -> None:
        for child in node.children:
            tag = child.tagname
            if tag in ("system_message", "comment", "target", "substitution_definition", "header", "footer"):
                continue
            self._blank()
            if tag == "title":
                # docutils promotes a lone top-level heading to the document
                # title; surface it as the document's ``#`` level and sink its
                # former children sections one level below.
                self.out.append("# " + child.astext().strip())
                self._doc_title_emitted = True
                continue
            if tag == "section":
                self.write_section(child, 2 if self._doc_title_emitted else 1)
                continue
            self.write_block(child, 0)

    def write_section(self, section, depth: int) -> None:
        titles = [c for c in section.children if c.tagname == "title"]
        if titles:
            self.out.append("#" * depth + " " + titles[0].astext().strip())
        for child in section.children:
            tag = child.tagname
            if tag in ("title", "system_message", "comment"):
                continue
            self._blank()
            if tag == "section":
                self.write_section(child, depth + 1)
                continue
            self.write_block(child, depth)

    def write_block(self, node, depth: int) -> None:
        tag = node.tagname
        if tag == "paragraph":
            text = " ".join(node.astext().split())
            if text:
                self.out.append(text)
        elif tag in ("literal_block", "doctest_block"):
            lang = next((c for c in node.get("classes", ()) if c != "code"), "")
            body = node.astext().strip("\n")
            self.out.append(f"```{lang}")
            self.out.append(body)
            self.out.append("```")
        elif tag == "bullet_list":
            for item in node.children:
                self.out.append("- " + item.astext().strip())
        elif tag == "enumerated_list":
            for i, item in enumerate(node.children, 1):
                self.out.append(f"{i}. " + item.astext().strip())
        elif tag in ("definition_list", "field_list"):
            for child in node.children:
                if child.tagname not in ("definition_list_item", "field"):
                    continue
                names = [c.astext() for c in child.children if c.tagname in ("term", "field_name")]
                body = [c.astext() for c in child.children if c.tagname in ("definition", "field_body")]
                label = ", ".join(names)
                value = " ".join(body)
                self.out.append(f"**{label}**: {value}")
        elif tag == "table":
            self._write_table(node)
        elif tag == "block_quote":
            for line in node.astext().splitlines():
                if line:
                    self.out.append("> " + line)
        elif tag == "line_block":
            for lb in node.get("children", ()):
                if lb.tagname == "line":
                    self.out.append(lb.astext())
        elif tag == "image":
            self.out.append(f"![image]({node.get('uri', '')})")
        elif tag in ("raw", "math_block"):
            text = node.astext()
            if text.strip():
                self.out.append(text)
        else:
            text = node.astext()
            if text.strip():
                self.out.append(text)

    def _write_table(self, table) -> None:
        rows: list[list[str]] = []
        for row in table.findall(lambda n: n.tagname == "row"):
            rows.append([entry.astext().replace("\n", " ").strip() for entry in row.children])
        if not rows:
            return
        widths: list[int] = []
        for row in rows:
            for i, cell in enumerate(row):
                if i >= len(widths):
                    widths.append(0)
                widths[i] = max(widths[i], len(cell))
        for i, row in enumerate(rows):
            padded = list(row) + [""] * (len(widths) - len(row))
            self.out.append("| " + " | ".join(cell.ljust(w) for cell, w in zip(padded, widths, strict=True)) + " |")
            if i == 0:
                self.out.append("| " + " | ".join("-" * w for w in widths) + " |")


def rst_to_markdown(text: str, source_path: str | None = None) -> str:
    """Convert RST *text* into Markdown headings/code/tables (no directives).

    Reads include/literalinclude directives relative to ``source_path``'s
    directory when resolvable. Output keeps the document's order and all real
    content; system messages, comments, and targets are dropped.
    """
    from docutils.core import publish_doctree

    text = _strip_site_license_preamble(text)
    text = _expand_includes(text, source_path)
    try:
        doctree = publish_doctree(text, settings_overrides={"report_level": 5, "halt_level": 6})
    except Exception:  # catastrophic input only; degrade to the regex path
        logger.warning("docutils parse failed; falling back to underline headings")
        return _rst_underline_headings(text)
    writer = _MarkdownWriter()
    writer.write_children(doctree)
    return "\n".join(writer.out)


def _rst_underline_headings(text: str) -> str:
    """Fallback: convert ``====`` underlined headings into ``#`` headings."""
    lines = text.splitlines()
    out: list[str] = []
    level_by_char = {"=": 1, "-": 2, "~": 3, "^": 4, '"': 5}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and i + 1 < len(lines):
            underline = lines[i + 1].strip()
            chars = set(underline)
            if len(chars) == 1 and len(underline) >= len(line):
                level = level_by_char.get(next(iter(chars)), 1)
                out.append("#" * level + " " + line)
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def strip_jsx_wrappers(markdown: str) -> str:
    """Strip JSX/MDX open/close/self-closing wrapper tags from *markdown*.

    Applies to the Claude/Delta mirror normalization BEFORE chunking; inner
    Markdown content is preserved. Props (e.g. ``<Tabs titles=…>``) are removed
    along with their tags so chunk text carries no JSX noise.
    """
    return _JSX_TAG_RE.sub("", markdown)
