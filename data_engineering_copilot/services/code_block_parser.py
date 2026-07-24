"""Lightweight code block parser for markdown documentation.

Extracts fenced code blocks, detects function/class boundaries via regex
or Python AST, and enriches chunks with scope metadata.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import replace

from data_engineering_copilot.domain.models import DocumentChunk

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)

# Python-aware boundary detection (lightweight, no AST dependency)
_PY_DEF_RE = re.compile(
    r"^(?:[ \t]*(?:async\s+)?def\s+\w+|[ \t]*class\s+\w+)",
    re.MULTILINE,
)


class CodeBlockParser:
    """Post-processor that identifies code-only chunks and enriches them.

    After the main chunker produces chunks, run this to:
    1. Detect chunks that consist entirely of code blocks
    2. Split large code blocks at function/class boundaries
    3. Prepend scope metadata for better retrieval
    """

    def __init__(self, enabled: bool = True, max_code_lines: int = 500) -> None:
        self._enabled = enabled
        self._max_code_lines = max_code_lines

    def extract(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Return *chunks* with code metadata injected where applicable."""
        if not self._enabled:
            return chunks
        result: list[DocumentChunk] = []
        for chunk in chunks:
            result.extend(self._process_one(chunk))
        return result

    # ------------------------------------------------------------------

    def _process_one(self, chunk: DocumentChunk) -> list[DocumentChunk]:
        if chunk.chunk_type == "code":
            return [chunk]

        # Extract fenced code blocks from the chunk text
        fences = list(_FENCE_RE.finditer(chunk.text))
        if not fences:
            return [chunk]

        # Compute what fraction of the text is code
        total_chars = len(chunk.text)
        code_chars = sum(m.end() - m.start() for m in fences)

        # If more than 60% of the chunk is code, treat as code chunk
        if code_chars / max(total_chars, 1) < 0.6:
            return [chunk]

        # Scope metadata
        scope = self._scope_prefix(chunk)

        # Check if code is small enough to keep as-is
        code_text = "\n\n".join(m.group(0) for m in fences)
        if len(code_text.splitlines()) <= self._max_code_lines:
            enriched = f"{scope}{chunk.text}"
            return [
                replace(
                    chunk,
                    text=enriched,
                    chunk_type="code",
                    word_count=len(enriched.split()),
                )
            ]

        # Large code block: split at function/class boundaries
        result: list[DocumentChunk] = []
        for i, fence_match in enumerate(fences):
            lang = fence_match.group(1) or "code"
            body = fence_match.group(2)
            parts = self._split_at_boundaries(body)
            for j, part in enumerate(parts):
                part_lines = part.strip().splitlines()
                if not part_lines:
                    continue
                enriched = f"{scope}```{lang}\n{part.strip()}\n```"
                result.append(
                    replace(
                        chunk,
                        text=enriched,
                        chunk_type="code",
                        chunk_id=f"{chunk.chunk_id}:code:{i}:{j}",
                        word_count=len(enriched.split()),
                    )
                )

        return result if result else [chunk]

    @staticmethod
    def _scope_prefix(chunk: DocumentChunk) -> str:
        parts = [f"# Source: {chunk.source_name}"]
        if chunk.title:
            parts.append(f"# Document: {chunk.title}")
        if chunk.section_header:
            parts.append(f"# Section: {chunk.section_header}")
        parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _split_at_boundaries(code: str) -> list[str]:
        """Split Python code at def/class boundaries using AST when possible."""
        # Try AST parsing first (more accurate)
        ast_parts = CodeBlockParser._split_with_ast(code)
        if ast_parts:
            return ast_parts

        # Fallback to regex-based splitting
        lines = code.splitlines(keepends=True)
        boundaries = []
        for i, line in enumerate(lines):
            if _PY_DEF_RE.match(line):
                boundaries.append(i)

        if not boundaries:
            return [code]

        parts: list[str] = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
            part = "".join(lines[start:end]).strip()
            if part:
                parts.append(part)

        return parts

    @staticmethod
    def _split_with_ast(code: str) -> list[str]:
        """Split Python code at top-level function/class definitions using AST.

        Returns a list of code chunks, each containing a single top-level
        definition with its docstring and body.  Returns empty list on
        parse failure (caller should fall back to regex).
        """
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return []

        lines = code.splitlines(keepends=True)
        parts: list[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1  # 0-indexed
                # Find end: use next sibling's start or end of file
                end = len(lines)
                for sibling in ast.iter_child_nodes(tree):
                    if sibling is node:
                        continue
                    if hasattr(sibling, "lineno") and sibling.lineno > node.lineno:
                        end = min(end, sibling.lineno - 1)
                        break
                part = "".join(lines[start:end]).strip()
                if part:
                    parts.append(part)

        return parts
