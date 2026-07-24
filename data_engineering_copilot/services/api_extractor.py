"""API documentation structured extractor.

Detects API-style content in markdown (method signatures, parameter tables,
definition lists) and enriches chunk metadata with structured fields.
"""

from __future__ import annotations

import re
from dataclasses import replace

from data_engineering_copilot.domain.models import DocumentChunk

# Patterns that suggest an API/method signature
_SIG_PATTERNS = [
    # Python-style: def method(args) or class.method(args)
    re.compile(r"^\s*(?:def|async\s+def)\s+(\w+(?:\.\w+)*)\s*\(", re.MULTILINE),
    # PySpark / Java style: ClassName.methodName( at start of line
    re.compile(r"^\s*([A-Z]\w+(?:\.\w+)*)\s*\(", re.MULTILINE),
    # Module chain anywhere: spark.read.parquet( or pyspark.sql.functions.col(
    re.compile(r"\b(\w+\.\w+(?:\.\w+)*)\s*\("),
    # Backtick signatures: `spark.read.parquet(path)`
    re.compile(r"`(\w+(?:\.\w+)*)\s*\(`"),
]

# Parameter list patterns
_PARAM_PATTERNS = [
    # :param name: description
    re.compile(r"^:param\s+(\w+)\s*:", re.MULTILINE),
    # | Parameter | Type | Description | (markdown table row)
    re.compile(r"^\|\s*(\w+)\s*\|", re.MULTILINE),
]

# Return type hints
_RETURN_PATTERN = re.compile(r"->\s*([\w\[\], .|]+)")


class ApiDocExtractor:
    """Post-processor that detects API documentation and enriches chunks.

    Run this *after* the main chunker to tag API-relevant chunks with
    structured metadata (method name, module, parameters, return type).
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def extract(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Return *chunks* with API metadata injected where applicable."""
        if not self._enabled:
            return chunks
        return [self._enrich_one(c) for c in chunks]

    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_one(chunk: DocumentChunk) -> DocumentChunk:
        text = chunk.text
        # Skip already-tagged chunks
        if chunk.chunk_type == "api":
            return chunk

        # Detect method signature
        method_name = ""
        module = ""
        for pat in _SIG_PATTERNS:
            m = pat.search(text)
            if m:
                full = m.group(1)
                parts = full.rsplit(".", 1)
                if len(parts) == 2:
                    module, method_name = parts
                else:
                    method_name = full
                break

        if not method_name:
            return chunk

        # Detect parameters
        params: list[str] = []
        for pat in _PARAM_PATTERNS:
            params.extend(m.group(1) for m in pat.finditer(text))
            if params:
                break

        # Detect return type
        ret_match = _RETURN_PATTERN.search(text)
        return_type = ret_match.group(1).strip() if ret_match else ""

        # Build prefix
        parts = []
        if module:
            parts.append(f"Module: {module}")
        parts.append(f"Method: {method_name}")
        if params:
            parts.append(f"Params: {', '.join(dict.fromkeys(params))}")
        if return_type:
            parts.append(f"Returns: {return_type}")

        prefix = f"[API: {' | '.join(parts)}]\n\n"
        return replace(chunk, text=f"{prefix}{chunk.text}", chunk_type="api")
