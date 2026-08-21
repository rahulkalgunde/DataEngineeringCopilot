"""Schema and validation for chunking-quality gold datasets.

A gold dataset is a JSONL file where each line is a :class:`ChunkingGoldDoc`:
a source document together with the spans a "correct" chunker should recover.
Spans are character offsets relative to ``text``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingGoldSpan:
    """A single gold span within a document."""

    content: str
    start: int
    end: int
    structural_type: str


@dataclass(frozen=True)
class ChunkingGoldDoc:
    """A document annotated with gold chunk spans."""

    doc_id: str
    text: str
    gold_spans: list[ChunkingGoldSpan]


def validate_gold_doc(doc: ChunkingGoldDoc) -> None:
    """Raise ``ValueError`` when *doc* violates the gold schema.

    Checks that every span is in bounds and that ``doc.text[start:end]`` exactly
    matches the recorded span content.
    """
    for span in doc.gold_spans:
        if not (0 <= span.start <= span.end <= len(doc.text)):
            raise ValueError(
                f"Span {span.start}:{span.end} out of bounds for doc {doc.doc_id!r} (len={len(doc.text)})"
            )
        if doc.text[span.start : span.end] != span.content:
            raise ValueError(
                f"Span content mismatch in {doc.doc_id!r} at {span.start}:{span.end}: "
                f"expected {span.content!r}, got {doc.text[span.start:span.end]!r}"
            )
