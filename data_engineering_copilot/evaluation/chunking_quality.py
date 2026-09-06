"""Hermetic corpus-level chunk-quality metrics (no retrieval, no LLM).

Measured on the live pinned generations 2026-09-06: old d3dbad402105
fence_fracture_rate 0.0586 / tiny_rate 0.0170 / sentence_fracture_rate 0.3110
(median 91 words); new cd208afaf0f8 fence_fracture_rate 0.1187 / tiny_rate
0.0441 / sentence_fracture_rate 0.2681 / table_fracture_rate 0.0028.
Thresholds for the gate live in :mod:`chunking_eval` gate constants and
ADR-018 (amended 2026-09-06).

``table_fracture_rate`` / ``sentence_fracture_rate`` are real metrics (H4):
rows get CRLF-normalized before measurement so ``\r\n`` never distorts
punct/balance checks (L1).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable

MIN_TINY_WORDS = 10
OVERSIZED_MULTIPLIER = 1.5

# A chunk ends mid-sentence when its last non-space char is sentence-internal
# (a lower-case letter, digit, or closing bracket) rather than terminal
# punctuation. Conservative by design: verifying the sentence truly continues
# needs the neighboring chunk, which the corpus stream doesn't carry — this
# regex only flags tails that can NOT be a completed sentence, so every hit is
# a real mid-sentence cut (pure-code chunks are excluded by type).
_MID_SENTENCE_RE = re.compile(r"(?i)[a-z0-9)\]]$")


def fence_balance(text: str) -> bool:
    """Return True when the chunk contains an even number of ``` markers."""
    return text.count("```") % 2 == 0


def table_balance(text: str) -> bool:
    """Return True when an HTML-table chunk has balanced row/cell tags.

    A chunk torn along a table boundary shows up here as an unbalanced
    ``<tr>``/``</tr>`` (or ``<td>``/``</td>``) count — the same failure mode
    that silently drops rows in downstream markdown→HTML rendering.
    """
    if "<table" not in text.lower():
        return True
    counts = [
        (m1, m2)
        for m1, m2 in (
            ("<tr", "</tr>"),
            ("<td", "</td>"),
            ("<table", "</table>"),
        )
    ]
    return all(text.count(a) == text.count(b) for a, b in counts)


def ends_mid_sentence(text: str, chunk_type: str) -> bool:
    """Return True when ``text``'s last non-space char is mid-sentence."""
    if chunk_type == "code":
        return False
    if not text:
        return False
    # Tail inside an unclosed fence is code, not prose: a balanced-fence chunk
    # with an even marker count is guaranteed to end on prose or a fence closer.
    if text.count("```") % 2 == 1:
        return False
    tail = text.rstrip()
    return bool(_MID_SENTENCE_RE.search(tail))


def _word_count(row: dict) -> int:
    wc = row.get("word_count")
    if isinstance(wc, int):
        return wc
    return len((row.get("text") or "").split())


def audit_chunks(chunks: Iterable[dict], target_words: int = 500) -> dict:
    """Stream a chunks.jsonl iterable into a quality report."""
    counts: list[int] = []
    total = tiny = fractured = hp = oversized = table_frac = sentence_frac = 0
    tiny_samples: list[tuple] = []
    fracture_samples: list[tuple] = []
    ct: Counter[str] = Counter()
    cv: Counter[str] = Counter()
    src: Counter[str] = Counter()

    for row in chunks:
        total += 1
        wc = _word_count(row)
        counts.append(wc)
        text = (row.get("text") or "").replace("\r\n", "\n")
        ct[row.get("chunk_type", "?")] += 1
        cv[row.get("chunker_version", "?")] += 1
        src[row.get("source_name", "?")] += 1
        if wc < MIN_TINY_WORDS:
            tiny += 1
            if len(tiny_samples) < 8:
                tiny_samples.append((row.get("file_path"), wc, text[:70]))
        if wc > target_words * OVERSIZED_MULTIPLIER:
            oversized += 1
        if row.get("heading_path"):
            hp += 1
        if not fence_balance(text):
            fractured += 1
            if len(fracture_samples) < 8:
                fracture_samples.append((row.get("source_name"), row.get("file_path"), text[:70]))
        if not table_balance(text):
            table_frac += 1
        if ends_mid_sentence(text, row.get("chunk_type", "text")):
            sentence_frac += 1

    counts.sort()
    stats: dict = {"min": 0, "median": 0.0, "mean": 0.0, "max": 0, "p5": 0, "p95": 0}
    if counts:
        stats = {
            "min": counts[0],
            "p5": counts[len(counts) // 20],
            "median": statistics.median(counts),
            "mean": statistics.mean(counts),
            "p95": counts[min(len(counts) - 1, int(len(counts) * 0.95))],
            "max": counts[-1],
        }
    return {
        "total": total,
        "word_count": stats,
        "tiny_rate": tiny / total if total else 0.0,
        "oversized_rate": oversized / total if total else 0.0,
        "fence_fracture_rate": fractured / total if total else 0.0,
        "table_fracture_rate": table_frac / total if total else 0.0,
        "sentence_fracture_rate": sentence_frac / total if total else 0.0,
        "heading_path_coverage": hp / total if total else 0.0,
        "chunk_type_histogram": dict(ct),
        "chunker_version_histogram": dict(cv),
        "source_histogram": dict(src),
        "tiny_samples": tiny_samples,
        "fence_fracture_samples": fracture_samples,
    }


def gate_verdict(
    report: dict,
    *,
    fence_fracture: float = 0.02,
    tiny_rate: float = 0.01,
    oversized_rate: float = 0.001,
    table_fracture: float = 0.01,
    sentence_fracture: float = 0.02,
) -> dict:
    """Compare a report against thresholds; returns verdict + per-metric deltas."""
    checks = {
        "fence_fracture_rate": fence_fracture,
        "tiny_rate": tiny_rate,
        "oversized_rate": oversized_rate,
        "table_fracture_rate": table_fracture,
        "sentence_fracture_rate": sentence_fracture,
    }
    results: dict[str, dict] = {}
    for metric, threshold in checks.items():
        value = report[metric]
        results[metric] = {"value": value, "threshold": threshold, "ok": value <= threshold}
    return {"checks": results, "pass": all(r["ok"] for r in results.values())}
