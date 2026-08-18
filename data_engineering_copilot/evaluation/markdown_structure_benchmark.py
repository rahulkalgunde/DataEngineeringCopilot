"""Offline structural benchmark for the HTML-to-Markdown converter and parser.

Locks the structure-preservation behavior of ``html_to_markdown`` and
``NativeDocumentParser`` so future parser swaps are justified by data, not
hype. The benchmark runs on golden HTML fixtures plus a deterministic sample
of representative corpus Markdown files and records, per file:

* normalized reconstruction (converted/parsed output vs. expected source)
* table and fence defects (unclosed fences, malformed table separators,
  stray raw HTML) via a dependency-free block scanner
* section count, runtime, and peak Python allocation memory

``MarkdownBenchmarkReport.passes()`` is the fixed gate: every fixture must
reconstruct exactly with zero defects and the report must be schema-valid.
This module never changes production routing.
"""

from __future__ import annotations

import argparse
import re
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_engineering_copilot.infrastructure.html_to_markdown import html_to_markdown
from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
from data_engineering_copilot.utils.text import normalize_whitespace

# Fixed thresholds (plan gate).
MAX_RUNTIME_MS_PER_FILE = 5000.0
MAX_PEAK_MEMORY_KB = 1_000_000

# Dependency-free Markdown block scanner patterns.
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[^`]*$")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,}) *$")
_ATX_HEADING_RE = re.compile(r"^#{1,6} ")
_TABLE_SEPARATOR_RE = re.compile(r"^ *\|? *:?-{2,}:? *(?:\| *:?-{2,}:? *)*\|? *$")
_LOOKS_LIKE_SEPARATOR_RE = re.compile(r"^ *[\|: -]+ *$")
_HTML_TAG_RE = re.compile(r"<[a-zA-Z][a-zA-Z0-9-]*(?:\s[^>]*)?>")


@dataclass(frozen=True)
class MarkdownScan:
    """Structural metrics for one Markdown text."""

    section_count: int
    table_count: int
    fence_count: int
    defects: tuple[str, ...]


def scan_markdown(text: str) -> MarkdownScan:
    """Scan Markdown text without parsing dependencies.

    Tracks ATX headings, GFM tables (header row followed by a separator row),
    fenced code blocks, and defects: unclosed fences, malformed table
    separators, and stray raw HTML tags outside fences. Defects are unique and
    reported in deterministic order.
    """
    defects: set[str] = set()
    in_fence = False
    section_count = 0
    table_count = 0
    fence_count = 0
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if in_fence:
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
                fence_count += 1
            i += 1
            continue
        if _FENCE_OPEN_RE.match(line):
            in_fence = True
            i += 1
            continue
        if _ATX_HEADING_RE.match(line):
            section_count += 1
        if "|" in line:
            if i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
                table_count += 1
                i += 1
            elif (
                i + 1 < len(lines)
                and "|" in lines[i + 1]
                and "-" in lines[i + 1]
                and not _TABLE_SEPARATOR_RE.match(lines[i + 1])
            ):
                defects.add("malformed_table_separator")
        if _HTML_TAG_RE.search(line):
            defects.add("raw_html_present")
        i += 1
    if in_fence:
        defects.add("unclosed_fence")
    return MarkdownScan(
        section_count=section_count,
        table_count=table_count,
        fence_count=fence_count,
        defects=tuple(sorted(defects)),
    )


@dataclass(frozen=True)
class MarkdownFileResult:
    """One file's benchmark record."""

    path: str
    kind: str
    section_count: int
    table_count: int
    fence_count: int
    defects: tuple[str, ...]
    exact_reconstruction: bool
    runtime_ms: float
    peak_memory_kb: int


@dataclass(frozen=True)
class MarkdownBenchmarkReport:
    """Aggregate benchmark report over fixtures and/or corpus files."""

    label: str
    results: tuple[MarkdownFileResult, ...]

    @property
    def total_defects(self) -> int:
        return sum(len(r.defects) for r in self.results)

    def schema_errors(self) -> tuple[str, ...]:
        """Schema-validation violations; empty when the report is valid."""
        errors: list[str] = []
        if not self.results:
            return ("empty_report",)
        for i, r in enumerate(self.results):
            if not r.path:
                errors.append(f"result[{i}]: empty path")
            if r.kind not in ("html_fixture", "markdown_fixture", "corpus"):
                errors.append(f"result[{i}]: unknown kind {r.kind!r}")
            if r.section_count < 0 or r.table_count < 0 or r.fence_count < 0:
                errors.append(f"result[{i}]: negative structural count")
            if r.runtime_ms < 0 or r.peak_memory_kb < 0:
                errors.append(f"result[{i}]: negative measurement")
            if r.runtime_ms > MAX_RUNTIME_MS_PER_FILE:
                errors.append(f"result[{i}]: runtime {r.runtime_ms:.1f}ms exceeds limit")
            if r.peak_memory_kb > MAX_PEAK_MEMORY_KB:
                errors.append(f"result[{i}]: peak memory {r.peak_memory_kb}KB exceeds limit")
            if r.kind != "corpus" and not r.exact_reconstruction:
                errors.append(f"result[{i}]: fixture reconstruction is not exact")
            if r.kind != "corpus" and r.defects:
                errors.append(f"result[{i}]: fixture defects {r.defects}")
        return tuple(errors)

    def passes(self) -> bool:
        """Fixed gate: schema-valid and every fixture reconstructs exactly with zero defects."""
        return not self.schema_errors()


def _measure(work: Callable[[], Any]) -> tuple[Any, float, int]:
    """Run *work* (zero-arg callable), returning (result, elapsed_ms, peak_kb)."""
    tracemalloc.start()
    start = time.perf_counter()
    result = work()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_ms, peak // 1024


def _convert_html(path: Path) -> MarkdownFileResult:
    golden = path.with_suffix(".md")
    if not golden.exists():
        return MarkdownFileResult(
            path=path.as_posix(),
            kind="html_fixture",
            section_count=0,
            table_count=0,
            fence_count=0,
            defects=("missing_golden",),
            exact_reconstruction=False,
            runtime_ms=0.0,
            peak_memory_kb=0,
        )
    html = path.read_text(encoding="utf-8")

    def work() -> str | None:
        return html_to_markdown(html, min_words=1)

    result, elapsed_ms, peak_kb = _measure(work)
    if result is None:
        return MarkdownFileResult(
            path=path.as_posix(),
            kind="html_fixture",
            section_count=0,
            table_count=0,
            fence_count=0,
            defects=("empty_output",),
            exact_reconstruction=False,
            runtime_ms=elapsed_ms,
            peak_memory_kb=peak_kb,
        )
    scan = scan_markdown(result)
    expected = golden.read_text(encoding="utf-8")
    exact = normalize_whitespace(result).strip() == normalize_whitespace(expected).strip()
    return MarkdownFileResult(
        path=path.as_posix(),
        kind="html_fixture",
        section_count=scan.section_count,
        table_count=scan.table_count,
        fence_count=scan.fence_count,
        defects=scan.defects,
        exact_reconstruction=exact,
        runtime_ms=elapsed_ms,
        peak_memory_kb=peak_kb,
    )


def _parse_markdown(path: Path, kind: str, parser: NativeDocumentParser) -> MarkdownFileResult:
    text = path.read_text(encoding="utf-8")

    def work() -> object:
        return parser.parse(path, "guide")

    result, elapsed_ms, peak_kb = _measure(work)
    scan = scan_markdown(text)
    exact = normalize_whitespace(result.text).strip() == normalize_whitespace(text).strip()
    return MarkdownFileResult(
        path=path.as_posix(),
        kind=kind,
        section_count=scan.section_count,
        table_count=scan.table_count,
        fence_count=scan.fence_count,
        defects=scan.defects,
        exact_reconstruction=exact,
        runtime_ms=elapsed_ms,
        peak_memory_kb=peak_kb,
    )


def sample_corpus_files(corpus_dir: str | Path, limit: int = 100) -> tuple[Path, ...]:
    """Deterministically sample *limit* representative Markdown files.

    Balances across top-level subdirectories (stride walk of the sorted file
    list) so one huge subtree cannot dominate the sample.
    """
    root = Path(corpus_dir)
    if not root.exists():
        return ()
    md_files = sorted(p for p in root.rglob("*.md") if p.is_file())
    if not md_files:
        return ()
    if len(md_files) <= limit:
        return tuple(md_files)
    stride = max(1, len(md_files) // limit)
    sampled = md_files[::stride][:limit]
    return tuple(sampled)


def run_offline_benchmark(
    html_dir: str | Path,
    markdown_files: Sequence[str | Path] = (),
    corpus_dir: str | Path | None = None,
    corpus_limit: int = 100,
    label: str = "html_markdown",
    parser: NativeDocumentParser | None = None,
) -> MarkdownBenchmarkReport:
    """Run the offline structure benchmark.

    Converts every ``*.html`` fixture in *html_dir* and compares against the
    sibling ``*.md`` golden; parses the explicit *markdown_files* (fixtures)
    and an optional deterministic sample from *corpus_dir*.
    """
    parser = parser or NativeDocumentParser()
    results: list[MarkdownFileResult] = []
    html_dir = Path(html_dir)
    if html_dir.exists():
        for html in sorted(html_dir.glob("*.html")):
            results.append(_convert_html(html))
    for md in markdown_files:
        results.append(_parse_markdown(Path(md), "markdown_fixture", parser))
    if corpus_dir is not None:
        for md in sample_corpus_files(corpus_dir, corpus_limit):
            results.append(_parse_markdown(md, "corpus", parser))
    return MarkdownBenchmarkReport(label=label, results=tuple(results))


def main() -> None:
    argparser = argparse.ArgumentParser(description="Offline HTML/Markdown structure benchmark")
    argparser.add_argument("--html-dir", type=Path, default=Path("tests/fixtures/html_markdown"))
    argparser.add_argument("--markdown-files", nargs="*", type=Path, default=())
    argparser.add_argument("--corpus-dir", type=Path, default=None)
    argparser.add_argument("--corpus-limit", type=int, default=100)
    args = argparser.parse_args()

    report = run_offline_benchmark(
        args.html_dir,
        markdown_files=args.markdown_files,
        corpus_dir=args.corpus_dir,
        corpus_limit=args.corpus_limit,
    )
    print(f"label: {report.label}")
    print(f"files: {len(report.results)}  defects: {report.total_defects}  passes: {report.passes()}")
    for r in report.results:
        flags = "OK" if (r.exact_reconstruction and not r.defects) else "FAIL"
        print(
            f"  {flags} {r.kind:<16} {r.path} "
            f"sections={r.section_count} tables={r.table_count} fences={r.fence_count} "
            f"exact={r.exact_reconstruction} defects={','.join(r.defects) or '-'} "
            f"{r.runtime_ms:.1f}ms {r.peak_memory_kb}KB"
        )
    if not report.passes():
        raise SystemExit("benchmark gate failed: schema errors present")


if __name__ == "__main__":
    main()
