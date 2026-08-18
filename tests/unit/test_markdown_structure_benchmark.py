"""Tests for the offline HTML/Markdown structure benchmark (plan Task 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_engineering_copilot.evaluation.markdown_structure_benchmark import (
    MAX_PEAK_MEMORY_KB,
    MAX_RUNTIME_MS_PER_FILE,
    MarkdownBenchmarkReport,
    MarkdownFileResult,
    run_offline_benchmark,
    sample_corpus_files,
    scan_markdown,
)

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "html_markdown"


@pytest.fixture(scope="module")
def report() -> MarkdownBenchmarkReport:
    structure = _FIXTURES / "structure.md"
    return run_offline_benchmark(
        _FIXTURES,
        markdown_files=[structure],
        label="test-fixtures",
    )


def test_benchmark_passes_on_fixtures(report) -> None:
    assert report.schema_errors() == ()
    assert report.passes()


def test_fixture_high_water_mark(report) -> None:
    max_ms = max(r.runtime_ms for r in report.results)
    max_kb = max(r.peak_memory_kb for r in report.results)
    assert max_ms < MAX_RUNTIME_MS_PER_FILE
    assert max_kb < MAX_PEAK_MEMORY_KB


def test_all_html_fixtures_reconstruct_exactly(report) -> None:
    html_results = [r for r in report.results if r.kind == "html_fixture"]
    assert len(html_results) == 5
    assert all(r.exact_reconstruction for r in html_results)
    assert all(not r.defects for r in html_results)


def test_structure_fixture_reconstructs_exactly(report) -> None:
    structure = [r for r in report.results if r.kind == "markdown_fixture"]
    assert len(structure) == 1
    assert structure[0].exact_reconstruction
    assert structure[0].defects == ()


def test_structure_counts(report) -> None:
    structure = [r for r in report.results if r.kind == "markdown_fixture"][0]
    assert structure.section_count == 4  # #, ##, ###, ##
    assert structure.table_count == 1
    assert structure.fence_count == 2


def test_code_fixture_counts(report) -> None:
    code = [r for r in report.results if r.path.endswith("code.html")][0]
    assert code.fence_count == 2
    assert code.exact_reconstruction


def test_table_fixture_counts(report) -> None:
    table = [r for r in report.results if r.path.endswith("table.html")][0]
    assert table.table_count == 1
    assert table.exact_reconstruction


def test_navigation_fixture_has_no_defects(report) -> None:
    nav = [r for r in report.results if r.path.endswith("navigation.html")][0]
    assert nav.exact_reconstruction
    assert nav.defects == ()


def test_scan_flags_unclosed_fence() -> None:
    scan = scan_markdown("prose\n```python\nx = 1\n")
    assert "unclosed_fence" in scan.defects
    assert scan.fence_count == 0


def test_scan_counts_closed_fences() -> None:
    scan = scan_markdown("```python\nx = 1\n```\n\n```sql\nselect 1\n```\n")
    assert scan.fence_count == 2
    assert scan.defects == ()


def test_scan_counts_heading_and_table() -> None:
    text = "# Title\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n## Sub\n\nbody"
    scan = scan_markdown(text)
    assert scan.section_count == 2
    assert scan.table_count == 1


def test_scan_flags_malformed_table_separator() -> None:
    text = "# T\n\n| A | B |\n| -- | x- |\n| 1 | 2 |\n"
    scan = scan_markdown(text)
    assert "malformed_table_separator" in scan.defects


def test_scan_flags_raw_html_outside_fences() -> None:
    scan = scan_markdown("text <b>bold</b> here\n\n```\n<div>kept</div>\n```\n")
    assert "raw_html_present" in scan.defects


def test_scan_ignores_html_inside_fences() -> None:
    scan = scan_markdown("```html\n<div>kept</div>\n```\n")
    assert scan.defects == ()


def test_sample_corpus_files_is_deterministic(tmp_path) -> None:
    for i in range(10):
        sub = tmp_path / f"sub{i % 2}"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"file{i}.md").write_text("body\n")
    first = sample_corpus_files(tmp_path, limit=6)
    second = sample_corpus_files(tmp_path, limit=6)
    assert first == second
    assert len(first) == 6
    # Balanced across both subdirectories.
    assert len({p.parent.name for p in first}) == 2


def test_sample_corpus_files_missing_dir() -> None:
    assert sample_corpus_files("/nonexistent/path", limit=5) == ()


def test_schema_error_for_non_exact_fixture() -> None:
    report = MarkdownBenchmarkReport(
        label="broken",
        results=(
            MarkdownFileResult(
                path="x.html",
                kind="html_fixture",
                section_count=1,
                table_count=0,
                fence_count=0,
                defects=(),
                exact_reconstruction=False,
                runtime_ms=1.0,
                peak_memory_kb=1,
            ),
        ),
    )
    assert not report.passes()
    assert "fixture reconstruction is not exact" in " ".join(report.schema_errors())


def test_empty_report_fails_gate() -> None:
    report = MarkdownBenchmarkReport(label="empty", results=())
    assert not report.passes()
    assert report.schema_errors() == ("empty_report",)
