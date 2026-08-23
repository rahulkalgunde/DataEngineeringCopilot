"""Default eval-coverage sweep must include golden recall datasets."""

from data_engineering_copilot.cli import _default_coverage_paths


def test_sweep_includes_golden_recall_files(tmp_path):
    (tmp_path / "golden").mkdir()
    (tmp_path / "recall_top.jsonl").write_text('{"question":"q","expected_terms":["x"]}\n')
    (tmp_path / "golden" / "recall_all.jsonl").write_text('{"question":"q","expected_terms":["x"]}\n')
    (tmp_path / "golden" / "qa_spark.jsonl").write_text('{"question":"q","ground_truth":"g"}\n')
    paths = _default_coverage_paths(tmp_path)
    names = [p.name for p in paths]
    assert "recall_top.jsonl" in names
    assert "recall_all.jsonl" in names  # golden recall included
    assert "qa_spark.jsonl" not in names  # qa rows are not recall rows
