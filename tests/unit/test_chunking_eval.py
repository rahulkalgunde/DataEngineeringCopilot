from pathlib import Path


def test_chunking_eval_wrapper_exists() -> None:
    assert Path("scripts/run_chunking_eval.py").exists()
