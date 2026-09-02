"""Unit tests for gen-bm25-rebuild CLI."""

from __future__ import annotations


def test_gen_bm25_rebuild_help():
    from data_engineering_copilot.cli import build_parser

    parser = build_parser()
    # Parse help flag via ArgumentParser — should exit 0
    try:
        parser.parse_args(["gen-bm25-rebuild", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit for --help")

    # Check help output contains generation
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parser.parse_args(["gen-bm25-rebuild", "--help"])
    output = buf.getvalue()
    assert "generation" in output.lower()


def test_gen_bm25_rebuild_function_exists():
    from data_engineering_copilot import cli

    assert hasattr(cli, "gen_bm25_rebuild")
    assert callable(cli.gen_bm25_rebuild)
