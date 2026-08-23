"""dec evaluate runs RAGAS metrics only when --ragas is passed."""

from data_engineering_copilot.cli import build_parser


def test_ragas_defaults_off() -> None:
    args = build_parser().parse_args(["evaluate"])
    assert getattr(args, "ragas", False) is False


def test_ragas_opt_in() -> None:
    args = build_parser().parse_args(["evaluate", "--ragas"])
    assert args.ragas is True
