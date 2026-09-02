from pathlib import Path


def test_promote_baseline_exists() -> None:
    assert Path("scripts/promote_baseline.py").exists()


def test_baseline_has_provenance() -> None:
    # When baseline is promoted, provenance sidecar must exist;
    # when deferred, ADR-016 documents the deferral (stale baseline kept).
    has_prov = Path("tests/evaluation/benchmarks/baseline_inscope.provenance.json").exists()
    has_adr = (
        Path("docs/adr/ADR-016-baseline-refresh.md").exists()
        or Path("docs/adr/ADR-016-baseline-refresh-deferred.md").exists()
    )
    assert has_prov or has_adr, "expected provenance sidecar or ADR-016 deferral"
