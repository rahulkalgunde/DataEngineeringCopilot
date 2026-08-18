"""Contract tests for generation naming — prevents cross-module naming bugs."""

from __future__ import annotations

from data_engineering_copilot.config.naming import (
    GenerationNaming,
    resolve_naming,
    validate_naming,
)


def test_artifact_dir_equals_collection_name() -> None:
    """Core contract: artifact_dir_name MUST equal collection_name."""
    for gen_id in ["pinned-abc123", "pinned-xyz789", "spark-abc", "test-gen"]:
        naming = resolve_naming(gen_id)
        assert naming.artifact_dir_name == naming.collection_name, (
            f"FAIL for {gen_id}: artifact_dir={naming.artifact_dir_name} != collection={naming.collection_name}"
        )


def test_collection_name_format() -> None:
    naming = resolve_naming("pinned-abc123")
    assert naming.collection_name == "data_engineering_docs__pinned-abc123"
    assert naming.collection_name.startswith("data_engineering_docs__")


def test_validate_naming_passes_for_valid() -> None:
    naming = resolve_naming("pinned-abc123")
    validate_naming(naming)


def test_validate_naming_fails_on_mismatch() -> None:
    bad = GenerationNaming(
        generation_id="x",
        collection_name="data_engineering_docs__x",
        artifact_dir_name="different-name",
        active_alias="data_engineering_docs",
    )
    try:
        validate_naming(bad)
        raise AssertionError("should have raised RuntimeError")
    except RuntimeError as exc:
        assert "artifact_dir_name" in str(exc)
        assert "collection_name" in str(exc)


def test_resolve_naming_rejects_empty() -> None:
    try:
        resolve_naming("")
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass


def test_cli_uses_naming_module() -> None:
    """CLI must derive collections through the naming module."""
    import data_engineering_copilot.cli as cli

    assert cli.resolve_naming is resolve_naming
    assert cli.validate_naming is validate_naming
    assert cli._spark_generation_collection("pinned-abc123") == resolve_naming("pinned-abc123").collection_name


def test_pinned_index_builder_validates_naming() -> None:
    """Builder must validate naming contract in __init__."""
    import inspect

    from data_engineering_copilot.config.naming import validate_naming as real_validate
    from data_engineering_copilot.services import pinned_index_builder

    assert pinned_index_builder.resolve_naming is resolve_naming
    assert pinned_index_builder.validate_naming is real_validate
    assert "validate_naming" in inspect.getsource(pinned_index_builder.PinnedIndexBuilder.__init__)
