"""Tests for config/naming.py."""

from __future__ import annotations

import pytest

from data_engineering_copilot.config.naming import (
    GenerationNaming,
    resolve_naming,
    validate_naming,
)


class TestResolveNaming:
    def test_basic(self) -> None:
        n = resolve_naming("pinned-abc123")
        assert n.generation_id == "pinned-abc123"
        assert n.collection_name == "data_engineering_docs__pinned-abc123"
        assert n.artifact_dir_name == n.collection_name
        assert n.active_alias == "data_engineering_docs"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            resolve_naming("")


class TestValidateNaming:
    def test_valid_passes(self) -> None:
        n = resolve_naming("test-123")
        validate_naming(n)

    def test_mismatched_dir_raises(self) -> None:
        n = GenerationNaming(
            generation_id="x",
            collection_name="data_engineering_docs__x",
            artifact_dir_name="wrong",
            active_alias="data_engineering_docs",
        )
        with pytest.raises(RuntimeError, match="Contract violated"):
            validate_naming(n)

    def test_missing_prefix_raises(self) -> None:
        n = GenerationNaming(
            generation_id="x",
            collection_name="wrong__x",
            artifact_dir_name="wrong__x",
            active_alias="data_engineering_docs",
        )
        with pytest.raises(RuntimeError, match="must start with"):
            validate_naming(n)
