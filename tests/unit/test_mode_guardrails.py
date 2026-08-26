"""Tests for services/mode_guardrails.py."""

from __future__ import annotations

from data_engineering_copilot.services.mode_guardrails import (
    build_mode_guardrail_block,
    detect_modes,
)


class TestDetectModes:
    def test_detects_yarn(self) -> None:
        assert "yarn" in detect_modes("How does Spark run on YARN?")

    def test_detects_kubernetes(self) -> None:
        assert "kubernetes" in detect_modes("Spark on Kubernetes")

    def test_detects_k8s_abbreviation(self) -> None:
        assert "kubernetes" in detect_modes("Spark on k8s")

    def test_no_modes(self) -> None:
        assert detect_modes("What is Spark?") == []


class TestBuildModeGuardrailBlock:
    def test_returns_block_for_yarn(self) -> None:
        block = build_mode_guardrail_block("How does Spark run on YARN?")
        assert block is not None
        assert "VERIFIED DOCUMENTATION FACTS" in block
        assert "yarn" in block.lower()

    def test_returns_none_for_unrelated(self) -> None:
        assert build_mode_guardrail_block("What is Spark?") is None

    def test_includes_multiple_modes(self) -> None:
        block = build_mode_guardrail_block("Compare YARN and Kubernetes")
        assert block is not None
        assert "[yarn]" in block
        assert "[kubernetes]" in block
