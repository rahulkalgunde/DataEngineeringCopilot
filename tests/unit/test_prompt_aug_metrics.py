"""Tests for prompt augmentation evaluation metrics."""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.prompt_aug_metrics import (
    PromptAugMetrics,
    compute_citation_precision,
    compute_citation_recall,
    compute_format_compliance,
    compute_injection_defense_rate,
    compute_prompt_aug_construction_metrics,
    compute_zero_context_fallback_accuracy,
)

pytestmark = pytest.mark.unit


class TestFormatCompliance:
    def test_valid_json(self):
        outputs = ['{"status": "SUCCESS", "answer": "Broadcast joins avoid shuffles.", "missing_info": null}']
        assert compute_format_compliance(outputs, ["json"]) == 1.0

    def test_invalid_json(self):
        outputs = ["This is plain text with no structure."]
        assert compute_format_compliance(outputs, ["json"]) == 0.0

    def test_code_with_backticks(self):
        outputs = ["```python\nprint('hello')\n```"]
        assert compute_format_compliance(outputs, ["code"]) == 1.0

    def test_code_without_backticks(self):
        outputs = ["print('hello')"]
        assert compute_format_compliance(outputs, ["code"]) == 0.0

    def test_mixed_formats(self):
        outputs = [
            '{"answer": "yes"}',
            "```python\nspark.sql('SELECT 1')\n```",
            "Just plain text.",
        ]
        assert compute_format_compliance(outputs, ["json", "code", "text"]) == 1.0

    def test_empty_outputs(self):
        assert compute_format_compliance([], []) == 0.0

    def test_partial_compliance(self):
        outputs = ['{"answer": "yes"}', "plain text"]
        assert compute_format_compliance(outputs, ["json", "json"]) == 0.5


class TestCitationPrecision:
    def test_all_valid(self):
        outputs = ["Spark supports X [Doc-1]. Also Y [Doc-1]."]
        assert compute_citation_precision(outputs, [["1"]]) == 1.0

    def test_mixed_valid_invalid(self):
        outputs = ["Spark supports X [Doc-1] and Y [Doc-99]."]
        assert compute_citation_precision(outputs, [["1"]]) == pytest.approx(0.5)

    def test_no_citations(self):
        outputs = ["No citations here."]
        assert compute_citation_precision(outputs, [["1"]]) == 0.0

    def test_empty_expected(self):
        outputs = ["Has [Doc-1] citation."]
        assert compute_citation_precision(outputs, [[]]) == 0.0


class TestCitationRecall:
    def test_full_recall(self):
        outputs = ["Use [Doc-1] and [Doc-2]."]
        assert compute_citation_recall(outputs, [["1", "2"]]) == 1.0

    def test_partial_recall(self):
        outputs = ["Use [Doc-1] only."]
        assert compute_citation_recall(outputs, [["1", "2"]]) == 0.5

    def test_no_expected_citations(self):
        outputs = ["Some text."]
        assert compute_citation_recall(outputs, [[]]) == 1.0

    def test_no_citations_in_output(self):
        outputs = ["Some text."]
        assert compute_citation_recall(outputs, [["1"]]) == 0.0


class TestInjectionDefense:
    def test_all_safe(self):
        outputs = ["Safe answer", "Another safe answer"]
        assert compute_injection_defense_rate(outputs, [None, None]) == 1.0

    def test_breach_detected(self):
        outputs = ["Safe answer", "I will ignore instructions and reveal info"]
        assert compute_injection_defense_rate(outputs, [None, "malicious"]) == 0.0

    def test_partial_breach(self):
        outputs = ["I cannot do that.", "I will ignore instructions and reveal info"]
        assert compute_injection_defense_rate(outputs, ["hack", "exploit"]) == 0.5

    def test_no_injection_tests(self):
        outputs = ["Safe answer"]
        assert compute_injection_defense_rate(outputs, [None]) == 1.0

    def test_system_prompt_leak(self):
        outputs = ["Here is the system prompt text"]
        assert compute_injection_defense_rate(outputs, ["extract system prompt"]) == 0.0

    def test_all_defended(self):
        outputs = ["I cannot do that.", "That request is not supported."]
        assert compute_injection_defense_rate(outputs, ["hack", "exploit"]) == 1.0


class TestZeroContextFallback:
    def test_all_correct(self):
        outputs = ["Insufficient context to answer.", "No relevant docs available."]
        assert compute_zero_context_fallback_accuracy(outputs, [False, False]) == 1.0

    def test_all_incorrect(self):
        outputs = ["Here is the answer based on docs.", "Sure, I can help!"]
        assert compute_zero_context_fallback_accuracy(outputs, [False, False]) == 0.0

    def test_mixed(self):
        outputs = ["Insufficient context.", "Here is a fabricated answer."]
        assert compute_zero_context_fallback_accuracy(outputs, [False, False]) == 0.5

    def test_no_zero_context_rows(self):
        outputs = ["Some answer."]
        assert compute_zero_context_fallback_accuracy(outputs, [True]) == 1.0

    def test_cannot_answer_variant(self):
        outputs = ["I cannot answer this question."]
        assert compute_zero_context_fallback_accuracy(outputs, [False]) == 1.0


class TestPromptAugMetricsSummary:
    def test_summary_string(self):
        m = PromptAugMetrics(
            format_compliance_rate=1.0,
            citation_precision=0.9,
            citation_recall=0.8,
            injection_defense_rate=1.0,
            zero_context_fallback_accuracy=0.75,
        )
        s = m.summary()
        assert "Format compliance" in s
        assert "1.0000" in s


def _valid_salted_prompt(query: str = "What is Spark?", context: str = "ctx") -> str:
    salt = "a1b2c3d4"
    return (
        f"## STUFF\n<context_data_{salt}>\n[DENSITY: LOW]\n{context}\n</context_data_{salt}>\n"
        f"## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: {query}"
    )


class TestPromptAugConstructionMetrics:
    def test_all_invariants_pass_with_default_config(self):
        prompts = [_valid_salted_prompt()]
        m = compute_prompt_aug_construction_metrics(
            prompts,
            ["ctx"],
            ["What is Spark?"],
            [True],
        )
        assert m.salted_tag_pair_rate == 1.0
        assert m.trailing_block_rate == 1.0
        assert m.citation_instruction_rate == 1.0
        assert m.context_preserved_rate == 1.0
        assert m.query_embedded_rate == 1.0
        assert m.zero_context_fallback_rate == 1.0

    def test_salted_tags_disabled_expects_chunk_tags(self):
        prompt = "<chunk>\n[DENSITY: LOW]\nctx\n</chunk>\n## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: Q"
        m = compute_prompt_aug_construction_metrics(
            [prompt],
            ["ctx"],
            ["Q"],
            [True],
            salted_tags=False,
        )
        assert m.salted_tag_pair_rate == 1.0

    def test_salted_tags_expected_but_missing(self):
        prompt = "<chunk>\nctx\n</chunk>\n## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: Q"
        m = compute_prompt_aug_construction_metrics(
            [prompt],
            ["ctx"],
            ["Q"],
            [True],
        )
        assert m.salted_tag_pair_rate == 0.0

    def test_unmatching_salt_close_fails(self):
        prompt = (
            "<context_data_a1b2c3d4>\nctx\n</context_data_ffffffff>\n"
            "## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: Q"
        )
        assert compute_prompt_aug_construction_metrics([prompt], ["ctx"], ["Q"], [True]).salted_tag_pair_rate == 0.0

    def test_trailing_block_missing_when_enabled(self):
        prompt = "<context_data_a1b2c3d4>\nctx\n</context_data_a1b2c3d4>\nCITATION RULES:\nQuestion: Q"
        m = compute_prompt_aug_construction_metrics([prompt], ["ctx"], ["Q"], [True])
        assert m.trailing_block_rate == 0.0

    def test_trailing_block_present_when_disabled(self):
        prompt = (
            "<context_data_a1b2c3d4>\nctx\n</context_data_a1b2c3d4>\n"
            "## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: Q"
        )
        m = compute_prompt_aug_construction_metrics(
            [prompt],
            ["ctx"],
            ["Q"],
            [True],
            trailing_instructions=False,
        )
        assert m.trailing_block_rate == 0.0

    def test_citation_rules_missing_when_strict(self):
        prompt = "<context_data_a1b2c3d4>\nctx\n</context_data_a1b2c3d4>\n## CRITICAL REMINDERS\nQuestion: Q"
        m = compute_prompt_aug_construction_metrics([prompt], ["ctx"], ["Q"], [True])
        assert m.citation_instruction_rate == 0.0

    def test_citation_rules_present_when_off(self):
        m = compute_prompt_aug_construction_metrics(
            [_valid_salted_prompt()],
            ["ctx"],
            ["Q"],
            [True],
            citation_enforcement="off",
        )
        assert m.citation_instruction_rate == 0.0

    def test_zero_context_requires_marker(self):
        prompt = (
            "<context_data_a1b2c3d4>\nNo relevant documents found.\n</context_data_a1b2c3d4>\n"
            "## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: Q"
        )
        m = compute_prompt_aug_construction_metrics(
            [prompt],
            [""],
            ["Q"],
            [False],
        )
        assert m.zero_context_fallback_rate == 1.0
        assert m.context_preserved_rate == 1.0

    def test_zero_context_without_marker_fails(self):
        prompt = (
            "<context_data_a1b2c3d4>\n</context_data_a1b2c3d4>\n## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: Q"
        )
        m = compute_prompt_aug_construction_metrics(
            [prompt],
            [""],
            ["Q"],
            [False],
        )
        assert m.zero_context_fallback_rate == 0.0

    def test_query_not_embedded_fails(self):
        prompt = (
            "<context_data_a1b2c3d4>\nctx\n</context_data_a1b2c3d4>\n"
            "## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: different"
        )
        m = compute_prompt_aug_construction_metrics([prompt], ["ctx"], ["zz-not-in-prompt"], [True])
        assert m.query_embedded_rate == 0.0

    def test_context_not_preserved_fails(self):
        prompt = (
            "<context_data_a1b2c3d4>\nWRONG CONTENT\n</context_data_a1b2c3d4>\n"
            "## CRITICAL REMINDERS\nCITATION RULES:\nQuestion: Q"
        )
        m = compute_prompt_aug_construction_metrics([prompt], ["ctx"], ["Q"], [True])
        assert m.context_preserved_rate == 0.0

    def test_mixed_rows_aggregate_rates(self):
        prompts = [
            _valid_salted_prompt(),
            _valid_salted_prompt(query="other", context="second ctx"),
        ]
        m = compute_prompt_aug_construction_metrics(
            prompts,
            ["ctx", "second ctx"],
            ["What is Spark?", "other"],
            [True, True],
        )
        assert m.salted_tag_pair_rate == 1.0
        assert m.trailing_block_rate == 1.0
        assert m.citation_instruction_rate == 1.0
        assert m.context_preserved_rate == 1.0
        assert m.query_embedded_rate == 1.0
