"""Tests for prompt augmentation evaluation metrics."""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.prompt_aug_metrics import (
    PromptAugMetrics,
    compute_citation_precision,
    compute_citation_recall,
    compute_format_compliance,
    compute_injection_defense_rate,
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
