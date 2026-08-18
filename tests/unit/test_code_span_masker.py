"""Unit tests for the code-span masker."""

from __future__ import annotations

from data_engineering_copilot.services.code_span_masker import (
    mask_code_spans,
    unmask_code_spans,
)


class TestCodeSpanMasker:
    def test_fenced_code_round_trip(self):
        text = "Intro.\n```python\ndef foo():\n    return 1\n```\nOutro."
        masked = mask_code_spans(text)
        assert masked.text != text
        assert "def foo():" not in masked.text
        assert unmask_code_spans(masked.text, masked) == text

    def test_tilde_fence_round_trip(self):
        text = "```\ncode\n```\n\n~~~txt\nmore\n~~~\ndone"
        masked = mask_code_spans(text)
        assert "code" not in masked.text
        assert "more" not in masked.text
        assert unmask_code_spans(masked.text, masked) == text

    def test_language_tag_masked(self):
        text = "```python\nprint(1)\n```"
        masked = mask_code_spans(text)
        assert "python" not in masked.text

    def test_inline_code_round_trip(self):
        text = "Use `spark.sql()` for queries."
        masked = mask_code_spans(text)
        assert "spark.sql()" not in masked.text
        assert unmask_code_spans(masked.text, masked) == text

    def test_multiple_spans_round_trip(self):
        text = "`a` then ```\nb\n``` then `c` then ~~~d~~~ then `e`"
        masked = mask_code_spans(text)
        assert unmask_code_spans(masked.text, masked) == text

    def test_double_backtick_inline_round_trip(self):
        text = "Use ``a `b` c`` here."
        masked = mask_code_spans(text)
        assert unmask_code_spans(masked.text, masked) == text

    def test_malformed_fence_returns_text_unchanged(self):
        text = "```python\ndef foo():\n    return 1\nno closing fence"
        masked = mask_code_spans(text)
        assert masked.text == text
        assert masked.spans == ()

    def test_placeholder_like_text_not_colliding(self):
        text = "Sentinel CODE0 here. `real code` done. CODE0 again."
        masked = mask_code_spans(text)
        assert unmask_code_spans(masked.text, masked) == text
        assert "real code" not in masked.text

    def test_plain_text_unchanged(self):
        text = "Just prose with no code at all."
        masked = mask_code_spans(text)
        assert masked.text == text
        assert masked.spans == ()

    def test_empty_text(self):
        masked = mask_code_spans("")
        assert masked.text == ""
        assert masked.spans == ()

    def test_spans_store_originals(self):
        text = "before\n```\nsecret\n```\nafter"
        masked = mask_code_spans(text)
        originals = [original for _sentinel, original in masked.spans]
        assert "secret" in "".join(originals)
        assert len(masked.spans) >= 1

    def test_unmask_is_idempotent_for_unmasked_input(self):
        text = "plain prose"
        masked = mask_code_spans(text)
        assert unmask_code_spans(text, masked) == text
