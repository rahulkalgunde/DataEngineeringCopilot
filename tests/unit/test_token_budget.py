"""Task 8 tests: lossless token-budget splitting and validation."""

from __future__ import annotations

import textwrap

import pytest

from data_engineering_copilot.infrastructure.token_budget import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    count_tokens,
    split_text_losslessly,
    validate_segments,
)


def test_empty_and_whitespace_input() -> None:
    assert split_text_losslessly("") == []
    assert split_text_losslessly("   \n  ") == []


def test_short_text_returns_single_segment() -> None:
    text = "Apache Spark SQL structured data processing engine"
    segments = split_text_losslessly(text)
    assert segments == [text]


def test_invalid_budgets_raise() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        split_text_losslessly("x", max_tokens=0)
    with pytest.raises(ValueError, match="max_chars"):
        split_text_losslessly("x", max_chars=0)


def test_10000_token_document_becomes_complete_segments() -> None:
    text = " ".join(["word"] * 10000)
    segments = split_text_losslessly(text, max_tokens=3800, max_chars=6000)
    assert len(segments) > 1
    for segment in segments:
        assert count_tokens(segment) <= 3800
        assert len(segment) <= 6000
    assert "".join(segments).strip() == text.strip()


def test_no_segment_exceeds_budgets() -> None:
    text = " ".join(["word"] * 8000)
    segments = split_text_losslessly(text, max_tokens=1000, max_chars=3000)
    for segment in segments:
        assert count_tokens(segment) <= 1000
        assert len(segment) <= 3000


def test_paragraph_boundaries_preferred() -> None:
    text = textwrap.dedent(
        """\
        # Heading
        First paragraph with a decent number of words that is used to test splitting.

        Second paragraph here also with a decent number of words for testing splitting.
        """
    ).strip()
    segments = split_text_losslessly(text, max_tokens=30, max_chars=2000)
    assert len(segments) >= 2
    assert "".join(segments).strip() == text.strip()


def test_code_fence_not_split_when_fits() -> None:
    text = textwrap.dedent(
        """\
        # Window Functions

        Spark supports window functions that compute a result per row over a group.

        ```sql
        SELECT dense_rank() OVER (PARTITION BY category ORDER BY amount DESC)
        FROM sales
        ```

        These are useful for ranking and rolling aggregates.
        """
    ).strip()
    segments = split_text_losslessly(text, max_tokens=40, max_chars=2000)
    # The fence must appear whole inside one segment (two fence markers = one block).
    fence_counts = [s.count("```") for s in segments if "```" in s]
    assert fence_counts, "fence must survive"
    assert all(count % 2 == 0 for count in fence_counts)
    assert "".join(segments).strip() == text.strip()


def test_oversized_code_fence_split_losslessly() -> None:
    fence = "```python\n" + "\n".join(f"line{i} = {i} + 1  # padding words to grow" for i in range(80)) + "\n```"
    segments = split_text_losslessly(fence, max_tokens=60, max_chars=2000)
    assert len(segments) > 1
    for segment in segments:
        assert count_tokens(segment) <= 60
        assert len(segment) <= 2000
    assert "".join(segments).strip() == fence.strip()
    assert "line79" in "".join(segments)


def test_fence_continuation_piece_at_max_chars_does_not_overflow() -> None:
    """Regression: an intermediate fence piece that packs to exactly max_chars
    was rejected once its trailing newline was attached (piece became
    max_chars+1). The packer must reserve the newline (and closer) up front.

    Lines here pack so that ``opener + line_a + line_b`` is exactly
    ``max_chars``; the next line then forces a flush, and the flushed piece
    (previous content + trailing newline) must still fit.
    """
    max_chars = 1024
    opener = "```"
    closer = "```"
    sep = 2  # two "\n" join separators inside the three-line candidate
    line_a = "a" * ((max_chars - len(opener) - sep) // 2)
    line_b = "b" * (max_chars - len(opener) - sep - len(line_a))
    line_c = "c"
    fence = f"{opener}\n{line_a}\n{line_b}\n{line_c}\n{closer}"

    assert len(f"{opener}\n{line_a}\n{line_b}") == max_chars  # the tight candidate

    segments = split_text_losslessly(fence, max_tokens=4000, max_chars=max_chars)
    assert len(segments) > 1
    for segment in segments:
        assert len(segment) <= max_chars, f"piece of {len(segment)} chars > {max_chars}"
    assert "".join(segments).strip() == fence.strip()


@pytest.mark.parametrize("max_chars", [256, 384, 512, 768, 1024, 1536, 2048])
def test_fence_pieces_never_exceed_max_chars_across_boundaries(max_chars: int) -> None:
    """Deterministic property sweep: for several budgets, no emitted fence
    piece (including its trailing newline or closer) may exceed ``max_chars``,
    and reconstruction must be lossless. Mirrors the original off-by-one bug
    (a piece of exactly ``max_chars`` became ``max_chars + 1`` once its
    trailing newline was attached) across multiple boundaries."""
    opener = "```"
    closer = "```"
    sep = 2  # two "\n" join separators inside a three-line candidate
    line_a = "a" * ((max_chars - len(opener) - sep) // 2)
    line_b = "b" * (max_chars - len(opener) - sep - len(line_a))
    fence = f"{opener}\n{line_a}\n{line_b}\n{closer}"

    assert len(f"{opener}\n{line_a}\n{line_b}") == max_chars  # tightest candidate

    segments = split_text_losslessly(fence, max_tokens=4000, max_chars=max_chars)
    assert len(segments) > 1
    for segment in segments:
        assert len(segment) <= max_chars, f"piece of {len(segment)} chars > {max_chars}"
    assert "".join(segments).strip() == fence.strip()


def test_validate_segments_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="reconstruction"):
        validate_segments("original text here", ["original", " text", " HERE"])


def test_validate_segments_matches() -> None:
    validate_segments("original text", ["original text"])


def test_single_token_over_budget_raises() -> None:
    with pytest.raises(ValueError, match="exceeds budget"):
        split_text_losslessly("x" * (DEFAULT_MAX_CHARS + 100))


def test_custom_separators_honored() -> None:
    text = "alpha | beta | gamma | delta"
    segments = split_text_losslessly(text, max_tokens=2, max_chars=1000, separators=(" | ",))
    assert len(segments) >= 2
    assert "".join(segments).strip() == text.strip()


def test_default_budgets() -> None:
    assert DEFAULT_MAX_TOKENS == 3800
    assert DEFAULT_MAX_CHARS == 6000
