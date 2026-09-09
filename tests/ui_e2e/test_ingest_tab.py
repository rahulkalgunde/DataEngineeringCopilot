"""Ingestion tab E2E.

Strict contracts (zero-cost, always determinable):
- ``🔄 Start`` is disabled when no source is selected and enabled otherwise.
- The number input accepts a small ``max_pages`` value.

Start click is exercised but bounded: after ``Start`` the UI must react (a
visible warning/error from the ingestion API *or* a Stop/Dismiss control for a
live run), and we immediately stop/reset any run that appears so the test never
leaves a crawl on the shared box.
"""

import pytest

from .conftest import Tab, open_tab, visible_button

pytestmark = pytest.mark.ui

ALL_SOURCES = ("Apache Spark Documentation", "Databricks Documentation", "Delta Lake Documentation")


def _open_ingest(page):
    open_tab(page, Tab.INGEST)
    page.get_by_test_id("stMultiSelect").filter(visible=True).wait_for(timeout=5000)


def _clear_multiselect(page) -> None:
    """Remove all selected source tags via Delete key in the filter input."""
    ms = page.get_by_test_id("stMultiSelect").filter(visible=True)
    closes = ms.locator('[data-baseweb="tag"]').count()
    for _ in range(closes + 1):
        inputs = ms.locator("input")
        inputs.first.click()
        page.wait_for_timeout(250)
        page.keyboard.press("Delete")
        page.wait_for_timeout(700)
        if ms.locator('[data-baseweb="tag"]').count() == 0:
            return


def test_start_disabled_without_sources(page):
    """Strict: with every source deselected the Start button is disabled."""
    _open_ingest(page)
    start = visible_button(page, "🔄 Start")
    assert start.count() == 1
    assert start.first.is_enabled()

    _clear_multiselect(page)
    page.wait_for_timeout(800)
    start = visible_button(page, "🔄 Start")
    assert start.count() == 1
    assert start.first.is_disabled()


def test_start_enabled_after_reselecting_source(page):
    """Strict: clearing then re-selecting one source re-enables Start."""
    _open_ingest(page)
    _clear_multiselect(page)
    page.wait_for_timeout(800)

    ms = page.get_by_test_id("stMultiSelect").filter(visible=True)
    ms.locator('[data-baseweb="select"]').first.click()
    page.wait_for_timeout(600)
    option = page.get_by_role("option").filter(has_text=ALL_SOURCES[0]).filter(visible=True).first
    if option.count() == 0:
        # baseweb renders a virtualized menu; fall back to tappable list items.
        option = page.locator('[data-baseweb="menu"] [aria-disabled="false"]').filter(has_text=ALL_SOURCES[0])
    option.click()
    page.wait_for_timeout(800)

    start = visible_button(page, "🔄 Start")
    assert start.first.is_enabled()


def test_number_input_accepts_small_value(page):
    """Strict: the max-pages number input reflects a typed value."""
    _open_ingest(page)
    number = page.get_by_test_id("stNumberInput").filter(visible=True).locator("input")
    number.first.fill("10")
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)
    assert number.first.input_value() == "10"


def test_start_button_click_reacts_or_starts(page):
    """Tolerant: Start click must produce a visible reaction.

    Depending on broker/worker state this is (a) an ingestion API warning/error
    rendered in the tab, or (b) a live run surfaced as a Stop/Dismiss control —
    which we then stop to leave the box clean.
    """
    _open_ingest(page)
    start = visible_button(page, "🔄 Start")
    assert start.count() == 1 and start.first.is_enabled()

    # Bound the run if the worker is healthy enough to start one.
    number = page.get_by_test_id("stNumberInput").filter(visible=True).locator("input")
    number.first.fill("1")
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)

    start.first.click()
    page.wait_for_timeout(2500)

    body = page.get_by_test_id("stMainBlockContainer").inner_text(timeout=5000)

    st_button = page.get_by_test_id("stButton").filter(visible=True)
    stop = st_button.get_by_text("⏹ Stop", exact=True).filter(visible=True)
    dismiss = st_button.get_by_text("Dismiss", exact=True).filter(visible=True)

    if stop.count() == 1:
        # A run materialized — verify the progress sub-tabs (Overview/Sources/
        # Live Log/History) render, then stop to leave the box clean.
        for sub in ("Overview", "Sources", "Live Log", "History"):
            assert page.get_by_role("tab", name=sub, exact=True).filter(visible=True).count() == 1
        stop.first.click()
        page.wait_for_timeout(1500)
        assert visible_button(page, "🔄 Start").count() in (0, 1)
        return
    if dismiss.count() == 1:
        dismiss.first.click()
        page.wait_for_timeout(1500)
        return

    # No run materialized (worker/broker unavailable): the UI must have shown
    # the ingestion API's guard reaction instead.
    assert any(
        marker in body
        for marker in (
            "Already running.",
            "Cannot start ingestion",
            "API unreachable",
            "Failed to start",
            "Qdrant",
        )
    ), f"Start click produced no visible reaction:\n{body[:400]}"
