"""Metrics tab E2E: fresh-session empty state + reset button surfacing."""

import pytest

from .conftest import Tab, open_tab

pytestmark = pytest.mark.ui

EMPTY_STATE = "No queries recorded yet. Ask questions in the 💬 Ask tab to see metrics."


def test_metrics_empty_state(page):
    """Strict: with no queries recorded the tab shows the empty-state info."""
    open_tab(page, Tab.METRICS)
    page.get_by_text("RAG Service Metrics", exact=True).filter(visible=True).wait_for(timeout=5000)
    main = page.get_by_test_id("stMainBlockContainer")
    main.get_by_test_id("stAlertContentInfo").filter(visible=True).first.wait_for(timeout=5000)
    assert EMPTY_STATE in main.get_by_test_id("stAlertContentInfo").filter(visible=True).first.inner_text()
    assert main.get_by_test_id("stMetric").filter(visible=True).count() == 0


def test_metrics_never_shows_reset_button_before_any_query(page):
    """Strict: Reset Metrics only exists once a query was recorded."""
    open_tab(page, Tab.METRICS)
    page.get_by_test_id("stAlertContentInfo").filter(visible=True).first.wait_for(timeout=5000)
    assert page.get_by_role("button", name="Reset Metrics", exact=True).filter(visible=True).count() == 0
