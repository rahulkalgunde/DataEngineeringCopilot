"""Ask tab E2E: empty-question guard (strict) + one live RAG query and feedback.

The live query runs the full pipeline (retrieval + rerank + LLM) against the
running stack on the free/local provider chain only. Assertions are tolerant:
the pipeline must *react* (answer, fallback answer, or explicit error), not
return a specific string.
"""

import pytest

from .conftest import Tab, open_tab, visible_button

# This module hits the live LLM (up to ~2 min cold) so it overrides the global
# 60s pytest timeout. Pure-UI modules keep the default and only ui.
pytestmark = [pytest.mark.ui, pytest.mark.timeout(300)]


def test_empty_question_guard(page):
    """Strict: clicking Ask with an empty question shows the guard warning."""
    open_tab(page, Tab.ASK)
    textarea = page.get_by_test_id("stTextArea").filter(visible=True)
    textarea.wait_for(timeout=10000)
    ask_btn = visible_button(page, "Ask")
    ask_btn.first.wait_for(timeout=10000)
    guard = page.get_by_text("Enter a question.", exact=True).filter(visible=True)
    # The app's initial health-check rerun can drop the first click; retry until
    # the guard proves the click registered.
    for _ in range(5):
        ask_btn.first.click()
        try:
            guard.wait_for(timeout=5000)
            return
        except Exception:
            continue
    guard.wait_for(timeout=15000)


def ask_question(session_page, question: str) -> None:
    open_tab(session_page, Tab.ASK)
    textarea = session_page.get_by_test_id("stTextArea").filter(visible=True)
    textarea.wait_for(timeout=5000)
    inner = textarea.locator("textarea").first
    inner.fill(question)
    visible_button(session_page, "Ask").first.click()


def test_live_rag_query_and_feedback(session_page):
    """Full pipeline: ask a real question, surface any answer state, click feedback."""
    question = "What is Delta Lake in one sentence?"
    ask_question(session_page, question)

    # The turn either answers or surfaces an explicit failure state.
    page = session_page
    try:
        page.get_by_role("heading", name="Answer").filter(visible=True).wait_for(timeout=120_000)
        assert page.get_by_text("Confidence:", exact=False).filter(visible=True).count() >= 1
    except Exception:
        body = page.get_by_test_id("stMainBlockContainer").inner_text()
        assert any(
            marker in body
            for marker in (
                "Failed to get answer",
                "No answer could be generated",
                "Could not connect to the RAG service",
            )
        ), f"Ask turn neither answered nor failed explicitly: {body[:400]}"

    # Feedback buttons only render on the same script run as the answer; if the
    # query failed they are absent — accept either, the UI must stay alive.
    helpful = visible_button(page, "👍 Helpful")
    not_helpful = visible_button(page, "👎 Not Helpful")
    if helpful.count() == 1:
        helpful.first.click()
        page.wait_for_timeout(1200)
        assert not page.get_by_test_id("stAlertContentError").count()
    if not_helpful.count() == 1:
        not_helpful.first.click()
        page.wait_for_timeout(1200)
        assert not page.get_by_test_id("stAlertContentError").count()


def test_reset_metrics_after_query(session_page):
    """After the recorded query the Session Summary renders and Reset works.

    Lives here (not in ``test_metrics_tab.py``) because it must run in the same
    module as ``test_live_rag_query_and_feedback`` — both share the module-scoped
    ``session_page`` so the collector recorded by the live query is visible to
    the metrics tab. ``test_metrics_tab.py`` covers the fresh-session state.
    """
    open_tab(session_page, Tab.METRICS)
    main = session_page.get_by_test_id("stMainBlockContainer")
    main.get_by_text("RAG Service Metrics", exact=True).filter(visible=True).wait_for(timeout=10000)
    summary = main.get_by_text("Session Summary", exact=True).filter(visible=True)
    if summary.count() == 0:
        pytest.skip("live query did not record metrics (provider degraded)")
        return

    reset = session_page.get_by_role("button", name="Reset Metrics", exact=True).filter(visible=True)
    reset.first.wait_for(timeout=10000)
    reset.first.click()
    session_page.wait_for_timeout(2500)
    session_page.get_by_test_id("stAlertContentInfo").filter(visible=True).first.wait_for(timeout=15000)
    assert (
        "No queries recorded yet"
        in session_page.get_by_test_id("stAlertContentInfo").filter(visible=True).first.inner_text()
    )
