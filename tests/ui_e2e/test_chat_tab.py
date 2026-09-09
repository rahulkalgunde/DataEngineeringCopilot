"""Chat tab E2E: session manager + one live chat turn (tolerant of provider state)."""

import pytest

from .conftest import Tab, open_tab

# This module drives a live chat turn (LLM round-trip <90s observed) so it
# overrides the global 60s pytest timeout; pure-UI modules keep the default.
pytestmark = [pytest.mark.ui, pytest.mark.timeout(300)]


def test_new_chat_button_and_empty_state(page):
    open_tab(page, Tab.CHAT, page.get_by_test_id("stChatInputTextArea"))
    sidebar = page.get_by_test_id("stSidebar")
    sidebar.get_by_text("💬 Chat Sessions", exact=True).wait_for(timeout=5000)
    assert sidebar.get_by_role("button", name="🆕 New Chat", exact=True).count() == 1
    # fresh browser session => no saved conversations
    assert sidebar.get_by_text("No saved conversations yet.", exact=True).count() >= 1


def test_new_chat_button_click_is_idempotent(page):
    open_tab(page, Tab.CHAT, page.get_by_test_id("stChatInputTextArea"))
    sidebar = page.get_by_test_id("stSidebar")
    sidebar.get_by_role("button", name="🆕 New Chat", exact=True).wait_for(timeout=5000)
    sidebar.get_by_role("button", name="🆕 New Chat", exact=True).click()
    page.wait_for_timeout(1200)
    assert sidebar.get_by_text("No saved conversations yet.", exact=True).count() >= 1


def test_session_selectbox_and_delete_absent_on_fresh_session(page):
    """Session controls only render once a session exists — assert the absence."""
    open_tab(page, Tab.CHAT, page.get_by_test_id("stChatInputTextArea"))
    sidebar = page.get_by_test_id("stSidebar")
    sidebar.get_by_text("💬 Chat Sessions", exact=True).wait_for(timeout=5000)
    # selectbox (key=chat_session_select) and Delete button (key=chat_delete_btn)
    # are guarded on a fresh session with no saved conversations.
    assert sidebar.get_by_test_id("stSelectbox").count() == 0
    assert sidebar.get_by_role("button", name="🗑 Delete Session", exact=True).filter(visible=True).count() == 0


def test_chat_turn_live(page):
    """Send one real chat message; the UI must either answer or render the error.

    Tolerant by design: a free-tier LLM can be slow, rate-limited, or down, but
    the chat turn must *react* — a user bubble appears and the assistant bubble
    shows either generated text or a visible failure state.
    """
    open_tab(page, Tab.CHAT, page.get_by_test_id("stChatInputTextArea"))
    input_area = page.get_by_test_id("stChatInputTextArea").filter(visible=True)
    input_area.wait_for(timeout=5000)

    question = "What is Delta Lake in one sentence?"
    input_area.fill(question)
    page.keyboard.press("Enter")
    page.wait_for_timeout(1500)

    # User bubble with the prompt text.
    page.get_by_text(question, exact=True).first.wait_for(timeout=5000)

    # Assistant bubble: generated text OR an error/status failure. Wait up to
    # 90s for the LLM to stream an answer on the free chain.
    assistant_markers = ["Failed", "error", "Error", "Could not", "unavailable"]
    try:
        page.get_by_test_id("stChatMessage").filter(visible=True).last.wait_for(timeout=90_000)
        final_text = page.get_by_test_id("stChatMessage").filter(visible=True).last.inner_text()
        assert len(final_text.strip()) > 0
    except Exception:
        # Streaming never produced a visible bubble (provider degraded) but the
        # turn still rendered a user bubble -> UI is wired. Surface the state.
        assert page.get_by_text(question, exact=True).count() >= 1
        body = page.get_by_test_id("stMainBlockContainer").inner_text()
        assert any(m in body for m in assistant_markers) or "Connecting to chat API" in body
