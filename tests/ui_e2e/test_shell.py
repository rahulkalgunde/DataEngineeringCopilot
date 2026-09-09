"""Shell tests: app loads, all six tabs render, sidebar status surfaces."""

import pytest

from .conftest import Tab, open_tab, visible_button

pytestmark = pytest.mark.ui

ALL_TABS = [Tab.CHAT, Tab.ASK, Tab.INGEST, Tab.LAB, Tab.HEALTH, Tab.METRICS]


def test_app_title_and_tab_bar(page):
    page.get_by_role("heading", name="DataEngineeringCopilot").wait_for(timeout=15000)
    for tab_name in ALL_TABS:
        page.get_by_role("tab", name=tab_name, exact=True).wait_for(timeout=5000)
    tab_badges = [t for t in page.get_by_role("tab").all_inner_texts()]
    assert len(tab_badges) == len(ALL_TABS)


def test_sidebar_service_status(page):
    sidebar = page.get_by_test_id("stSidebar")
    sidebar.get_by_text("System Status", exact=True).wait_for(timeout=5000)
    sidebar.get_by_text("Idle", exact=True).wait_for(timeout=5000)
    # Health indicators render after the health checks complete; auto-wait on
    # the shared prefix then assert one of the up/down variants is present.
    sidebar.get_by_text("Qdrant:", exact=False).first.wait_for(timeout=5000)
    assert (
        sidebar.get_by_text("Qdrant: up", exact=False).count()
        + sidebar.get_by_text("Qdrant: down", exact=False).count()
        >= 1
    )
    sidebar.get_by_text("Ollama:", exact=False).first.wait_for(timeout=5000)
    assert (
        sidebar.get_by_text("Ollama: up", exact=False).count()
        + sidebar.get_by_text("Ollama: down", exact=False).count()
        >= 1
    )
    sidebar.get_by_text("Chunks in Store", exact=True).wait_for(timeout=15000)
    assert sidebar.get_by_text("Chunks in Store", exact=True).count() >= 1


def test_ask_tab_renders_controls(page):
    open_tab(page, Tab.ASK, page.get_by_test_id("stTextArea"))
    page.get_by_test_id("stTextArea").filter(visible=True).wait_for(timeout=5000)
    assert (
        page.get_by_test_id("stTextArea")
        .filter(visible=True)
        .get_by_placeholder("How do I configure Spark dynamic allocation?")
        .count()
        == 1
    )
    assert visible_button(page, "Ask").count() == 1


def test_chat_tab_renders_input(page):
    open_tab(page, Tab.CHAT, page.get_by_test_id("stChatInputTextArea"))
    chat_area = page.get_by_test_id("stChatInputTextArea").filter(visible=True)
    chat_area.wait_for(timeout=5000)
    assert chat_area.first.get_attribute("placeholder") == "Ask a follow-up about Spark, Airflow, Delta Lake…"


def test_ingest_tab_renders_controls(page):
    open_tab(page, Tab.INGEST, page.get_by_test_id("stMultiSelect"))
    page.get_by_test_id("stMultiSelect").filter(visible=True).wait_for(timeout=5000)
    number = page.get_by_test_id("stNumberInput").filter(visible=True)
    number.wait_for(timeout=5000)
    assert number.locator("input").input_value() == "100000"
    assert visible_button(page, "🔄 Start").count() == 1


def test_lab_tab_renders_controls(page):
    open_tab(page, Tab.LAB, page.get_by_test_id("stRadio"))
    page.get_by_test_id("stRadio").filter(visible=True).wait_for(timeout=5000)
    radio_labels = page.get_by_test_id("stRadio").filter(visible=True).locator("label").all_inner_texts()
    joined = " ".join(radio_labels)
    assert "URL" in joined
    assert "Paste raw HTML" in joined
    assert "Sample: PySpark API page" in joined
    assert page.get_by_test_id("stTextInput").filter(visible=True).count() >= 2
    toggle = page.get_by_test_id("stCheckbox").filter(visible=True)
    assert toggle.count() >= 1
    assert visible_button(page, "Run pipeline").count() == 1


def test_tabs_navigate_independently(page):
    """Tab switching must show only the active tab's content."""
    open_tab(page, Tab.HEALTH, page.get_by_role("heading", name="System Health"))
    page.get_by_text("System Health", exact=True).filter(visible=True).wait_for(timeout=5000)
    open_tab(page, Tab.METRICS, page.get_by_text("RAG Service Metrics"))
    page.get_by_text("RAG Service Metrics", exact=True).filter(visible=True).wait_for(timeout=5000)
    open_tab(page, Tab.ASK, page.get_by_test_id("stTextArea"))
    assert visible_button(page, "Ask").count() == 1
