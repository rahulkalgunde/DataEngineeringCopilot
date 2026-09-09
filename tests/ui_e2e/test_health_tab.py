"""System Health tab E2E: status columns, vector store, config, expander."""

import pytest

from .conftest import Tab, open_tab

pytestmark = pytest.mark.ui


def test_health_tab_sections(page):
    open_tab(page, Tab.HEALTH, page.get_by_role("heading", name="System Health"))
    main = page.get_by_test_id("stMainBlockContainer")
    for heading in ("System Health", "Service Status", "Vector Store", "Ollama Configuration", "Ingestion History"):
        main.get_by_text(heading, exact=True).filter(visible=True).wait_for(timeout=5000)


def test_service_status_columns(page):
    open_tab(page, Tab.HEALTH, page.get_by_role("heading", name="System Health"))
    main = page.get_by_test_id("stMainBlockContainer")
    body = main.inner_text(timeout=5000)
    assert any(s in body for s in ("Qdrant", "Ollama", "Langfuse", "Docker image"))


def test_vector_store_metric_present(page):
    open_tab(page, Tab.HEALTH, page.get_by_role("heading", name="System Health"))
    main = page.get_by_test_id("stMainBlockContainer")
    metric = main.get_by_test_id("stMetric").filter(has_text="Total Chunks Indexed").first
    metric.wait_for(timeout=5000)
    value = metric.locator('[data-testid="stMetricValue"]').inner_text()
    assert value.isdigit() or "unavailable" in value


def test_ollama_config_metrics(page):
    open_tab(page, Tab.HEALTH, page.get_by_role("heading", name="System Health"))
    main = page.get_by_test_id("stMainBlockContainer")
    for label in ("Model", "Embedding Model", "Base URL", "Timeout", "Output Limit"):
        main.get_by_test_id("stMetric").filter(has_text=label).first.wait_for(timeout=5000)


def test_advanced_config_expander_toggles(page):
    open_tab(page, Tab.HEALTH, page.get_by_role("heading", name="System Health"))
    main = page.get_by_test_id("stMainBlockContainer")
    expander = main.get_by_text("Advanced Configuration", exact=True).first
    expander.wait_for(timeout=5000)
    expander.click()
    page.wait_for_timeout(800)
    for label in ("Retrieval Top-K", "Confidence Threshold", "Chunk Strategy"):
        assert main.get_by_test_id("stMetric").filter(has_text=label).first.count() == 1
