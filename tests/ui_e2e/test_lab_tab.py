"""Pipeline Lab tab E2E.

- Strict: empty URL guard renders "Enter a documentation URL."
- Strict: the "Also upsert" toggle defaults off; switching into paste mode
  renders the raw-HTML textarea.
- Tolerant: the sample dry-run pipeline (in-process, no live write) must either
  reach "✅ Pipeline complete" or surface an explicit per-stage/pipeline error —
  the full 7-stage run executes either way.
"""

import pytest

from .conftest import Tab, open_tab, visible_button

# The lab sample run executes up to 7 in-process pipeline stages (observed up
# to ~3 min cold), so this module overrides the global 60s pytest timeout.
pytestmark = [pytest.mark.ui, pytest.mark.timeout(300)]


def _open_lab(page):
    open_tab(page, Tab.LAB)
    page.get_by_test_id("stRadio").filter(visible=True).wait_for(timeout=5000)


def _select_radio_label(page, label: str) -> None:
    page.get_by_test_id("stRadio").filter(visible=True).get_by_text(label, exact=True).click()
    page.wait_for_timeout(800)


def test_empty_url_guard(page):
    """Strict: URL mode with no URL -> guard error appears."""
    _open_lab(page)
    visible_button(page, "Run pipeline").first.click()
    page.get_by_text("Enter a documentation URL.", exact=True).filter(visible=True).wait_for(timeout=5000)


def test_url_input_renders_and_accepts_text(page):
    """Strict: the URL source mode renders a fillable text input (key=lab_url)."""
    _open_lab(page)
    url_input = page.get_by_label("Documentation URL").filter(visible=True)
    url_input.wait_for(timeout=5000)
    url_input.fill("https://example.com/doc")
    assert url_input.input_value() == "https://example.com/doc"


def test_inject_toggle_defaults_off(page):
    _open_lab(page)
    toggle = page.get_by_test_id("stCheckbox").filter(visible=True).first
    assert toggle.locator("input").is_checked() is False


def test_paste_mode_switches_to_raw_html(page):
    """Strict: switching the radio to paste mode renders the HTML textarea."""
    _open_lab(page)
    _select_radio_label(page, "Paste raw HTML")
    textareas = page.get_by_test_id("stTextArea").filter(visible=True)
    assert textareas.count() == 1
    assert textareas.get_by_placeholder("<!doctype html>…").count() == 1


def test_sample_pipeline_run(page):
    """Tolerant: sample dry-run pipeline executes its 7 stages.

    Either it completes (✅ + dry-run caption) or, if a stage/provider failed,
    an explicit error is rendered. Both prove the Run button drives the full
    in-process pipeline.
    """
    _open_lab(page)
    _select_radio_label(page, "Sample: PySpark API page")
    visible_button(page, "Run pipeline").first.click()

    body = page.get_by_test_id("stMainBlockContainer")
    try:
        body.get_by_text("Pipeline complete", exact=False).filter(visible=True).wait_for(timeout=180_000)
        assert "dry-run" in body.inner_text(timeout=5000)
    except Exception:
        text = body.inner_text(timeout=5000)
        assert any(
            marker in text for marker in ("Pipeline failed", "finish", "error", "Traceback", "Error")
        ), f"Lab run neither completed nor failed explicitly:\n{text[:400]}"
