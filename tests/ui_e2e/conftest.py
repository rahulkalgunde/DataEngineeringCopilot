"""Browser-based Streamlit UI E2E tests.

These tests drive a *live* Streamlit app (``make streamlit``, default
``http://localhost:8501``) with Playwright. They are marked ``ui`` and skip
cleanly when either the app or the Playwright chromium browser is unavailable,
so CI (hermetic, unit-only) is unaffected.

Design notes
------------
- A single session-scoped chromium browser is shared across the module; each
  test gets a fresh page against the live app (fresh Streamlit websocket =
  fresh session state).
- ``session_page`` is module-scoped so an end-to-end session (ask → metrics →
  reset) can persist across tests in the same file.
- Screenshots of failures are saved to ``tests/ui_e2e/artifacts/``.
- Live-LLM assertions are intentionally tolerant: they verify the UI *wired
  and reacted* (answer, fallback answer, or explicit error rendered), not that
  a specific free-tier provider returned a specific string. Zero-cost signals
  (button disabled/enabled contracts, guard errors, in-process lab runs) are
  asserted strictly.
"""

from __future__ import annotations

import pathlib
import urllib.request
from typing import NoReturn

import pytest

UI_BASE_URL = "http://localhost:8501"

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parent / "artifacts"

pytestmark = pytest.mark.ui


def ui_unavailable(reason: str) -> NoReturn:
    """Skip cleanly when the app/browser are not present (ui tests never fail CI)."""
    pytest.skip(reason)


def _app_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/_stcore/health", timeout=3) as resp:
            return resp.read().decode().strip() == "ok"
    except Exception:
        return False


def _browser_installed() -> bool:
    import os

    cache = pathlib.Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "~/.cache/ms-playwright")).expanduser()
    return (cache / "chromium-1228").exists() or (cache / "chromium_headless_shell-1228").exists()


@pytest.fixture(scope="module")
def app_url() -> str:
    url = UI_BASE_URL
    if not _app_reachable(url):
        ui_unavailable(f"Streamlit app not reachable at {url} (run `make streamlit`)")
    return url


@pytest.fixture(scope="module")
def browser():
    if not _browser_installed():
        ui_unavailable("Playwright chromium not installed (run `playwright install chromium`)")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        ui_unavailable("playwright package not installed")

    with sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception as exc:  # system deps missing
            ui_unavailable(f"chromium launch failed: {exc}")
        yield b
        b.close()


def _open_app(browser, app_url):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    heading = page.get_by_role("heading", name="DataEngineeringCopilot")
    # The single Streamlit dev server can stall ~30-60s right after a live-LLM
    # test (ask/chat/lab). Retry the load instead of failing on one slow reply.
    for attempt in range(3):
        page.goto(app_url, wait_until="domcontentloaded")
        try:
            heading.wait_for(timeout=25000)
            page.wait_for_timeout(1200)  # first rerun + sidebar health settle
            return page
        except Exception:
            if attempt == 2:
                raise
            page.wait_for_timeout(2000)
    return page


@pytest.fixture()
def page(browser, app_url):
    page = _open_app(browser, app_url)
    yield page
    page.close()


@pytest.fixture(scope="module")
def session_page(browser, app_url):
    """Module-scoped page: keeps one Streamlit websocket session across tests."""
    page = _open_app(browser, app_url)
    yield page
    page.close()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):  # pragma: no cover - side-effect hook
    """Save a full-page screenshot when a ui test fails."""
    if call.when == "call" and call.excinfo is not None:
        page = None
        for fixture_name in ("session_page", "page"):
            try:
                page = item.funcargs.get(fixture_name)
            except Exception:
                page = None
            if page is not None:
                break
        if page is not None and not page.is_closed():
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            try:
                path = ARTIFACT_DIR / f"{item.name}.png"
                page.screenshot(path=str(path), full_page=True)
                print(f"\n[ui_e2e] screenshot saved: {path}")
            except Exception:
                pass


class Tab:
    """Stable tab selectors for the six Streamlit tabs."""

    CHAT = "💬 Chat"
    ASK = "💬 Ask"
    INGEST = "📥 Ingestion"
    LAB = "🧪 Pipeline Lab"
    HEALTH = "🔧 System Health"
    METRICS = "📊 Metrics"


def open_tab(page, name: str, ready=None) -> None:
    """Click *name* tab and wait for the rerun to settle.

    Pass *ready* (a locator bound to the target tab's content) to retry the
    click until that content renders: the single-threaded Streamlit server can
    stall for tens of seconds during LLM/health work, so a click can be dropped
    or the rerun delayed. ``ready`` filters to visible elements because Streamlit
    keeps other tabs' content in the DOM (hidden, not removed).
    """
    tab = page.get_by_role("tab", name=name, exact=True)
    if ready is None:
        tab.click()
        page.wait_for_timeout(1500)
        return
    for _ in range(4):
        tab.click()
        try:
            ready.filter(visible=True).wait_for(timeout=6000)
            return
        except Exception:
            continue
    ready.filter(visible=True).wait_for(timeout=15000)


def visible_button(page, label: str):
    """Return the *visible* button locator matching *label* (tabs hide content)."""
    return page.get_by_role("button", name=label, exact=True).filter(visible=True)


def visible_text(page, text: str):
    """Return *visible* text locator matching *text*."""
    return page.get_by_text(text, exact=False).filter(visible=True)
