"""Tests for Langfuse score-config seeding/reconciliation (fake clients only).

Never touches real Langfuse: get_langfuse_instance is monkeypatched to return
fake API doubles, pinning pagination, idempotency, suffix handling, drift
reconciliation, and never-raise semantics of get_score_config_id.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from data_engineering_copilot.evaluation import langfuse_score_configs as lsc

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_id_cache():
    lsc._CONFIG_ID_CACHE.clear()
    yield
    lsc._CONFIG_ID_CACHE.clear()


def _page(names: list[str], total_items: int | None = None) -> SimpleNamespace:
    items = [SimpleNamespace(name=n, id=f"id-{n}") for n in names]
    meta = SimpleNamespace(total_items=total_items) if total_items is not None else None
    return SimpleNamespace(data=items, meta=meta)


class TestExistingConfigsPagination:
    def test_single_partial_page(self):
        client = MagicMock()
        client.get.side_effect = [_page(["a", "b"])]
        assert lsc._existing_configs(client) == {"a", "b"}
        client.get.assert_called_once_with(page=1, limit=100)

    def test_full_pages_continue_until_short_page(self):
        client = MagicMock()
        full = _page([f"n{i}" for i in range(100)])
        short = _page(["tail"])
        client.get.side_effect = [full, short]
        names = lsc._existing_configs(client)
        assert len(names) == 101
        assert client.get.call_count == 2

    def test_total_items_stops_paging_early(self):
        client = MagicMock()
        exact = _page([f"n{i}" for i in range(100)], total_items=100)
        client.get.side_effect = [exact]
        assert len(lsc._existing_configs(client)) == 100
        client.get.assert_called_once()

    def test_empty_result(self):
        client = MagicMock()
        client.get.side_effect = [_page([])]
        assert lsc._existing_configs(client) == set()


class TestGetScoreConfigId:
    def test_none_when_client_unavailable(self, monkeypatch):
        monkeypatch.setattr(lsc, "get_langfuse_instance", lambda: None)
        assert lsc.get_score_config_id("confidence") is None

    def test_resolves_across_pages_and_caches(self, monkeypatch):
        client = MagicMock()
        full_page = _page([f"filler{i}" for i in range(100)])  # full page -> keep paging
        client._client.api.score_configs.get.side_effect = [
            full_page,
            _page(["confidence"]),
        ]
        monkeypatch.setattr(lsc, "get_langfuse_instance", lambda: client)
        assert lsc.get_score_config_id("confidence") == "id-confidence"
        # second call served from cache: no additional client paging
        assert lsc.get_score_config_id("confidence") == "id-confidence"
        assert client._client.api.score_configs.get.call_count == 2

    def test_missing_name_caches_negative_result(self, monkeypatch):
        client = MagicMock()
        # single short page without the target: loop stops (<100 items), None cached
        client._client.api.score_configs.get.side_effect = [_page(["a"])]
        monkeypatch.setattr(lsc, "get_langfuse_instance", lambda: client)
        assert lsc.get_score_config_id("absent") is None
        assert lsc.get_score_config_id("absent") is None
        assert client._client.api.score_configs.get.call_count == 1

    def test_api_exception_returns_none_not_raise(self, monkeypatch):
        client = MagicMock()
        client._client.api.score_configs.get.side_effect = RuntimeError("boom")
        monkeypatch.setattr(lsc, "get_langfuse_instance", lambda: client)
        assert lsc.get_score_config_id("confidence") is None


class TestSeedScoreConfigs:
    def _client_with(self, existing: list[str]):
        # NOTE: seed_score_configs resolves `score_configs = client._client.api.score_configs`
        # and calls .get/.create/.update ON that object — so the fake methods
        # live directly on api, not on api.score_configs.
        client = MagicMock()
        api = MagicMock()
        api.get.side_effect = lambda page=1, limit=100: _page(existing) if page == 1 else _page([])
        client._client.api.score_configs = api
        return client, api

    def test_raises_without_client(self, monkeypatch):
        monkeypatch.setattr(lsc, "get_langfuse_instance", lambda: None)
        with pytest.raises(RuntimeError, match="Langfuse is unavailable"):
            lsc.seed_score_configs()

    def test_creates_missing_and_skips_existing(self, monkeypatch):
        client, api = self._client_with(existing=["confidence"])
        monkeypatch.setattr(lsc, "get_langfuse_instance", lambda: client)
        created = lsc.seed_score_configs()
        assert created["confidence"] is False
        assert created["groundedness"] is True
        created_names = {c.kwargs["name"] for c in api.create.call_args_list}
        assert "confidence" not in created_names
        assert "groundedness" in created_names

    def test_description_suffix_appended(self, monkeypatch):
        client, api = self._client_with(existing=[])
        monkeypatch.setattr(lsc, "get_langfuse_instance", lambda: client)
        lsc.seed_score_configs(description_suffix="itest")
        call = next(c for c in api.create.call_args_list if c.kwargs["name"] == "faithfulness")
        assert call.kwargs["description"].endswith("(itest)")

    def test_create_failure_recorded_as_false_not_raised(self, monkeypatch):
        client, api = self._client_with(existing=[])
        api.create.side_effect = RuntimeError("api down")
        monkeypatch.setattr(lsc, "get_langfuse_instance", lambda: client)
        created = lsc.seed_score_configs()
        assert set(created.values()) == {False}


class TestReconcileExistingConfig:
    def test_non_categorical_returns_without_calls(self):
        api = MagicMock()
        lsc._reconcile_existing_config(api, "confidence", "NUMERIC", {"min_value": 0.0})
        api.get.assert_not_called()

    def test_matching_labels_no_update(self):
        from data_engineering_copilot.evaluation.langfuse_score_configs import SCORE_CONFIGS

        expected = SCORE_CONFIGS["intent"][1]["categories"]
        item = SimpleNamespace(
            name="intent",
            id="id-intent",
            categories=[SimpleNamespace(label=c["label"]) for c in expected],
        )
        api = MagicMock()
        api.get.return_value = SimpleNamespace(data=[item])
        lsc._reconcile_existing_config(api, "intent", "CATEGORICAL", {"categories": expected})
        api.update.assert_not_called()

    def test_drifted_labels_trigger_update(self):
        from data_engineering_copilot.evaluation.langfuse_score_configs import SCORE_CONFIGS

        expected = SCORE_CONFIGS["intent"][1]["categories"]
        item = SimpleNamespace(name="intent", id="id-intent", categories=[SimpleNamespace(label="stale")])
        api = MagicMock()
        api.get.return_value = SimpleNamespace(data=[item])
        lsc._reconcile_existing_config(api, "intent", "CATEGORICAL", {"categories": expected})
        api.update.assert_called_once()
        args, kwargs = api.update.call_args
        assert args[0] == "id-intent"
        assert {c["label"] for c in kwargs["categories"]} == {c["label"] for c in expected}

    def test_exception_swallowed(self):
        api = MagicMock()
        api.get.side_effect = RuntimeError("boom")
        lsc._reconcile_existing_config(api, "intent", "CATEGORICAL", {"categories": [{"value": 0.0, "label": "x"}]})


class TestCatalogIntegrity:
    """Pin the catalog shape so UI/pipeline consumers cannot silently break."""

    def test_all_numeric_configs_bounded_0_1(self):
        for name, (data_type, extra) in lsc.SCORE_CONFIGS.items():
            if data_type == "NUMERIC":
                assert extra["min_value"] == 0.0 and extra["max_value"] == 1.0, name

    def test_intent_categories_cover_pipeline_intents(self):
        labels = {c["label"] for c in lsc.SCORE_CONFIGS["intent"][1]["categories"]}
        assert {"factual", "code_example", "api_lookup", "how_to"} <= labels

    def test_names_are_unique_and_snake_case(self):
        for name in lsc.SCORE_CONFIGS:
            assert name.replace("_", "").isalnum(), name
