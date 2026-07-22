from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache
from data_engineering_copilot.infrastructure.crawl_db import CrawlFrontierDB


@pytest.fixture
def mock_frontier():
    f = AsyncMock(spec=CrawlFrontierDB)
    f.hash_url = CrawlFrontierDB.hash_url
    return f


@pytest.fixture
def mock_cache():
    return AsyncMock(spec=CrawlCache)


def _make_crawler(**kwargs):
    defaults = dict(
        frontier=AsyncMock(spec=CrawlFrontierDB),
        cache=AsyncMock(spec=CrawlCache),
        timeout_seconds=5,
        delay_seconds=0.0,
        concurrency=10,
        max_concurrency=40,
        thread_pool_size=4,
        per_domain_concurrency=40,
    )
    defaults.update(kwargs)
    return AsyncDocumentationCrawler(**defaults)


class TestDynamicAllocation:
    def test_single_domain_gets_full_budget(self):
        c = _make_crawler(max_concurrency=40)
        state = c._get_domain_state("https://spark.apache.org/docs")
        assert state.semaphore._value == 40

    def test_two_domains_split_equally(self):
        c = _make_crawler(max_concurrency=40)
        s1 = c._get_domain_state("https://spark.apache.org/docs")
        s2 = c._get_domain_state("https://flink.apache.org/docs")
        assert s1.semaphore._value == 20
        assert s2.semaphore._value == 20

    def test_three_domains_split_equally(self):
        c = _make_crawler(max_concurrency=40)
        c._get_domain_state("https://a.com")
        c._get_domain_state("https://b.com")
        s3 = c._get_domain_state("https://c.com")
        assert s3.semaphore._value == 13

    def test_minimum_one_slot_per_domain(self):
        c = _make_crawler(max_concurrency=2)
        c._get_domain_state("https://a.com")
        c._get_domain_state("https://b.com")
        c._get_domain_state("https://c.com")
        s4 = c._get_domain_state("https://d.com")
        assert s4.semaphore._value >= 1

    def test_same_domain_reuses_state(self):
        c = _make_crawler(max_concurrency=40)
        s1 = c._get_domain_state("https://example.com/a")
        s2 = c._get_domain_state("https://example.com/b")
        assert s1 is s2


class TestDictPriorityAllocation:
    def test_high_priority_gets_double(self):
        c = _make_crawler(
            max_concurrency=40,
            priority_domains={"spark.apache.org": 2},
            priority_multipliers={1: 1.0, 2: 2.0, 3: 4.0},
        )
        state = c._get_domain_state("https://spark.apache.org/docs")
        assert state.semaphore._value == 40

    def test_priority_with_multiple_domains(self):
        c = _make_crawler(
            max_concurrency=40,
            priority_domains={"spark.apache.org": 3},
            priority_multipliers={1: 1.0, 2: 2.0, 3: 4.0},
        )
        c._get_domain_state("https://normal.com")
        s2 = c._get_domain_state("https://spark.apache.org/docs")
        assert s2.semaphore._value == 32

    def test_subdomain_matches_parent(self):
        c = _make_crawler(
            max_concurrency=40,
            priority_domains={"apache.org": 2},
            priority_multipliers={1: 1.0, 2: 2.0},
        )
        state = c._get_domain_state("https://spark.apache.org/docs")
        assert c._get_domain_priority("spark.apache.org") == 2
        assert state.semaphore._value == 40

    def test_unknown_domain_gets_default_priority(self):
        c = _make_crawler(max_concurrency=40)
        assert c._get_domain_priority("unknown.com") == 1

    def test_priority_multiplier_default(self):
        c = _make_crawler(max_concurrency=40)
        assert c._get_priority_multiplier(1) == 1.0
        assert c._get_priority_multiplier(99) == 1.0


class TestSourcePriorityAllocation:
    def test_source_priority_used_for_new_domain(self):
        c = _make_crawler(
            max_concurrency=40,
            priority_multipliers={1: 1.0, 2: 2.0, 3: 4.0},
        )
        state = c._get_domain_state("https://spark.apache.org/docs", source_priority=3)
        assert state.priority == 3
        assert state.semaphore._value == 40

    def test_source_priority_wins_over_dict(self):
        c = _make_crawler(
            max_concurrency=40,
            priority_domains={"spark.apache.org": 1},
            priority_multipliers={1: 1.0, 2: 2.0, 3: 4.0},
        )
        state = c._get_domain_state("https://spark.apache.org/docs", source_priority=3)
        assert state.priority == 3

    def test_dict_priority_used_as_fallback(self):
        c = _make_crawler(
            max_concurrency=40,
            priority_domains={"spark.apache.org": 3},
            priority_multipliers={1: 1.0, 2: 2.0, 3: 4.0},
        )
        state = c._get_domain_state("https://spark.apache.org/docs", source_priority=1)
        assert state.priority == 3

    def test_source_priority_weighted_share(self):
        c = _make_crawler(
            max_concurrency=40,
            priority_multipliers={1: 1.0, 2: 2.0, 3: 4.0},
        )
        c._get_domain_state("https://normal.com", source_priority=1)
        s2 = c._get_domain_state("https://spark.apache.org/docs", source_priority=3)
        assert s2.semaphore._value == 32

    def test_source_priority_not_mutated_after_creation(self):
        c = _make_crawler(
            max_concurrency=40,
            priority_multipliers={1: 1.0, 2: 2.0, 3: 4.0},
        )
        state = c._get_domain_state("https://spark.apache.org/docs", source_priority=2)
        # Second call with different priority should not change existing state
        c._get_domain_state("https://spark.apache.org/docs", source_priority=3)
        assert state.priority == 2


class TestThreadPoolSize:
    def test_thread_pool_uses_configured_size(self):
        c = _make_crawler(thread_pool_size=8)
        assert c._executor._max_workers == 8

    def test_thread_pool_default_is_4(self):
        c = _make_crawler()
        assert c._executor._max_workers == 4

    def test_thread_pool_respects_setting(self):
        from data_engineering_copilot.config.settings import settings

        c = _make_crawler(thread_pool_size=settings.crawl_thread_pool_size)
        assert c._executor._max_workers == settings.crawl_thread_pool_size


class TestPerDomainConcurrencyCap:
    def test_respects_per_domain_cap(self):
        c = _make_crawler(max_concurrency=40, per_domain_concurrency=3)
        state = c._get_domain_state("https://spark.apache.org/docs")
        assert state.semaphore._value == 3

    def test_cap_does_not_affect_when_higher_than_allocated(self):
        c = _make_crawler(max_concurrency=40, per_domain_concurrency=50)
        state = c._get_domain_state("https://spark.apache.org/docs")
        assert state.semaphore._value == 40
