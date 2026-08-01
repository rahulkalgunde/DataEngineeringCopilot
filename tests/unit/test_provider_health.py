from __future__ import annotations

import time

from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
from data_engineering_copilot.infrastructure.provider_health import (
    ModelHealth,
    ProviderHealth,
    ProviderHealthRegistry,
)


def test_model_health_defaults():
    mh = ModelHealth(provider="openrouter", model="openrouter/free")
    assert mh.consecutive_failures == 0
    assert mh.total_success == 0
    assert mh.total_failures == 0
    assert mh.avg_latency == 0.0
    assert mh.success_rate == 1.0
    assert mh.is_available is True


def test_model_health_success_rate():
    mh = ModelHealth(provider="openrouter", model="openrouter/free", total_success=8, total_failures=2)
    assert mh.success_rate == 0.8


def test_model_health_avg_latency():
    mh = ModelHealth(
        provider="openrouter", model="openrouter/free", total_success=5, total_failures=0, total_latency=10.0
    )
    assert mh.avg_latency == 2.0


def test_model_health_not_available_during_cooldown():
    mh = ModelHealth(provider="openrouter", model="openrouter/free")
    mh.cooldown_until = time.monotonic() + 999
    assert mh.is_available is False


def test_provider_health_defaults():
    ph = ProviderHealth(provider="openrouter")
    assert ph.is_available is True
    assert ph.models == {}


def test_provider_health_not_available_during_cooldown():
    ph = ProviderHealth(provider="openrouter")
    ph.cooldown_until = time.monotonic() + 999
    assert ph.is_available is False


def test_provider_health_aggregate_success_rate():
    ph = ProviderHealth(provider="openrouter")
    ph.models["a"] = ModelHealth(provider="openrouter", model="a", total_success=9, total_failures=1)
    ph.models["b"] = ModelHealth(provider="openrouter", model="b", total_success=1, total_failures=9)
    assert ph.aggregate_success_rate == 0.5


def test_register_provider():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free", "openrouter/smart"])
    ph = reg.get_provider_health("openrouter")
    assert ph is not None
    assert "openrouter/free" in ph.models
    assert "openrouter/smart" in ph.models


def test_register_provider_twice_is_idempotent():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.register_provider("openrouter", ["openrouter/smart"])
    ph = reg.get_provider_health("openrouter")
    assert "openrouter/free" in ph.models
    assert "openrouter/smart" in ph.models


def test_track_success():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.track_success("openrouter", "openrouter/free", 2.5)
    mh = reg.get_model_health("openrouter", "openrouter/free")
    assert mh.total_success == 1
    assert mh.consecutive_failures == 0
    assert mh.last_error_category is None
    assert mh.cooldown_until == 0.0


def test_track_success_resets_consecutive_failures():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.track_failure("openrouter", "openrouter/free", ProviderErrorCategory.TEMPORARY_UNAVAILABLE)
    reg.track_success("openrouter", "openrouter/free", 1.0)
    mh = reg.get_model_health("openrouter", "openrouter/free")
    assert mh.consecutive_failures == 0


def test_track_failure_temporary_unavailable_sets_cooldown():
    reg = ProviderHealthRegistry(default_cooldown_seconds=30.0)
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.track_failure("openrouter", "openrouter/free", ProviderErrorCategory.TEMPORARY_UNAVAILABLE)
    mh = reg.get_model_health("openrouter", "openrouter/free")
    assert mh.total_failures == 1
    assert mh.consecutive_failures == 1
    assert mh.last_error_category == ProviderErrorCategory.TEMPORARY_UNAVAILABLE
    assert mh.cooldown_until > time.monotonic()


def test_track_failure_rate_limited_sets_cooldown():
    reg = ProviderHealthRegistry(default_cooldown_seconds=30.0)
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.track_failure("openrouter", "openrouter/free", ProviderErrorCategory.RATE_LIMITED, retry_after=10.0)
    mh = reg.get_model_health("openrouter", "openrouter/free")
    assert mh.cooldown_until > time.monotonic()


def test_track_failure_rate_limited_without_retry_after_uses_category_default():
    reg = ProviderHealthRegistry(default_cooldown_seconds=30.0)
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.track_failure("openrouter", "openrouter/free", ProviderErrorCategory.RATE_LIMITED)
    mh = reg.get_model_health("openrouter", "openrouter/free")
    assert mh.cooldown_until > time.monotonic() + 50.0


def test_track_failure_retryable_uses_short_cooldown():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.track_failure("openrouter", "openrouter/free", ProviderErrorCategory.RETRYABLE)
    mh = reg.get_model_health("openrouter", "openrouter/free")
    assert 0 < mh.cooldown_until - time.monotonic() <= 10.0


def test_track_failure_permanent_error_sets_cooldown():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.track_failure("openrouter", "openrouter/free", ProviderErrorCategory.PERMANENT_ERROR)
    mh = reg.get_model_health("openrouter", "openrouter/free")
    assert mh.cooldown_until > time.monotonic()


def test_get_healthy_providers_returns_registered():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.register_provider("nvidia", ["nvidia/model"])
    healthy = reg.get_healthy_providers()
    assert "openrouter" in healthy
    assert "nvidia" in healthy


def test_get_healthy_providers_excludes_cooldowned():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.register_provider("nvidia", ["nvidia/model"])
    reg.mark_provider_cooldown("openrouter", duration=9999)
    healthy = reg.get_healthy_providers()
    assert "openrouter" not in healthy
    assert "nvidia" in healthy


def test_get_healthy_providers_excludes_explicit():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["openrouter/free"])
    reg.register_provider("nvidia", ["nvidia/model"])
    healthy = reg.get_healthy_providers(exclude=["openrouter"])
    assert "openrouter" not in healthy
    assert "nvidia" in healthy


def test_get_healthy_models_returns_sorted_by_score():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["model-a", "model-b"])
    reg.track_success("openrouter", "model-a", 1.0)
    reg.track_failure("openrouter", "model-b", ProviderErrorCategory.PERMANENT_ERROR)
    models = reg.get_healthy_models("openrouter")
    names = [m for m, _ in models]
    assert names[0] == "model-a"


def test_get_healthy_models_excludes_cooldowned():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["model-a", "model-b"])
    reg.mark_model_cooldown("openrouter", "model-a", duration=9999)
    models = reg.get_healthy_models("openrouter")
    names = [m for m, _ in models]
    assert "model-a" not in names
    assert "model-b" in names


def test_mark_provider_cooldown_sets_all_models():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["model-a", "model-b"])
    reg.mark_provider_cooldown("openrouter", duration=9999)
    for name in ("model-a", "model-b"):
        mh = reg.get_model_health("openrouter", name)
        assert mh.cooldown_until > time.monotonic()


def test_provider_is_healthy():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["model-a"])
    assert reg.provider_is_healthy("openrouter") is True
    reg.mark_provider_cooldown("openrouter", duration=9999)
    assert reg.provider_is_healthy("openrouter") is False


def test_get_least_recently_selected():
    reg = ProviderHealthRegistry()
    reg.register_provider("a", ["m1"])
    reg.register_provider("b", ["m2"])
    reg.register_provider("c", ["m3"])
    reg.mark_selected("a")
    reg.mark_selected("b")
    chosen = reg.get_least_recently_selected(["a", "b", "c"])
    assert chosen == "c"


def test_get_least_recently_selected_empty():
    reg = ProviderHealthRegistry()
    assert reg.get_least_recently_selected([]) is None


def test_all_providers():
    reg = ProviderHealthRegistry()
    reg.register_provider("openrouter", ["model-a"])
    reg.register_provider("nvidia", ["model-b"])
    all_p = reg.all_providers
    assert "openrouter" in all_p
    assert "nvidia" in all_p
    assert len(all_p) == 2


def test_get_provider_health_unknown():
    reg = ProviderHealthRegistry()
    assert reg.get_provider_health("unknown") is None


def test_get_model_health_unknown():
    reg = ProviderHealthRegistry()
    assert reg.get_model_health("unknown", "unknown") is None
