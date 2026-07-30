from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

from data_engineering_copilot.domain.exceptions import ProviderErrorCategory

logger = logging.getLogger(__name__)


@dataclass
class ModelHealth:
    provider: str
    model: str
    consecutive_failures: int = 0
    total_success: int = 0
    total_failures: int = 0
    total_latency: float = 0.0
    last_used_at: float = 0.0
    last_success_at: float = 0.0
    last_failure_at: float = 0.0
    last_error_category: ProviderErrorCategory | None = None
    cooldown_until: float = 0.0

    @property
    def avg_latency(self) -> float:
        total_calls = self.total_success + self.total_failures
        if total_calls == 0:
            return 0.0
        return self.total_latency / total_calls

    @property
    def success_rate(self) -> float:
        total = self.total_success + self.total_failures
        if total == 0:
            return 1.0
        return self.total_success / total

    @property
    def is_available(self) -> bool:
        return time.monotonic() >= self.cooldown_until


@dataclass
class ProviderHealth:
    provider: str
    models: dict[str, ModelHealth] = field(default_factory=dict)
    cooldown_until: float = 0.0

    @property
    def is_available(self) -> bool:
        return time.monotonic() >= self.cooldown_until

    @property
    def aggregate_success_rate(self) -> float:
        total_success = sum(m.total_success for m in self.models.values())
        total_failures = sum(m.total_failures for m in self.models.values())
        total = total_success + total_failures
        if total == 0:
            return 1.0
        return total_success / total


class ProviderHealthRegistry:
    def __init__(
        self,
        success_rate_weight: float = 0.6,
        latency_weight: float = 0.2,
        recency_weight: float = 0.2,
        consecutive_failure_penalty: float = 0.3,
        default_cooldown_seconds: float = 60.0,
    ) -> None:
        self._providers: dict[str, ProviderHealth] = {}
        self._lock = Lock()
        self.success_rate_weight = success_rate_weight
        self.latency_weight = latency_weight
        self.recency_weight = recency_weight
        self.consecutive_failure_penalty = consecutive_failure_penalty
        self.default_cooldown_seconds = default_cooldown_seconds
        self._last_selected: dict[str, float] = {}

    def register_provider(self, provider: str, models: list[str]) -> None:
        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = ProviderHealth(provider=provider)
            ph = self._providers[provider]
            for model in models:
                if model not in ph.models:
                    ph.models[model] = ModelHealth(provider=provider, model=model)

    def track_success(self, provider: str, model: str, latency: float) -> None:
        with self._lock:
            ph = self._providers.get(provider)
            if ph is None:
                return
            mh = ph.models.get(model)
            if mh is None:
                return
            mh.consecutive_failures = 0
            mh.total_success += 1
            mh.total_latency += latency
            mh.last_used_at = time.monotonic()
            mh.last_success_at = time.monotonic()
            mh.last_error_category = None
            mh.cooldown_until = 0.0

    def track_failure(
        self,
        provider: str,
        model: str,
        category: ProviderErrorCategory,
        retry_after: float | None = None,
    ) -> None:
        with self._lock:
            ph = self._providers.get(provider)
            if ph is None:
                return
            mh = ph.models.get(model)
            if mh is None:
                return
            mh.consecutive_failures += 1
            mh.total_failures += 1
            mh.last_failure_at = time.monotonic()
            mh.last_used_at = time.monotonic()
            mh.last_error_category = category

            if category in (
                ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
                ProviderErrorCategory.RATE_LIMITED,
            ):
                cooldown = retry_after if retry_after else self.default_cooldown_seconds
                mh.cooldown_until = time.monotonic() + cooldown

    def mark_provider_cooldown(self, provider: str, duration: float | None = None) -> None:
        with self._lock:
            ph = self._providers.get(provider)
            if ph is None:
                return
            duration = duration or self.default_cooldown_seconds
            ph.cooldown_until = time.monotonic() + duration
            for mh in ph.models.values():
                model_cooldown = max(mh.cooldown_until, ph.cooldown_until)
                mh.cooldown_until = model_cooldown

    def mark_model_cooldown(self, provider: str, model: str, duration: float | None = None) -> None:
        with self._lock:
            ph = self._providers.get(provider)
            if ph is None:
                return
            mh = ph.models.get(model)
            if mh is None:
                return
            duration = duration or self.default_cooldown_seconds
            mh.cooldown_until = time.monotonic() + duration

    def _health_score(self, mh: ModelHealth) -> float:
        score = self.success_rate_weight * mh.success_rate
        if mh.avg_latency > 0:
            normalized_latency = 1.0 / (1.0 + mh.avg_latency)
            score += self.latency_weight * normalized_latency
        recency = 0.0
        if mh.last_success_at > 0:
            time_since_success = time.monotonic() - mh.last_success_at
            recency = max(0.0, 1.0 - time_since_success / 300.0)
        score += self.recency_weight * recency
        score -= self.consecutive_failure_penalty * mh.consecutive_failures
        return max(0.0, score)

    def get_healthy_providers(self, exclude: list[str] | None = None) -> list[str]:
        exclude = exclude or []
        with self._lock:
            result = []
            for provider, ph in self._providers.items():
                if provider in exclude:
                    continue
                if not ph.is_available:
                    continue
                available_models = [m for m in ph.models.values() if m.is_available]
                if not available_models:
                    continue
                result.append(provider)
            return result

    def get_healthy_models(self, provider: str) -> list[tuple[str, float]]:
        with self._lock:
            ph = self._providers.get(provider)
            if ph is None:
                return []
            scored = []
            for model, mh in ph.models.items():
                if mh.is_available:
                    scored.append((model, self._health_score(mh)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

    def provider_is_healthy(self, provider: str) -> bool:
        return self.get_healthy_providers().count(provider) > 0

    def mark_selected(self, provider: str) -> None:
        with self._lock:
            self._last_selected[provider] = time.monotonic()

    def get_least_recently_selected(self, providers: list[str]) -> str | None:
        if not providers:
            return None
        with self._lock:
            return min(
                providers,
                key=lambda p: self._last_selected.get(p, 0.0),
            )

    def get_provider_health(self, provider: str) -> ProviderHealth | None:
        with self._lock:
            return self._providers.get(provider)

    def get_model_health(self, provider: str, model: str) -> ModelHealth | None:
        with self._lock:
            ph = self._providers.get(provider)
            if ph is None:
                return None
            return ph.models.get(model)

    @property
    def all_providers(self) -> dict[str, ProviderHealth]:
        with self._lock:
            return dict(self._providers)
