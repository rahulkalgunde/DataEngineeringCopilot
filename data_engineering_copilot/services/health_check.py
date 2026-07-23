"""Health check registry."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    overall: str  # "healthy", "degraded", "unhealthy"
    services: dict[str, bool] = field(default_factory=dict)


class HealthChecker:
    """Register service health checks and get aggregate status."""

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], bool]] = {}

    def register(self, name: str, check_fn: Callable[[], bool]) -> None:
        self._checks[name] = check_fn

    def check(self) -> HealthStatus:
        results: dict[str, bool] = {}
        for name, fn in self._checks.items():
            try:
                results[name] = bool(fn())
            except Exception as exc:
                logger.warning("Health check failed for %s: %s", name, exc)
                results[name] = False

        if not results:
            return HealthStatus(overall="healthy", services={})

        all_healthy = all(results.values())
        all_unhealthy = not any(results.values())

        if all_healthy:
            overall = "healthy"
        elif all_unhealthy:
            overall = "unhealthy"
        else:
            overall = "degraded"

        return HealthStatus(overall=overall, services=results)
