"""Bottleneck detection & concurrency math engine."""

from __future__ import annotations

from dataclasses import dataclass

from data_engineering_copilot.profiler.telemetry import StageMetrics


@dataclass
class StageRecommendation:
    """A single stage's concurrency recommendation."""

    stage_name: str
    current_concurrency: int
    recommended_concurrency: int
    action: str  # "SCALE_UP" | "SCALE_DOWN" | "OPTIMAL" | "RATE_LIMITED"
    bottleneck_reason: str
    max_throughput_achieved: float
    p99_latency: float = 0.0


class ConcurrencyTuner:
    """Analyzes telemetry and determines optimal service concurrency.

    Uses Little's Law (L = λW) to compute theoretical optimal concurrency,
    then adjusts based on rate-limit saturation and hardware constraints.
    """

    def __init__(
        self,
        target_cpu_max_pct: float = 80.0,
        max_rate_limit_ratio: float = 0.02,
        scale_up_factor: float = 1.5,
        scale_down_factor: float = 0.6,
    ) -> None:
        self.target_cpu_max = target_cpu_max_pct
        self.max_rate_limit_ratio = max_rate_limit_ratio
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor

    def analyze_stage(
        self,
        stage_name: str,
        current_concurrency: int,
        metrics: StageMetrics,
        peak_cpu_pct: float,
        rate_limit_hits: int,
    ) -> StageRecommendation:
        """Analyze a single stage and produce a concurrency recommendation.

        Decision logic priority:
        1. Rate-limit saturation → SCALE_DOWN
        2. CPU over-saturation → SCALE_DOWN
        3. Under-utilized → SCALE_UP
        4. Otherwise → OPTIMAL
        """
        total_items = max(metrics.items_processed, 1)
        rate_limit_ratio = rate_limit_hits / total_items
        current_cpu = min(peak_cpu_pct, 100.0)

        if rate_limit_ratio > self.max_rate_limit_ratio:
            reduction = max(0.5, 1.0 - rate_limit_ratio)
            rec = max(1, int(current_concurrency * reduction))
            return StageRecommendation(
                stage_name=stage_name,
                current_concurrency=current_concurrency,
                recommended_concurrency=rec,
                action="RATE_LIMITED",
                bottleneck_reason=(
                    f"Provider rate limit exceeded: {rate_limit_hits} HTTP 429s "
                    f"({rate_limit_ratio:.1%} of requests). Reduce concurrency."
                ),
                max_throughput_achieved=metrics.throughput_per_sec,
                p99_latency=metrics.p99,
            )

        if current_cpu > self.target_cpu_max:
            ratio = self.target_cpu_max / current_cpu
            rec = max(1, int(current_concurrency * ratio))
            return StageRecommendation(
                stage_name=stage_name,
                current_concurrency=current_concurrency,
                recommended_concurrency=rec,
                action="SCALE_DOWN",
                bottleneck_reason=(
                    f"CPU at {current_cpu:.1f}% exceeds target {self.target_cpu_max}%. "
                    f"Reduce workers to relieve backpressure."
                ),
                max_throughput_achieved=metrics.throughput_per_sec,
                p99_latency=metrics.p99,
            )

        if current_concurrency > 0 and metrics.throughput_per_sec > 0:
            rec = max(1, int(current_concurrency * self.scale_up_factor))
            if rec > current_concurrency:
                return StageRecommendation(
                    stage_name=stage_name,
                    current_concurrency=current_concurrency,
                    recommended_concurrency=rec,
                    action="SCALE_UP",
                    bottleneck_reason=(
                        f"Stage under-utilized (CPU {current_cpu:.1f}%). Headroom exists to increase concurrency."
                    ),
                    max_throughput_achieved=metrics.throughput_per_sec,
                    p99_latency=metrics.p99,
                )

        return StageRecommendation(
            stage_name=stage_name,
            current_concurrency=current_concurrency,
            recommended_concurrency=current_concurrency,
            action="OPTIMAL",
            bottleneck_reason="Stage operating at balanced concurrency within system limits.",
            max_throughput_achieved=metrics.throughput_per_sec,
            p99_latency=metrics.p99,
        )

    def analyze_all(
        self,
        stage_metrics: dict[str, StageMetrics],
        peak_cpu_pct: float,
        rate_limit_hits: dict[str, int],
        concurrency_map: dict[str, int],
    ) -> list[StageRecommendation]:
        """Analyze all stages and return recommendations."""
        recommendations: list[StageRecommendation] = []
        for stage_name, metrics in stage_metrics.items():
            current_concurrency = concurrency_map.get(stage_name, 1)
            rl_hits = rate_limit_hits.get(stage_name, 0)
            rec = self.analyze_stage(
                stage_name=stage_name,
                current_concurrency=current_concurrency,
                metrics=metrics,
                peak_cpu_pct=peak_cpu_pct,
                rate_limit_hits=rl_hits,
            )
            recommendations.append(rec)
        return recommendations

    @staticmethod
    def compute_optimal_concurrency(
        peak_throughput: float,
        avg_latency: float,
        headroom_factor: float = 1.2,
    ) -> int:
        """Apply Little's Law: L = λ * W, then add headroom."""
        if peak_throughput <= 0 or avg_latency <= 0:
            return 1
        raw = peak_throughput * avg_latency
        return max(1, int(raw * headroom_factor))
