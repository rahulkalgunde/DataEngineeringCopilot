"""Markdown/ASCII report generator for profiling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_engineering_copilot.profiler.concurrency_tuner import StageRecommendation


class ReportGenerator:
    """Generates human-readable performance reports from telemetry data."""

    def generate_markdown(
        self,
        summary: dict[str, Any],
        recommendations: list[StageRecommendation],
        production_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Generate a Markdown report with stage-level metrics and recommendations."""
        md: list[str] = []
        md.append("# Ingestion Pipeline Resource Trace & Concurrency Tuning Report")
        md.append("")
        md.append(f"**Execution Date**: {summary.get('timestamp', 'N/A')}")
        md.append(f"**Total Execution Time**: {summary.get('total_duration_sec', 0):.2f}s")
        md.append(f"**Peak CPU Usage**: {summary.get('peak_cpu_pct', 0):.1f}%")
        md.append(f"**Avg CPU Usage**: {summary.get('avg_cpu_pct', 0):.1f}%")
        md.append(f"**Peak RSS Memory**: {summary.get('peak_memory_mb', 0):.2f} MB")
        md.append("")

        if production_metrics is not None:
            md.append("## Production Ingestion Metrics (from Celery/Redis)")
            md.append("")
            md.append(f"- **Status**: {production_metrics.get('status', 'N/A')}")
            md.append(f"- **Pages Fetched**: {production_metrics.get('pages_fetched', 0)}")
            md.append(f"- **Chunks Indexed**: {production_metrics.get('chunks_indexed', 0)}")
            md.append(f"- **Pages Skipped**: {production_metrics.get('pages_skipped', 0)}")
            err = production_metrics.get("errors")
            md.append(f"- **Errors**: {err if err else 'none'}")
            md.append("")

        stages = summary.get("stages", {})
        if stages:
            total_items = sum(s.get("items_processed", 0) for s in stages.values())
            md.append(f"**Total Items Processed**: {total_items}")
            md.append("")

            md.append("## Per-Stage Performance")
            md.append("| Stage | Items | Avg Lat (s) | p50 (s) | p90 (s) | p99 (s) | Throughput/s | Errors |")
            md.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for name, s in sorted(stages.items()):
                md.append(
                    f"| `{name}` | {s['items_processed']} | {s['avg_latency']} | "
                    f"{s['p50']} | {s['p90']} | {s['p99']} | {s['throughput_per_sec']} | {s['errors']} |"
                )
            md.append("")

        if recommendations:
            md.append("## Concurrency Scaling Recommendations")
            md.append("| Stage | Current Workers | Recommended | Action | Bottleneck Reason |")
            md.append("| :--- | ---: | ---: | :--- | :--- |")
            for r in recommendations:
                action_icon = self._action_icon(r.action)
                md.append(
                    f"| `{r.stage_name}` | {r.current_concurrency} | "
                    f"**{r.recommended_concurrency}** | {action_icon} | {r.bottleneck_reason} |"
                )

        return "\n".join(md)

    def generate_json(
        self,
        summary: dict[str, Any],
        recommendations: list[StageRecommendation],
        production_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON-serializable output."""
        return {
            "summary": summary,
            "production_metrics": production_metrics,
            "recommendations": [
                {
                    "stage_name": r.stage_name,
                    "current_concurrency": r.current_concurrency,
                    "recommended_concurrency": r.recommended_concurrency,
                    "action": r.action,
                    "bottleneck_reason": r.bottleneck_reason,
                    "max_throughput_achieved": r.max_throughput_achieved,
                    "p99_latency": r.p99_latency,
                }
                for r in recommendations
            ],
        }

    def save_report(
        self,
        summary: dict[str, Any],
        recommendations: list[StageRecommendation],
        production_metrics: dict[str, Any] | None = None,
        output_dir: str = "./profiler_reports",
        name: str = "telemetry_report",
    ) -> Path:
        """Save both Markdown and JSON reports to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        md = self.generate_markdown(summary, recommendations, production_metrics)
        md_path = out / f"{name}.md"
        md_path.write_text(md)

        data = self.generate_json(summary, recommendations, production_metrics)
        json_path = out / f"{name}.json"
        json_path.write_text(json.dumps(data, indent=2))

        return out

    @staticmethod
    def _action_icon(action: str) -> str:
        icons = {
            "SCALE_UP": "SCALE UP",
            "SCALE_DOWN": "SCALE DOWN",
            "OPTIMAL": "OPTIMAL",
            "RATE_LIMITED": "RATE LIMITED",
        }
        return icons.get(action, action)
