import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class FeedbackEntry:
    query_id: str
    query: str
    answer: str
    provenance: list[dict[str, Any]]
    feedback: str | None = None
    relevance_score: float = 0.0


@dataclass
class ImplicitFeedbackEntry:
    query_id: str
    click_url: str
    event: str = "citation_click"


class FeedbackTelemetryService:
    def __init__(
        self,
        log_path: str = "data/telemetry_logs.jsonl",
        implicit_log_path: str | None = None,
    ) -> None:
        self.log_path = log_path
        self.implicit_log_path = implicit_log_path or f"{log_path}.implicit"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.implicit_log_path), exist_ok=True)

    async def log_interaction(
        self, query_id: str, query: str, answer: str, provenance: list[dict[str, Any]], feedback: str | None = None
    ) -> None:
        entry = FeedbackEntry(query_id=query_id, query=query, answer=answer, provenance=provenance, feedback=feedback)
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry.__dict__) + "\n")
        except Exception as e:
            print(f"Telemetry logging error: {str(e)}")

    async def log_implicit_feedback(self, query_id: str, click_url: str) -> None:
        entry = ImplicitFeedbackEntry(query_id=query_id, click_url=click_url)
        try:
            with open(self.implicit_log_path, "a") as f:
                f.write(json.dumps(entry.__dict__) + "\n")
        except Exception as e:
            print(f"Telemetry logging error: {str(e)}")
