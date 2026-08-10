"""Integration: Langfuse dataset upload + RAG experiment (Phase 6).

Uploads a small dataset via the v4 ``create_dataset``/``create_dataset_item``
API, runs ``run_rag_experiment`` against it (RAG task + faithfulness evaluator
stubbed, offline RAGAS metrics stubbed), and verifies the dataset and experiment
run are visible via the public API. Auto-skips when Langfuse is unreachable.
"""

from __future__ import annotations

import base64
import json
import urllib.request
import uuid

import pytest

from tests.conftest import require_langfuse


@pytest.mark.integration
@pytest.mark.langfuse
def test_dataset_upload_and_experiment_run(monkeypatch) -> None:
    require_langfuse()

    from data_engineering_copilot.config.settings import settings
    from data_engineering_copilot.evaluation import langfuse_datasets
    from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance
    from data_engineering_copilot.services.ragas_evaluation import RagasEvalResult

    if get_langfuse_instance() is None:
        pytest.skip("Langfuse client could not be initialized (missing LANGFUSE_PUBLIC_KEY/SECRET_KEY)")

    dataset_name = f"dec-integration-{uuid.uuid4().hex[:8]}"
    experiment_name = f"exp-{uuid.uuid4().hex[:8]}"

    class _StubAnswer:
        def __init__(self, query: str) -> None:
            self.text = f"stub answer to {query}"

    class _StubService:
        async def answer(self, query, source_filter=None):
            return _StubAnswer(query)

    class _FakeRagasEvaluator:
        def evaluate(self, **kwargs):
            return RagasEvalResult(
                context_recall=1.0,
                context_precision=1.0,
                faithfulness=1.0,
                answer_relevancy=1.0,
                overall=1.0,
            )

    monkeypatch.setattr("data_engineering_copilot.factory.build_rag_service", lambda: _StubService())
    monkeypatch.setattr(
        "data_engineering_copilot.services.ragas_evaluation.RagasEvaluator",
        _FakeRagasEvaluator,
    )

    ok = langfuse_datasets.upload_evaluation_dataset_rows(
        dataset_name=dataset_name,
        items=[
            {"input": {"query": "q1"}, "expected_output": {"answer": "a1"}, "metadata": {"contexts": ["c1"]}},
            {"input": {"query": "q2"}, "expected_output": {"answer": "a2"}, "metadata": {"contexts": ["c2"]}},
        ],
    )
    assert ok is True

    result = langfuse_datasets.run_rag_experiment(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        max_concurrency=2,
    )
    assert result is not None
    assert getattr(result, "dataset_run_url", None)

    # Verify the dataset is visible via the public API.
    url = f"{settings.langfuse_host.rstrip('/')}/api/public/datasets?limit=100"
    req = urllib.request.Request(url)
    req.add_header(
        "Authorization",
        "Basic "
        + base64.b64encode(
            f"{settings.langfuse_public_key.get_secret_value()}:{settings.langfuse_secret_key.get_secret_value()}".encode()
        ).decode(),
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())

    names = [d.get("name") for d in body.get("data", [])]
    assert dataset_name in names
