"""Unit tests for evaluation provenance capture."""

from __future__ import annotations

from data_engineering_copilot.evaluation.provenance import (
    active_generation,
    answer_config_fingerprint,
    config_fingerprint,
    embedding_model,
    eval_environment,
    git_commit,
    reranker,
)
from tests.conftest import make_settings


class TestGitCommit:
    def test_returns_commit_in_repo(self) -> None:
        sha = git_commit()
        assert len(sha) == 12
        assert sha.isalnum()

    def test_empty_when_not_a_repo(self, tmp_path) -> None:
        assert git_commit(repo_root=tmp_path) == ""

    def test_respects_short_flag(self) -> None:
        full = git_commit(short=False)
        assert len(full) == 40


class TestEnvResolution:
    def test_active_generation(self) -> None:
        s = make_settings(active_index_generation="pinned-test-123")
        assert active_generation(s) == "pinned-test-123"

    def test_embedding_model_ollama(self) -> None:
        s = make_settings(embedding_provider="ollama", embedding_model_name="nomic-embed-text")
        assert embedding_model(s) == "nomic-embed-text"

    def test_embedding_model_nvidia(self) -> None:
        s = make_settings(
            _test_allow_non_ollama=True,
            embedding_provider="nvidia",
            nvidia_api_key="sk-placeholder",
            nvidia_embedding_model="nvidia/foo",
        )
        assert embedding_model(s) == "nvidia/foo"

    def test_reranker_cloud_first(self) -> None:
        s = make_settings(
            reranker_enabled=True,
            rerank_fallback_order=["openrouter", "nvidia"],
            openrouter_rerank_model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
            nvidia_rerank_model="nv-rerank-qa-mistral-4b:1",
        )
        assert reranker(s) == "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

    def test_reranker_disabled_is_empty(self) -> None:
        s = make_settings(reranker_enabled=False)
        assert reranker(s) == ""

    def test_reranker_falls_back_to_local(self) -> None:
        s = make_settings(reranker_enabled=True, rerank_fallback_order=["nvidia"], nvidia_rerank_model="")
        assert reranker(s) == "BAAI/bge-reranker-v2-m3"


class TestEvalEnvironment:
    def test_contains_all_snapshot_fields(self) -> None:
        s = make_settings()
        env = eval_environment(s)
        for key in ("git_commit", "generation", "embedding_model", "reranker", "chunk_size", "retrieval_top_k"):
            assert key in env

    def test_chunk_settings(self) -> None:
        s = make_settings(chunk_size_words=300, chunk_overlap_words=75, retrieval_top_k=40)
        env = eval_environment(s)
        assert env["chunk_size"] == 300
        assert env["chunk_overlap"] == 75
        assert env["retrieval_top_k"] == 40


class TestConfigFingerprint:
    def test_stable_across_calls(self) -> None:
        s = make_settings()
        assert config_fingerprint(s) == config_fingerprint(s)

    def test_changes_when_embedding_model_changes(self) -> None:
        a = make_settings(embedding_provider="ollama", embedding_model_name="nomic-embed-text")
        b = make_settings(embedding_provider="ollama", embedding_model_name="different-model")
        assert config_fingerprint(a) != config_fingerprint(b)

    def test_excludes_git_commit(self) -> None:
        a = config_fingerprint(make_settings())
        b = config_fingerprint(make_settings())
        assert a == b

    def test_length(self) -> None:
        assert len(config_fingerprint(make_settings())) == 16


class TestAnswerConfigFingerprint:
    def test_stable_across_calls(self) -> None:
        s = make_settings()
        assert answer_config_fingerprint(s) == answer_config_fingerprint(s)

    def test_changes_when_answer_provider_changes(self) -> None:
        a = make_settings(
            answer_llm_provider="openrouter",
            openrouter_api_key="sk-placeholder",
            _test_allow_non_ollama=True,
        )
        b = make_settings(
            answer_llm_provider="groq",
            groq_api_key="sk-placeholder",
            _test_allow_non_ollama=True,
        )
        assert answer_config_fingerprint(a) != answer_config_fingerprint(b)

    def test_changes_when_scope_check_toggles(self) -> None:
        a = make_settings(scope_check_enabled=True)
        b = make_settings(scope_check_enabled=False)
        assert answer_config_fingerprint(a) != answer_config_fingerprint(b)

    def test_length(self) -> None:
        assert len(answer_config_fingerprint(make_settings())) == 16
