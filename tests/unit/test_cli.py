"""Tests for cli.py pure helper functions."""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

from data_engineering_copilot.cli import (
    _answer_correctness,
    _bm25_cache_path,
    _check_deps_before_dispatch,
    _claude_source_filter,
    _confirm_required,
    _default_coverage_paths,
    _delete_bm25_cache,
    _deterministic_sample_indices,
    _disable_rewrites_for_eval,
    _eval_retrieval_row,
    _get_plan_phases,
    _is_recall_file,
    _list_qdrant_collections,
    _load_active_state,
    _percentile,
    _poll_gave_up,
    _print_fast_eval,
    _purge_bm25_cache_dir,
    _purge_generation_bm25_caches,
    _purge_generation_state,
    _qdrant_collection_aliases,
    _qdrant_delete_collection,
    _qdrant_drop_alias,
    _resolve_spark_embedding_name,
    _spark_commit_short,
    _spark_generation_collection,
    _validation_report_path,
    _write_active_state,
    cancel,
    re_fullmatch_identifier,
)


class TestClaudeSourceFilter:
    def test_returns_source_names_when_provided(self) -> None:
        result = _claude_source_filter("any question", ["src1", "src2"])
        assert result == ["src1", "src2"]

    def test_returns_none_when_no_keywords(self) -> None:
        result = _claude_source_filter("how to use spark?", None)
        assert result is None

    def test_auto_routes_claude_keywords(self) -> None:
        result = _claude_source_filter("how to use Claude API?", None)
        assert result is not None
        assert len(result) == 2


class TestBm25CachePath:
    def test_returns_path_with_collection_name(self) -> None:
        mock_settings = MagicMock()
        mock_settings.collection_name = "test_collection"
        with (
            patch("data_engineering_copilot.cli.settings", mock_settings),
            patch("data_engineering_copilot.config.settings.PROJECT_ROOT", pathlib.Path("/tmp/project")),
            patch(
                "data_engineering_copilot.infrastructure.async_qdrant_store.PROJECT_ROOT", pathlib.Path("/tmp/project")
            ),
        ):
            result = _bm25_cache_path()
            assert result == pathlib.Path("/tmp/project/.bm25_cache/test_collection.json")


class TestSparkGenerationCollection:
    def test_returns_collection_name(self) -> None:
        result = _spark_generation_collection("spark-main-abc123")
        assert result == "data_engineering_docs__spark-main-abc123"


class TestSparkCommitShort:
    def test_truncates_to_8_chars(self) -> None:
        assert _spark_commit_short("abcdef1234567890") == "abcdef12"

    def test_short_commit_unchanged(self) -> None:
        assert _spark_commit_short("abc") == "abc"


class TestResolveSparkEmbeddingName:
    def test_nvidia_provider(self) -> None:
        with patch("data_engineering_copilot.cli.settings") as mock_settings:
            mock_settings.embedding_provider = "nvidia"
            mock_settings.nvidia_embedding_model = "nvidia/test-model"
            mock_settings.active_embedding_model_name.return_value = "other"
            assert _resolve_spark_embedding_name() == "nvidia/test-model"

    def test_openrouter_provider(self) -> None:
        with patch("data_engineering_copilot.cli.settings") as mock_settings:
            mock_settings.embedding_provider = "openrouter"
            mock_settings.openrouter_embedding_model = "openrouter/test-model"
            mock_settings.active_embedding_model_name.return_value = "other"
            assert _resolve_spark_embedding_name() == "openrouter/test-model"

    def test_fallback_to_active(self) -> None:
        with patch("data_engineering_copilot.cli.settings") as mock_settings:
            mock_settings.embedding_provider = "local-hf"
            mock_settings.active_embedding_model_name.return_value = "local-model"
            assert _resolve_spark_embedding_name() == "local-model"


class TestReFullmatchIdentifier:
    def test_valid_identifiers(self) -> None:
        assert re_fullmatch_identifier("abc123") is True
        assert re_fullmatch_identifier("spark-main-abc123") is True
        assert re_fullmatch_identifier("v1.2.3") is True
        assert re_fullmatch_identifier("test:colon") is True

    def test_invalid_identifiers(self) -> None:
        assert re_fullmatch_identifier("abc 123") is False
        assert re_fullmatch_identifier("abc/123") is False
        assert re_fullmatch_identifier("") is False
        assert re_fullmatch_identifier("abc@123") is False


class TestConfirmRequired:
    def test_force_env_var(self) -> None:
        with patch("os.environ.get", return_value="1"):
            assert _confirm_required("test") is True

    def test_user_confirms(self) -> None:
        with patch("os.environ.get", return_value=None), patch("builtins.input", return_value="y"):
            assert _confirm_required("test") is True

    def test_user_declines(self) -> None:
        with patch("os.environ.get", return_value=None), patch("builtins.input", return_value="n"):
            assert _confirm_required("test") is False

    def test_eof_error(self) -> None:
        with patch("os.environ.get", return_value=None), patch("builtins.input", side_effect=EOFError):
            assert _confirm_required("test") is False


class TestPercentile:
    def test_empty_returns_none(self) -> None:
        assert _percentile([], 0.5) is None

    def test_median(self) -> None:
        assert _percentile([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_single_value(self) -> None:
        assert _percentile([42], 0.5) == 42.0

    def test_quartiles(self) -> None:
        vals = [1, 2, 3, 4]
        assert _percentile(vals, 0.25) == 1.75
        assert _percentile(vals, 0.75) == 3.25


class TestDeterministicSampleIndices:
    def test_sample_within_bounds(self) -> None:
        result = _deterministic_sample_indices(n_total=10, n_sample=3)
        assert len(result) == 3
        assert all(0 <= i < 10 for i in result)

    def test_sample_exceeds_total(self) -> None:
        result = _deterministic_sample_indices(n_total=5, n_sample=10)
        assert len(result) == 5

    def test_zero_sample(self) -> None:
        result = _deterministic_sample_indices(n_total=10, n_sample=0)
        assert result == []

    def test_deterministic(self) -> None:
        result1 = _deterministic_sample_indices(n_total=100, n_sample=5, seed=42)
        result2 = _deterministic_sample_indices(n_total=100, n_sample=5, seed=42)
        assert result1 == result2


class TestAnswerCorrectness:
    def test_empty_ground_truth(self) -> None:
        assert _answer_correctness("any answer", "") == 0.0

    def test_with_ground_truth(self) -> None:
        with patch("data_engineering_copilot.services.rag_evaluation.answer_token_f1", return_value=0.75):
            result = _answer_correctness("test answer", "test answer")
            assert result == 0.75


class TestPollGaveUp:
    def test_prints_guidance(self, capsys) -> None:
        _poll_gave_up("task-123", "http://api/cancel/task-123")
        captured = capsys.readouterr()
        assert "Could not reach the ingestion API" in captured.out
        assert "task-123" in captured.out
        assert "http://api/cancel/task-123" in captured.out

    def test_prints_monitor_command(self, capsys) -> None:
        _poll_gave_up("task-456", "http://api/cancel/task-456")
        captured = capsys.readouterr()
        assert "dec ingestion-monitor --task-id task-456" in captured.out
        assert "dec cancel task-456" in captured.out


class TestValidationReportPath:
    def test_returns_path(self) -> None:
        mock_settings = MagicMock()
        mock_settings.index_state_dir = pathlib.Path("/tmp/state")
        with patch("data_engineering_copilot.cli.settings", mock_settings):
            result = _validation_report_path("pinned-abc123")
            assert result == pathlib.Path("/tmp/state/validation-pinned-abc123.json")


class TestEvalRetrievalRow:
    def test_perfect_recall(self) -> None:
        result = _eval_retrieval_row(
            "query",
            "how_to",
            ["url_a", "url_b"],
            ["url_a", "url_b"],
            k=5,
        )
        assert result["recall"] == 1.0
        assert result["precision"] == 0.4
        assert result["mrr"] == 1.0

    def test_partial_recall(self) -> None:
        result = _eval_retrieval_row(
            "query",
            "how_to",
            ["url_a", "url_b"],
            ["url_a", "url_c"],
            k=5,
        )
        assert result["recall"] == 0.5
        assert result["precision"] == 0.2

    def test_no_recall(self) -> None:
        result = _eval_retrieval_row(
            "query",
            "how_to",
            ["url_a"],
            ["url_b", "url_c"],
            k=5,
        )
        assert result["recall"] == 0.0
        assert result["mrr"] == 0.0

    def test_deduplicates_hits(self) -> None:
        result = _eval_retrieval_row(
            "query",
            "how_to",
            ["url_a"],
            ["url_a", "url_a", "url_a"],
            k=5,
        )
        assert result["recall"] == 1.0

    def test_empty_expected(self) -> None:
        result = _eval_retrieval_row("query", "how_to", [], ["url_a"], k=5)
        assert result["recall"] == 1.0

    def test_mrr_at_rank_2(self) -> None:
        result = _eval_retrieval_row(
            "query",
            "how_to",
            ["url_b"],
            ["url_a", "url_b"],
            k=5,
        )
        assert result["mrr"] == 0.5


class TestDisableRewritesForEval:
    def test_returns_disabled(self) -> None:
        service = MagicMock()
        result = _disable_rewrites_for_eval(service)
        assert result == "disabled"
        assert service.query_rewriter is None

    def test_suppresses_attribute_error(self) -> None:
        service = object()
        result = _disable_rewrites_for_eval(service)
        assert result == "disabled"


class TestIsRecallFile:
    def test_recall_file(self, tmp_path) -> None:
        p = tmp_path / "recall.jsonl"
        p.write_text(json.dumps({"id": "1", "question": "q", "expected_urls": ["u"]}) + "\n")
        assert _is_recall_file(p) is True

    def test_empty_file(self, tmp_path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert _is_recall_file(p) is False

    def test_non_recall_file(self, tmp_path) -> None:
        p = tmp_path / "other.jsonl"
        p.write_text(json.dumps({"id": "1", "input": "x", "expected_output": "y"}) + "\n")
        assert _is_recall_file(p) is False


class TestDefaultCoveragePaths:
    def test_finds_recall_files(self, tmp_path) -> None:
        (tmp_path / "recall1.jsonl").write_text(json.dumps({"id": "1", "question": "q", "expected_urls": ["u"]}) + "\n")
        (tmp_path / "other.jsonl").write_text(json.dumps({"id": "2", "input": "x"}) + "\n")
        result = _default_coverage_paths(tmp_path)
        assert len(result) == 1
        assert result[0].name == "recall1.jsonl"

    def test_empty_dir(self, tmp_path) -> None:
        result = _default_coverage_paths(tmp_path)
        assert result == []


class TestPrintFastEval:
    def test_prints_report(self, capsys) -> None:
        report = {
            "generation": "pinned-abc123",
            "layers": {
                "corpus": {"chunk_count": 100, "source_count": 10, "content_hash_duplicates": 2, "empty_chunks": 0},
                "chunk": {
                    "count": 50,
                    "mean_chars": 300,
                    "p95_chars": 500,
                    "p99_chars": 800,
                    "oversized": 3,
                    "over_token_budget": 1,
                    "boundary_issues": 0,
                },
                "coverage": {"pass": 45, "rows": 50},
                "embedding": {"consistency": {"similarity": 0.95}, "semantic_sanity": {"passed": 10, "pairs": 10}},
                "vectordb": {
                    "point_count": 100,
                    "chunk_count": 100,
                    "count_matches": True,
                    "self_retrieval_hits": 9,
                    "self_retrieval": list(range(10)),
                },
                "retrieval": {"source_recall": 0.85, "mrr": 0.72, "rows": 50},
            },
        }
        _print_fast_eval(report)
        captured = capsys.readouterr()
        assert "FAST EVAL" in captured.out
        assert "pinned-abc123" in captured.out
        assert "100 chunks" in captured.out

    def test_prints_vectordb_error(self, capsys) -> None:
        report = {
            "generation": "test",
            "layers": {
                "corpus": {},
                "vectordb": {"error": "connection failed"},
            },
        }
        _print_fast_eval(report)
        captured = capsys.readouterr()
        assert "connection failed" in captured.out


class TestGetPlanPhases:
    def test_returns_phases(self) -> None:
        result = _get_plan_phases()
        assert isinstance(result, tuple)
        assert len(result) > 0


class TestLoadActiveState:
    def test_returns_loaded_json(self, tmp_path) -> None:
        mock_settings = MagicMock()
        mock_settings.index_state_dir = tmp_path
        state_file = tmp_path / "active.json"
        state_file.write_text(json.dumps({"generation": "test"}))
        with patch("data_engineering_copilot.cli.settings", mock_settings):
            result = _load_active_state()
        assert result == {"generation": "test"}

    def test_missing_file_returns_empty_dict(self, tmp_path) -> None:
        mock_settings = MagicMock()
        mock_settings.index_state_dir = tmp_path
        with patch("data_engineering_copilot.cli.settings", mock_settings):
            result = _load_active_state()
        assert result == {}

    def test_invalid_json_returns_empty_dict(self, tmp_path) -> None:
        mock_settings = MagicMock()
        mock_settings.index_state_dir = tmp_path
        (tmp_path / "active.json").write_text("not valid json {{{")
        with patch("data_engineering_copilot.cli.settings", mock_settings):
            result = _load_active_state()
        assert result == {}


class TestWriteActiveState:
    def test_writes_json_and_appends_history(self, tmp_path) -> None:
        mock_settings = MagicMock()
        mock_settings.index_state_dir = tmp_path
        state = {"generation": "test", "version": 2}
        with patch("data_engineering_copilot.cli.settings", mock_settings):
            _write_active_state(state)
        assert (tmp_path / "active.json").read_text() == json.dumps(state, indent=2)
        history = (tmp_path / "history.jsonl").read_text().strip()
        assert json.loads(history) == state


class TestDeleteBm25Cache:
    def test_deletes_existing_cache(self, tmp_path) -> None:
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("{}")
        with patch("data_engineering_copilot.cli._bm25_cache_path", return_value=cache_file):
            _delete_bm25_cache()
        assert not cache_file.exists()

    def test_no_op_when_missing(self, tmp_path) -> None:
        cache_file = tmp_path / "missing.json"
        with patch("data_engineering_copilot.cli._bm25_cache_path", return_value=cache_file):
            _delete_bm25_cache()
        assert not cache_file.exists()


class TestPurgeBm25CacheDir:
    def test_deletes_all_json_files(self, tmp_path) -> None:
        cache_dir = tmp_path / ".bm25_cache"
        cache_dir.mkdir()
        (cache_dir / "a.json").write_text("{}")
        (cache_dir / "b.json").write_text("{}")
        (cache_dir / "keep.txt").write_text("keep")
        with patch("data_engineering_copilot.config.settings.PROJECT_ROOT", tmp_path):
            _purge_bm25_cache_dir()
        assert not (cache_dir / "a.json").exists()
        assert not (cache_dir / "b.json").exists()
        assert (cache_dir / "keep.txt").exists()

    def test_no_op_when_dir_missing(self, tmp_path) -> None:
        with patch("data_engineering_copilot.config.settings.PROJECT_ROOT", tmp_path):
            _purge_bm25_cache_dir()


class TestPurgeGenerationState:
    def test_deletes_matching_files(self, tmp_path) -> None:
        mock_settings = MagicMock()
        mock_settings.index_state_dir = tmp_path
        (tmp_path / "active.json").write_text("{}")
        (tmp_path / "history.jsonl").write_text("{}\n")
        (tmp_path / "validation-abc.json").write_text("{}")
        (tmp_path / "other-file.json").write_text("{}")
        with patch("data_engineering_copilot.cli.settings", mock_settings):
            _purge_generation_state()
        assert not (tmp_path / "active.json").exists()
        assert not (tmp_path / "history.jsonl").exists()
        assert not (tmp_path / "validation-abc.json").exists()
        assert (tmp_path / "other-file.json").exists()


class TestPurgeGenerationBm25Caches:
    def test_deletes_generation_cache_files(self, tmp_path) -> None:
        cache_dir = tmp_path / ".bm25_cache"
        cache_dir.mkdir()
        gen_file = cache_dir / "data_engineering_docs__gen1.json"
        gen_file.write_text("{}")
        other_file = cache_dir / "other_collection.json"
        other_file.write_text("{}")
        with patch("data_engineering_copilot.config.settings.PROJECT_ROOT", tmp_path):
            _purge_generation_bm25_caches()
        assert not gen_file.exists()
        assert other_file.exists()

    def test_no_op_when_dir_missing(self, tmp_path) -> None:
        with patch("data_engineering_copilot.config.settings.PROJECT_ROOT", tmp_path):
            _purge_generation_bm25_caches()


class TestListQdrantCollections:
    def test_returns_collection_names(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(
            {"result": {"collections": [{"name": "col1"}, {"name": "col2"}]}}
        ).encode()
        with (
            patch("data_engineering_copilot.cli.settings") as mock_settings,
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
        ):
            mock_settings.qdrant_url = "http://qdrant:6333"
            result = _list_qdrant_collections()
        assert result == ["col1", "col2"]
        mock_urlopen.assert_called_once()


class TestQdrantDeleteCollection:
    def test_sends_delete_request(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
        with (
            patch("data_engineering_copilot.cli.settings") as mock_settings,
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
        ):
            mock_settings.qdrant_url = "http://qdrant:6333"
            _qdrant_delete_collection("test_collection")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_full_url() == "http://qdrant:6333/collections/test_collection"
        assert req.get_method() == "DELETE"


class TestQdrantDropAlias:
    def test_sends_delete_alias_request(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
        with (
            patch("data_engineering_copilot.cli.settings") as mock_settings,
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
        ):
            mock_settings.qdrant_url = "http://qdrant:6333"
            mock_settings.active_collection_alias = "my_alias"
            _qdrant_drop_alias()
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_full_url() == "http://qdrant:6333/collections/aliases"
        assert req.get_method() == "POST"


class TestQdrantCollectionAliases:
    def test_returns_alias_names(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(
            {"result": {"aliases": [{"alias_name": "alias1"}, {"alias_name": "alias2"}]}}
        ).encode()
        with (
            patch("data_engineering_copilot.cli.settings") as mock_settings,
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
        ):
            mock_settings.qdrant_url = "http://qdrant:6333"
            result = _qdrant_collection_aliases("test_collection")
        assert result == ["alias1", "alias2"]
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_full_url() == "http://qdrant:6333/collections/test_collection/aliases"


class TestCheckDepsBeforeDispatch:
    def test_no_exception_on_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"deps_fingerprint_ok": True}).encode()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            _check_deps_before_dispatch()

    def test_no_exception_on_unreachable(self) -> None:
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            _check_deps_before_dispatch()


class TestCancel:
    def test_posts_cancel_request(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"status": "cancelled"}).encode()
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            cancel("task-123")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_full_url() == "http://localhost:8000/api/v1/ingest/task-123/cancel"
        assert req.get_method() == "POST"
