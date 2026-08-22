import json
import os

import pytest
from pydantic import SecretStr

from data_engineering_copilot.config.settings import (
    AppSettings,
    load_documentation_sources,
    resolve_active_generation,
)
from tests.conftest import make_settings


def test_resolve_active_generation_from_state_file(tmp_path, monkeypatch) -> None:
    from data_engineering_copilot.config import settings as settings_module

    state_dir = tmp_path / ".index_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "active.json").write_text(
        json.dumps({"generation": "spark-4.0.0-fa33ea00-test", "collection": "docs__gen"})
    )
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    assert resolve_active_generation() == "spark-4.0.0-fa33ea00-test"


def test_resolve_active_generation_falls_back_to_settings(tmp_path, monkeypatch) -> None:
    from data_engineering_copilot.config import settings as settings_module

    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        settings_module,
        "settings",
        make_settings(active_index_generation="legacy-gen"),
    )
    assert resolve_active_generation() == "legacy-gen"


def test_load_documentation_sources_from_json(tmp_path):
    config_path = tmp_path / "documentation_sources.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": "Example Docs",
                    "start_urls": ["https://example.com/docs/"],
                    "allowed_domains": ["example.com"],
                    "url_prefixes": ["https://example.com/docs/"],
                }
            ]
        ),
        encoding="utf-8",
    )

    sources = load_documentation_sources(config_path)

    assert len(sources) == 1
    assert sources[0].name == "Example Docs"
    assert sources[0].start_urls == ("https://example.com/docs/",)
    assert sources[0].allowed_domains == ("example.com",)


def test_app_settings_default_logging_enabled() -> None:
    settings = make_settings()

    assert settings.logging_enabled is True


def test_app_settings_hybrid_search_defaults() -> None:
    settings = make_settings()
    assert settings.hybrid_search_enabled is True
    assert settings.hybrid_rrf_k == 60
    assert settings.context_compression_enabled is False
    assert settings.max_context_tokens == 4096
    assert settings.query_rewrite_enabled is True
    assert settings.groundedness_enabled is True


def test_active_generation_defaults_empty() -> None:
    settings = make_settings()
    assert settings.active_index_generation == ""
    assert settings.active_collection_name == ""
    assert settings.active_collection_alias == "data_engineering_docs"


def test_active_generation_overridable() -> None:
    settings = make_settings(
        active_index_generation="spark-4.0.0-fa33ea00-abc",
        active_collection_name="data_engineering_docs__spark-4.0.0-fa33ea00-abc",
    )
    assert settings.active_index_generation == "spark-4.0.0-fa33ea00-abc"
    assert settings.active_collection_name == "data_engineering_docs__spark-4.0.0-fa33ea00-abc"


def test_active_generation_rejects_invalid() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_settings(active_index_generation="spark 4.0.0")
    with pytest.raises(ValidationError):
        make_settings(active_collection_name="bad/name")


def test_app_settings_hybrid_search_overridable() -> None:
    settings = make_settings(
        hybrid_search_enabled=False,
        hybrid_rrf_k=100,
        context_compression_enabled=True,
        max_context_tokens=8192,
    )
    assert settings.hybrid_search_enabled is False
    assert settings.hybrid_rrf_k == 100
    assert settings.context_compression_enabled is True
    assert settings.max_context_tokens == 8192


def test_prompt_citation_enforcement_rejects_invalid() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="prompt_citation_enforcement"):
        make_settings(prompt_citation_enforcement="invalid")


def test_prompt_citation_enforcement_accepts_valid_values() -> None:
    for value in ("strict", "soft", "off"):
        settings = make_settings(prompt_citation_enforcement=value)
        assert settings.prompt_citation_enforcement == value


def test_prompt_citation_enforcement_default_is_strict() -> None:
    assert make_settings().prompt_citation_enforcement == "strict"


def test_code_llm_defaults_empty() -> None:
    settings = make_settings()
    assert settings.code_llm_provider == ""
    assert settings.code_llm_model == ""
    assert settings.nvidia_rpm_limit == 36
    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"


def test_code_llm_overridable() -> None:
    settings = make_settings(
        code_llm_provider="nvidia",
        code_llm_model="qwen/qwen2.5-coder-32b-instruct",
        nvidia_api_key="nvapi-test",
        nvidia_rpm_limit=80,
        nvidia_base_url="https://custom.nvidia.com/v1",
        _test_allow_non_ollama=True,
    )
    assert settings.code_llm_provider == "nvidia"
    assert settings.code_llm_model == "qwen/qwen2.5-coder-32b-instruct"
    assert settings.nvidia_rpm_limit == 80
    assert settings.nvidia_base_url == "https://custom.nvidia.com/v1"


def test_embedding_model_dimensions_lookup() -> None:
    settings = make_settings()
    assert settings.get_embedding_dimension() == 2048


def test_get_embedding_dimension_nvidia() -> None:
    s = make_settings(
        embedding_provider="nvidia",
        nvidia_embedding_model="nvidia/nemotron-3-embed-1b",
        nvidia_api_key=SecretStr("nvapi-test"),
        _test_allow_non_ollama=True,
    )
    assert s.get_embedding_dimension() == 2048


def test_get_embedding_dimension_unknown_model() -> None:
    s = make_settings(embedding_model_name="unknown-model")
    assert s.get_embedding_dimension() == s.default_embedding_dimension


def test_get_embedding_dimension_huggingface() -> None:
    s = make_settings(
        embedding_provider="huggingface",
        huggingface_api_key="hf-test",
        huggingface_embedding_model="nvidia/Nemotron-3-Embed-1B-BF16",
        _test_allow_non_ollama=True,
    )
    assert s.get_embedding_dimension() == 2048


def test_huggingface_api_key_alias_reads_hf_token(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_secret\n", encoding="utf-8")
    from data_engineering_copilot.config.settings import AppSettings

    settings = AppSettings(_env_file=env_file)
    assert settings.huggingface_api_key.get_secret_value() == "hf_secret"


def test_nvidia_nim_rpd_limit_default() -> None:
    settings = make_settings()
    # Free Developer tier is 1000 RPD (not 10000).
    assert settings.nvidia_rpd_limit == 1000


def test_nvidia_nim_rpd_limit_accepts_legacy_alias(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("NVIDIA_NIM_RPD_LIMIT=0\n", encoding="utf-8")
    settings = make_settings(_env_file=env_file)
    assert settings.nvidia_rpd_limit == 0


def test_embedding_provider_nvidia_missing_api_key_raises() -> None:

    with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
        make_settings(
            embedding_provider="nvidia",
            nvidia_api_key="",
            _test_allow_non_ollama=True,
        )


def test_api_key_field_hermetic_default_empty() -> None:
    settings = make_settings()
    assert settings.api_key.get_secret_value() == ""


def test_api_key_overridable() -> None:
    settings = make_settings(api_key="sk-test-key")
    assert settings.api_key.get_secret_value() == "sk-test-key"


def test_api_key_not_in_str() -> None:
    settings = make_settings(api_key="sk-ultra-secret-42")
    rendered = str(settings)
    assert "sk-ultra-secret-42" not in rendered


def test_env_local_overrides_env(tmp_path) -> None:
    env_file = tmp_path / ".env"
    local_file = tmp_path / ".env.local"
    env_file.write_text("COLLECTION_NAME=from_env\n", encoding="utf-8")
    local_file.write_text("COLLECTION_NAME=from_env_local\n", encoding="utf-8")

    previous_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        kwargs: dict = {
            "_env_file": (".env", ".env.local"),
            "skip_provider_check": True,
            "llm_provider": "ollama",
            "llm_model": "llama3.2:3b",
            "embedding_provider": "ollama",
            "local_hf_embedding_model": "nvidia/Nemotron-3-Embed-1B-BF16",
            "ollama_base_url": "http://localhost:11434",
        }
        settings = AppSettings(**kwargs)
    finally:
        os.chdir(previous_cwd)

    assert settings.collection_name == "from_env_local"


def test_no_duplicate_embed_concurrency() -> None:
    settings = make_settings()
    assert settings.model_fields["embed_concurrency"].default == 1


def test_validate_all_passes_on_defaults() -> None:
    settings = make_settings()
    settings.validate_all()  # must not raise


def test_validate_all_detects_conflicts() -> None:
    from pydantic import ValidationError

    def _make_bad(**overrides) -> AppSettings:
        from tests.conftest import make_settings

        base = make_settings()
        kwargs: dict = {
            field: getattr(base, field)
            for field in base.model_fields
            if field not in {"sources", "skip_provider_check"}
        }
        kwargs.update(overrides)
        return AppSettings(**kwargs)

    bad = _make_bad(reranker_top_k=20, retrieval_top_k=5)
    with pytest.raises(ValidationError, match="reranker_top_k"):
        bad.validate_all()

    negative = _make_bad(max_pages_per_source=-1)
    with pytest.raises(ValidationError, match="max_pages_per_source"):
        negative.validate_all()

    no_hard_cap = _make_bad(max_pages_hard_cap=0)
    with pytest.raises(ValidationError, match="max_pages_hard_cap"):
        no_hard_cap.validate_all()

    no_multiplier = _make_bad(crawl_attempt_multiplier=0)
    with pytest.raises(ValidationError, match="crawl_attempt_multiplier"):
        no_multiplier.validate_all()

    no_recovery = _make_bad(recovery_max_pages=0)
    with pytest.raises(ValidationError, match="recovery_max_pages"):
        no_recovery.validate_all()

    no_provider = _make_bad(llm_provider="", embedding_provider="")
    with pytest.raises(ValidationError, match="provider"):
        no_provider.validate_all()


def test_chat_settings_defaults() -> None:
    settings = make_settings()
    assert settings.chat_enabled is True
    assert settings.chat_session_ttl_seconds == 259200
    assert settings.chat_history_max_turns == 10
    assert settings.chat_history_max_tokens == 2048
    assert settings.chat_db_url == ""
    assert settings.chat_title_max_chars == 60


def test_chat_settings_overridable() -> None:
    settings = make_settings(
        chat_enabled=False,
        chat_session_ttl_seconds=3600,
        chat_history_max_turns=5,
        chat_history_max_tokens=1024,
        chat_db_url="postgresql://user:pass@db:5432/chat",
        chat_title_max_chars=40,
    )
    assert settings.chat_enabled is False
    assert settings.chat_session_ttl_seconds == 3600
    assert settings.chat_history_max_turns == 5
    assert settings.chat_history_max_tokens == 1024
    assert settings.chat_db_url == "postgresql://user:pass@db:5432/chat"
    assert settings.chat_title_max_chars == 40


def test_chat_settings_validate_all() -> None:
    from pydantic import ValidationError

    def _make_bad(**overrides) -> AppSettings:
        base = make_settings()
        kwargs: dict = {
            field: getattr(base, field)
            for field in base.model_fields
            if field not in {"sources", "skip_provider_check"}
        }
        kwargs.update(overrides)
        return AppSettings(**kwargs)

    bad_ttl = _make_bad(chat_session_ttl_seconds=10)
    with pytest.raises(ValidationError, match="chat_session_ttl_seconds"):
        bad_ttl.validate_all()

    bad_turns = _make_bad(chat_history_max_turns=0)
    with pytest.raises(ValidationError, match="chat_history_max_turns"):
        bad_turns.validate_all()

    bad_tokens = _make_bad(chat_history_max_tokens=16)
    with pytest.raises(ValidationError, match="chat_history_max_tokens"):
        bad_tokens.validate_all()

    bad_title = _make_bad(chat_title_max_chars=2)
    with pytest.raises(ValidationError, match="chat_title_max_chars"):
        bad_title.validate_all()


def test_chat_speed_settings_defaults() -> None:
    settings = make_settings()
    assert settings.chat_rewrite_local is False
    assert settings.chat_scope_local is False
    assert settings.chat_answer_local is False
    assert settings.chat_rerank_local is True
    assert settings.chat_cache_recall_enabled is False
    assert settings.chat_cache_top_k == 3
    assert settings.chat_cache_recall_threshold == 0.70
    assert settings.chat_cache_max_age_seconds == 86400
    assert settings.chat_blocked_url_substrings == ["system-prompts.md"]
    assert settings.chat_domain_sources == []
    assert settings.chat_suggestions_enabled is True
    assert settings.chat_suggestions_count == 3
    assert settings.chat_suggestions_mode == "hybrid"


def test_chat_speed_settings_overridable() -> None:
    settings = make_settings(
        chat_rewrite_local=False,
        chat_scope_local=False,
        chat_answer_local=True,
        chat_rerank_local=False,
        chat_cache_recall_enabled=True,
        chat_cache_top_k=5,
        chat_cache_recall_threshold=0.60,
        chat_cache_max_age_seconds=3600,
        chat_blocked_url_substrings=["blocked-a"],
        chat_domain_sources=["Apache Spark 4.0.0"],
        chat_suggestions_enabled=False,
        chat_suggestions_count=5,
        chat_suggestions_mode="rule",
    )
    assert settings.chat_rewrite_local is False
    assert settings.chat_scope_local is False
    assert settings.chat_answer_local is True
    assert settings.chat_rerank_local is False
    assert settings.chat_cache_recall_enabled is True
    assert settings.chat_cache_top_k == 5
    assert settings.chat_cache_recall_threshold == 0.60
    assert settings.chat_cache_max_age_seconds == 3600
    assert settings.chat_blocked_url_substrings == ["blocked-a"]
    assert settings.chat_domain_sources == ["Apache Spark 4.0.0"]
    assert settings.chat_suggestions_enabled is False
    assert settings.chat_suggestions_count == 5
    assert settings.chat_suggestions_mode == "rule"


def test_cache_toggle_settings_defaults() -> None:
    settings = make_settings()
    assert settings.query_cache_enabled is True
    assert settings.query_cache_exact_enabled is True
    assert settings.query_cache_semantic_enabled is True
    assert settings.embedding_cache_enabled is True
    assert settings.crawl_cache_enabled is True


def test_cache_toggle_settings_overridable() -> None:
    settings = make_settings(
        query_cache_enabled=False,
        query_cache_exact_enabled=False,
        query_cache_semantic_enabled=False,
        embedding_cache_enabled=False,
        crawl_cache_enabled=False,
    )
    assert settings.query_cache_enabled is False
    assert settings.query_cache_exact_enabled is False
    assert settings.query_cache_semantic_enabled is False
    assert settings.embedding_cache_enabled is False
    assert settings.crawl_cache_enabled is False


def test_chat_speed_settings_validate_all() -> None:
    from pydantic import ValidationError

    def _make_bad(**overrides) -> AppSettings:
        base = make_settings()
        kwargs: dict = {
            field: getattr(base, field)
            for field in base.model_fields
            if field not in {"sources", "skip_provider_check"}
        }
        kwargs.update(overrides)
        return AppSettings(**kwargs)

    bad_threshold = _make_bad(chat_cache_recall_threshold=0.0)
    with pytest.raises(ValidationError, match="chat_cache_recall_threshold"):
        bad_threshold.validate_all()

    bad_top_k = _make_bad(chat_cache_top_k=0)
    with pytest.raises(ValidationError, match="chat_cache_top_k"):
        bad_top_k.validate_all()

    bad_age = _make_bad(chat_cache_max_age_seconds=10)
    with pytest.raises(ValidationError, match="chat_cache_max_age_seconds"):
        bad_age.validate_all()

    bad_count = _make_bad(chat_suggestions_count=0)
    with pytest.raises(ValidationError, match="chat_suggestions_count"):
        bad_count.validate_all()

    bad_mode = _make_bad(chat_suggestions_mode="bogus")
    with pytest.raises(ValidationError, match="chat_suggestions_mode"):
        bad_mode.validate_all()


def test_prompt_augmentation_config_wiring() -> None:
    """Verify that prompt augmentation config flows: AppSettings -> RagConfig -> factory -> PromptBuilder."""
    from data_engineering_copilot.config.settings import AppSettings
    from data_engineering_copilot.domain.models import RagConfig
    from data_engineering_copilot.factory import build_rag_service
    from tests.conftest import make_settings

    # Test AppSettings has the fields
    settings = AppSettings(
        prompt_salted_xml_tags=False,
        prompt_xml_content_escape=False,
        prompt_trailing_instructions=False,
        prompt_citation_enforcement="off",
    )
    assert settings.prompt_salted_xml_tags is False
    assert settings.prompt_xml_content_escape is False
    assert settings.prompt_trailing_instructions is False
    assert settings.prompt_citation_enforcement == "off"

    # Test RagConfig has the fields with same defaults
    rag_config = RagConfig()
    assert rag_config.prompt_salted_xml_tags is True
    assert rag_config.prompt_xml_content_escape is True
    assert rag_config.prompt_trailing_instructions is True
    assert rag_config.prompt_citation_enforcement == "strict"

    # Test factory wiring: build_rag_service propagates settings to RagConfig
    test_settings = make_settings(
        prompt_salted_xml_tags=False,
        prompt_xml_content_escape=False,
        prompt_trailing_instructions=False,
        prompt_citation_enforcement="off",
    )
    rag_service = build_rag_service(test_settings)
    assert rag_service.config.prompt_salted_xml_tags is False
    assert rag_service.config.prompt_xml_content_escape is False
    assert rag_service.config.prompt_trailing_instructions is False
    assert rag_service.config.prompt_citation_enforcement == "off"

    # Test that PromptBuilder and ContextAssembler receive the config
    from data_engineering_copilot.services.context_assembler import ContextAssembler
    from data_engineering_copilot.services.prompt_builder import PromptBuilder

    prompt_builder = PromptBuilder(
        prompt_salted_xml_tags=rag_service.config.prompt_salted_xml_tags,
        prompt_citation_enforcement=rag_service.config.prompt_citation_enforcement,
        prompt_trailing_instructions=rag_service.config.prompt_trailing_instructions,
    )
    assert prompt_builder._prompt_salted_xml_tags is False
    assert prompt_builder._prompt_citation_enforcement == "off"
    assert prompt_builder._prompt_trailing_instructions is False

    context_assembler = ContextAssembler(
        max_context_chars=1000,
        xml_content_escape=rag_service.config.prompt_xml_content_escape,
    )
    assert context_assembler._xml_content_escape is False
