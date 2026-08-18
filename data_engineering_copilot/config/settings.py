from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_active_generation() -> str:
    """Resolve the currently active Spark index generation.

    Prefers the runtime state file ``.index_state/active.json`` (written by
    ``dec spark-activate``) so cache scoping and collection routing reflect an
    activated generation without requiring environment changes or restarts.
    Falls back to ``settings.active_index_generation`` when no state file
    exists (legacy operation).
    """
    try:
        state_path = PROJECT_ROOT / ".index_state" / "active.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            generation = state.get("generation", "")
            if generation:
                return generation
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return settings.active_index_generation


@dataclass(frozen=True)
class DocumentationSource:
    name: str
    start_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    url_prefixes: tuple[str, ...] = ()
    priority: int = 1  # Crawl priority (higher = more concurrency slots)


def load_documentation_sources(config_path: Path) -> tuple[DocumentationSource, ...]:
    with config_path.open("r", encoding="utf-8") as file:
        raw_sources = json.load(file)

    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"Documentation source config must contain a non-empty list: {config_path}")

    sources: list[DocumentationSource] = []
    for index, raw_source in enumerate(raw_sources, start=1):
        if not isinstance(raw_source, dict):
            raise ValueError(f"Documentation source #{index} must be an object.")

        name = _required_string(raw_source, "name", index)
        start_urls = _required_string_tuple(raw_source, "start_urls", index)
        allowed_domains = _required_string_tuple(raw_source, "allowed_domains", index)
        url_prefixes = _optional_string_tuple(raw_source, "url_prefixes", index)

        sources.append(
            DocumentationSource(
                name=name,
                start_urls=start_urls,
                allowed_domains=allowed_domains,
                url_prefixes=url_prefixes,
            )
        )

    return tuple(sources)


def _required_string(raw_source: dict, field_name: str, index: int) -> str:
    value = raw_source.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Documentation source #{index} must define a non-empty `{field_name}` string.")
    return value.strip()


def _required_string_tuple(raw_source: dict, field_name: str, index: int) -> tuple[str, ...]:
    value = _optional_string_tuple(raw_source, field_name, index)
    if not value:
        raise ValueError(f"Documentation source #{index} must define at least one `{field_name}` value.")
    return value


def _optional_string_tuple(raw_source: dict, field_name: str, index: int) -> tuple[str, ...]:
    value = raw_source.get(field_name, [])
    if not isinstance(value, list):
        raise ValueError(f"Documentation source #{index} field `{field_name}` must be a list.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Documentation source #{index} field `{field_name}` must contain only non-empty strings.")
    return tuple(item.strip() for item in value)


def _optional_string(raw_source: dict, field_name: str) -> str:
    value = raw_source.get(field_name, "")
    if not isinstance(value, str):
        raise ValueError(f"pinned source config field `{field_name}` must be a string")
    return value.strip()


# ---------------------------------------------------------------------------
# Spark source configuration (Phase 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparkStreamConfig:
    """One controlled ingestion stream for a pinned Spark release."""

    name: str
    doc_type: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    language: str
    chunking: str
    content_requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class SparkSourceConfig:
    """An immutable Spark release pinned by repository ref and commit."""

    name: str
    repository: str
    ref: str
    commit: str
    license: str
    streams: tuple[SparkStreamConfig, ...]


_VALID_DOC_TYPES = frozenset({"guide", "api_reference", "code_example", "sql_function_ref"})
_VALID_CHUNKING = frozenset({"header_aware", "api", "code"})


def load_spark_source_config(path: Path) -> SparkSourceConfig:
    """Load and validate a Spark source configuration from a JSON file.

    Raises ``ValueError`` with a field path in the message on any invalid value.
    """
    import re

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("spark source config must be a JSON object")

    def _require_str(obj: dict, field: str) -> str:
        value = obj.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"spark source config `{field}` must be a non-empty string")
        return value.strip()

    name = _require_str(raw, "name")
    repository = _require_str(raw, "repository")
    ref = _require_str(raw, "ref")
    commit = _require_str(raw, "commit")
    license_name = _require_str(raw, "license")

    if not (repository.startswith("https://") and (repository.endswith(".git") or "github.com" in repository)):
        raise ValueError("spark source config `repository` must be an HTTPS GitHub repository")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("spark source config `commit` must be a 40-character hexadecimal SHA")

    raw_streams = raw.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise ValueError("spark source config `streams` must be a non-empty list")

    streams: list[SparkStreamConfig] = []
    seen_names: set[str] = set()
    for idx, raw_stream in enumerate(raw_streams):
        if not isinstance(raw_stream, dict):
            raise ValueError(f"spark source config `streams[{idx}]` must be an object")
        stream_name = _require_str(raw_stream, "name")
        if stream_name in seen_names:
            raise ValueError(f"spark source config `streams[{idx}].name` must be unique: {stream_name!r}")
        seen_names.add(stream_name)

        doc_type = _require_str(raw_stream, "doc_type")
        if doc_type not in _VALID_DOC_TYPES:
            raise ValueError(f"spark source config `streams[{idx}].doc_type` must be one of {sorted(_VALID_DOC_TYPES)}")

        chunking = _require_str(raw_stream, "chunking")
        if chunking not in _VALID_CHUNKING:
            raise ValueError(f"spark source config `streams[{idx}].chunking` must be one of {sorted(_VALID_CHUNKING)}")

        language = _require_str(raw_stream, "language")
        include = _optional_string_tuple(raw_stream, "include", idx)
        exclude = _optional_string_tuple(raw_stream, "exclude", idx)
        content_requires = _optional_string_tuple(raw_stream, "content_requires", idx)
        streams.append(
            SparkStreamConfig(
                name=stream_name,
                doc_type=doc_type,
                include=include,
                exclude=exclude,
                language=language,
                chunking=chunking,
                content_requires=content_requires,
            )
        )

    return SparkSourceConfig(
        name=name,
        repository=repository,
        ref=ref,
        commit=commit,
        license=license_name,
        streams=tuple(streams),
    )


# ---------------------------------------------------------------------------
# Pinned multi-source configuration (Phase 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinnedStreamConfig:
    """One controlled ingestion stream for a pinned GitHub source release."""

    name: str
    doc_type: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    language: str
    chunking: str
    content_requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class PinnedSourceConfig:
    """An immutable pinned source: a GitHub repo release or an llms.txt index.

    ``type`` is ``"github"`` (uses ``repository``/``ref``/``commit``/``streams``),
    ``"url_index"`` (uses ``index_url``/``url_prefix``/``cache_dir``/``doc_type``),
    or ``"local_mirror"`` (uses ``mirror_dir``/``url_prefix``/``doc_type`` —
    resolves against a local git mirror instead of the network).
    ``slug`` is a short stable identifier used in generation IDs and CLI source
    selection.
    """

    type: str
    name: str
    slug: str
    version: str
    license: str = ""
    repository: str = ""
    ref: str = ""
    commit: str = ""
    streams: tuple[PinnedStreamConfig, ...] = ()
    index_url: str = ""
    url_prefix: str = ""
    base_url: str = ""
    cache_dir: str = ""
    mirror_dir: str = ""
    doc_type: str = "guide"


_VALID_PINNED_SOURCE_TYPES = frozenset({"github", "url_index", "local_mirror"})


def load_pinned_sources(path: Path) -> tuple[PinnedSourceConfig, ...]:
    """Load and validate pinned sources from a JSON file.

    Raises ``ValueError`` with a field path in the message on any invalid value.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pinned source config must be a JSON object")

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("pinned source config `sources` must be a non-empty list")

    sources: list[PinnedSourceConfig] = []
    seen_slugs: set[str] = set()
    seen_names: set[str] = set()
    for idx, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(f"pinned source config `sources[{idx}]` must be an object")

        def _require_str(obj: dict, field: str, _idx: int = idx) -> str:
            value = obj.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"pinned source config `sources[{_idx}].{field}` must be a non-empty string")
            return value.strip()

        source_type = _require_str(raw_source, "type")
        if source_type not in _VALID_PINNED_SOURCE_TYPES:
            raise ValueError(
                f"pinned source config `sources[{idx}].type` must be one of {sorted(_VALID_PINNED_SOURCE_TYPES)}"
            )
        name = _require_str(raw_source, "name")
        if name in seen_names:
            raise ValueError(f"pinned source config `sources[{idx}].name` must be unique: {name!r}")
        seen_names.add(name)
        slug = _require_str(raw_source, "slug")
        if slug in seen_slugs:
            raise ValueError(f"pinned source config `sources[{idx}].slug` must be unique: {slug!r}")
        seen_slugs.add(slug)
        version = _require_str(raw_source, "version")

        if source_type == "github":
            repository = _require_str(raw_source, "repository")
            ref = _require_str(raw_source, "ref")
            commit = _require_str(raw_source, "commit")
            license_name = _require_str(raw_source, "license")
            if not (repository.startswith("https://") and (repository.endswith(".git") or "github.com" in repository)):
                raise ValueError(f"pinned source config `sources[{idx}].repository` must be an HTTPS GitHub repository")

            if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
                raise ValueError(f"pinned source config `sources[{idx}].commit` must be a 40-character hexadecimal SHA")

            raw_streams = raw_source.get("streams")
            if not isinstance(raw_streams, list) or not raw_streams:
                raise ValueError(f"pinned source config `sources[{idx}].streams` must be a non-empty list")
            streams: list[PinnedStreamConfig] = []
            seen_streams: set[str] = set()
            for s_idx, raw_stream in enumerate(raw_streams):
                if not isinstance(raw_stream, dict):
                    raise ValueError(f"pinned source config `sources[{idx}].streams[{s_idx}]` must be an object")
                stream_name = _require_str(raw_stream, "name")
                if stream_name in seen_streams:
                    raise ValueError(
                        f"pinned source config `sources[{idx}].streams[{s_idx}].name` must be unique: {stream_name!r}"
                    )
                seen_streams.add(stream_name)
                doc_type = _require_str(raw_stream, "doc_type")
                if doc_type not in _VALID_DOC_TYPES:
                    raise ValueError(
                        f"pinned source config `sources[{idx}].streams[{s_idx}].doc_type` "
                        f"must be one of {sorted(_VALID_DOC_TYPES)}"
                    )
                chunking = _require_str(raw_stream, "chunking")
                if chunking not in _VALID_CHUNKING:
                    raise ValueError(
                        f"pinned source config `sources[{idx}].streams[{s_idx}].chunking` "
                        f"must be one of {sorted(_VALID_CHUNKING)}"
                    )
                language = _require_str(raw_stream, "language")
                streams.append(
                    PinnedStreamConfig(
                        name=stream_name,
                        doc_type=doc_type,
                        include=_optional_string_tuple(raw_stream, "include", s_idx),
                        exclude=_optional_string_tuple(raw_stream, "exclude", s_idx),
                        language=language,
                        chunking=chunking,
                        content_requires=_optional_string_tuple(raw_stream, "content_requires", s_idx),
                    )
                )
            sources.append(
                PinnedSourceConfig(
                    type=source_type,
                    name=name,
                    slug=slug,
                    version=version,
                    license=license_name,
                    repository=repository,
                    ref=ref,
                    commit=commit,
                    streams=tuple(streams),
                )
            )
        else:
            url_prefix = _require_str(raw_source, "url_prefix")
            doc_type = _require_str(raw_source, "doc_type")
            if doc_type not in _VALID_DOC_TYPES:
                raise ValueError(
                    f"pinned source config `sources[{idx}].doc_type` must be one of {sorted(_VALID_DOC_TYPES)}"
                )
            if source_type == "url_index":
                sources.append(
                    PinnedSourceConfig(
                        type=source_type,
                        name=name,
                        slug=slug,
                        version=version,
                        index_url=_require_str(raw_source, "index_url"),
                        url_prefix=url_prefix,
                        base_url=_require_str(raw_source, "base_url"),
                        cache_dir=_require_str(raw_source, "cache_dir"),
                        doc_type=doc_type,
                    )
                )
            else:
                mirror_dir = _require_str(raw_source, "mirror_dir")
                commit = _require_str(raw_source, "commit")
                if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
                    raise ValueError(
                        f"pinned source config `sources[{idx}].commit` must be a 40-character hexadecimal SHA"
                    )
                license_name = _require_str(raw_source, "license")
                sources.append(
                    PinnedSourceConfig(
                        type=source_type,
                        name=name,
                        slug=slug,
                        version=version,
                        license=license_name,
                        commit=commit,
                        url_prefix=url_prefix,
                        base_url=_optional_string(raw_source, "base_url"),
                        mirror_dir=mirror_dir,
                        doc_type=doc_type,
                    )
                )

    return tuple(sources)


# ---------------------------------------------------------------------------
# Spark rendered documentation configuration (Phase 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparkRenderedBuildConfig:
    """One locally rendered Spark documentation build."""

    name: str
    doc_type: str
    language: str
    working_dir: str
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    output_root: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    content_root_selector: str
    excluded_selectors: tuple[str, ...]
    canonical_url: str
    renderer: str


@dataclass(frozen=True)
class SparkRenderedSourceConfig:
    """An immutable Spark rendered documentation configuration.

    Shares the pinned release identity with ``SparkSourceConfig`` so native and
    rendered builds always come from the same commit.
    """

    name: str
    repository: str
    ref: str
    commit: str
    license: str
    builds: tuple[SparkRenderedBuildConfig, ...]


def load_spark_rendered_source_config(path: Path) -> SparkRenderedSourceConfig:
    """Load and validate a Spark rendered documentation configuration.

    Raises ``ValueError`` with a field path in the message on any invalid value.
    """
    import re

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("spark rendered config must be a JSON object")

    def _require_str(obj: dict, field: str) -> str:
        value = obj.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"spark rendered config `{field}` must be a non-empty string")
        return value.strip()

    name = _require_str(raw, "name")
    repository = _require_str(raw, "repository")
    ref = _require_str(raw, "ref")
    commit = _require_str(raw, "commit")
    license_name = _require_str(raw, "license")

    if not (repository.startswith("https://") and (repository.endswith(".git") or "github.com" in repository)):
        raise ValueError("spark rendered config `repository` must be an HTTPS GitHub repository")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("spark rendered config `commit` must be a 40-character hexadecimal SHA")

    raw_builds = raw.get("builds")
    if not isinstance(raw_builds, list) or not raw_builds:
        raise ValueError("spark rendered config `builds` must be a non-empty list")

    builds: list[SparkRenderedBuildConfig] = []
    seen_names: set[str] = set()
    for idx, raw_build in enumerate(raw_builds):
        if not isinstance(raw_build, dict):
            raise ValueError(f"spark rendered config `builds[{idx}]` must be an object")
        build_name = _require_str(raw_build, "name")
        if build_name in seen_names:
            raise ValueError(f"spark rendered config `builds[{idx}].name` must be unique: {build_name!r}")
        seen_names.add(build_name)

        doc_type = _require_str(raw_build, "doc_type")
        if doc_type not in _VALID_DOC_TYPES:
            raise ValueError(
                f"spark rendered config `builds[{idx}].doc_type` must be one of {sorted(_VALID_DOC_TYPES)}"
            )

        command = _optional_string_tuple(raw_build, "command", idx)
        if not command:
            raise ValueError(f"spark rendered config `builds[{idx}].command` must be a non-empty list")
        env_raw = raw_build.get("env", {})
        if not isinstance(env_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
        ):
            raise ValueError(f"spark rendered config `builds[{idx}].env` must be a dict of strings")
        include = _optional_string_tuple(raw_build, "include", idx)
        exclude = _optional_string_tuple(raw_build, "exclude", idx)
        excluded_selectors = _optional_string_tuple(raw_build, "excluded_selectors", idx)
        if not include:
            raise ValueError(f"spark rendered config `builds[{idx}].include` must be a non-empty list")

        builds.append(
            SparkRenderedBuildConfig(
                name=build_name,
                doc_type=doc_type,
                language=_require_str(raw_build, "language"),
                working_dir=_require_str(raw_build, "working_dir"),
                command=command,
                env=tuple(sorted(env_raw.items())),
                output_root=_require_str(raw_build, "output_root"),
                include=include,
                exclude=exclude,
                content_root_selector=_require_str(raw_build, "content_root_selector"),
                excluded_selectors=excluded_selectors,
                canonical_url=_require_str(raw_build, "canonical_url"),
                renderer=_require_str(raw_build, "renderer"),
            )
        )

    return SparkRenderedSourceConfig(
        name=name,
        repository=repository,
        ref=ref,
        commit=commit,
        license=license_name,
        builds=tuple(builds),
    )


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
        populate_by_name=True,
        env_file=(".env", ".env.secrets", ".env.local"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Module-load seam ONLY (see module-level `settings` below).  Never set by tests.
    skip_provider_check: bool = Field(default=False, exclude=True)

    project_root: Path = PROJECT_ROOT
    documentation_sources_path: Path = (
        PROJECT_ROOT / "data_engineering_copilot" / "config" / "documentation_sources.json"
    )
    spark_sources_path: Path = PROJECT_ROOT / "data_engineering_copilot" / "config" / "spark_sources.json"
    spark_rendered_sources_path: Path = (
        PROJECT_ROOT / "data_engineering_copilot" / "config" / "spark_rendered_sources.json"
    )
    pinned_sources_path: Path = PROJECT_ROOT / "data_engineering_copilot" / "config" / "pinned_sources.json"
    spark_cache_dir: Path = PROJECT_ROOT / "data" / "spark_src"
    spark_corpus_dir: Path = PROJECT_ROOT / "data" / "spark_corpus"
    pinned_cache_dir: Path = PROJECT_ROOT / "data" / "pinned_src"
    pinned_corpus_dir: Path = PROJECT_ROOT / "data" / "pinned_corpus"
    # Local git mirror of Claude docs (populated by ``scripts/mirror_claude_docs.py``).
    claude_docs_mirror_dir: Path = PROJECT_ROOT / "data" / "claude_docs_mirror"
    index_state_dir: Path = PROJECT_ROOT / ".index_state"
    active_collection_alias: str = "data_engineering_docs"
    collection_name: str = "data_engineering_docs"
    # Active generation/collection for Spark-built indexes. When empty, the
    # legacy `collection_name` is used. Validated against Qdrant-safe chars.
    active_index_generation: str = ""
    active_collection_name: str = ""

    # URLs accessed from localhost
    qdrant_url: str = "http://localhost:6333"
    ollama_base_url: str = "http://localhost:11434"
    # Separate Ollama instances for embedding vs LLM to prevent resource contention.
    # Default to the same value; set to different ports for dedicated instances.
    embedding_ollama_base_url: str = ""
    llm_ollama_base_url: str = ""

    # Ollama Cloud (LLM only). Same Ollama protocol as local, but hosted at
    # ollama.com. Requires OLLAMA_API_KEY. Separate provider so local Ollama
    # remains the degraded fallback even when cloud is unreachable.
    ollama_cloud_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("OLLAMA_CLOUD_API_KEY", "OLLAMA_API_KEY"),
    )
    ollama_cloud_base_url: str = "https://ollama.com"
    ollama_cloud_model: str = "gpt-oss:20b"
    ollama_cloud_rpm_limit: int = 30
    ollama_cloud_rpd_limit: int = 1000

    # URLs accessed within docker
    redis_url: str = "redis://:local_secure_password_123@localhost:6379/0"
    langfuse_url: str = "http://langfuse:3000"
    langfuse_public_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGFUSE_SECRET_KEY",
    )
    langfuse_host: str = Field(
        default="http://langfuse:3000",
        validation_alias="LANGFUSE_HOST",
    )
    langfuse_enabled: bool = Field(default=True, validation_alias="LANGFUSE_ENABLED")
    langfuse_sample_rate: float = Field(default=1.0, validation_alias="LANGFUSE_SAMPLE_RATE")
    langfuse_environment: str = Field(
        default="development",
        validation_alias="LANGFUSE_TRACING_ENVIRONMENT",
    )
    image_git_sha: str = Field(default="unknown", validation_alias="IMAGE_GIT_SHA")

    embedding_batch_size: int = 64
    embed_concurrency: int = 1
    enrichment_batch_size: int = 20

    # LLM Provider selection: "ollama" | "openrouter"
    # Embedding provider selection: "ollama" | "openrouter"
    embedding_model_name: str = "nomic-embed-text"
    # Local HuggingFace embedding model (provider "local-hf"): runs
    # sentence-transformers on the local CPU, mirroring the reranker. Produces
    # vectors identical to the hosted NVIDIA nemotron-3-embed-1b (verified cos
    # ~1.0) with zero provider dependence.
    local_hf_embedding_model: str = "nvidia/Nemotron-3-Embed-1B-BF16"
    # Embedding dimension is model-dependent, not provider-dependent.
    # Map model names to their known output dimensions.
    embedding_model_dimensions: dict[str, int] = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "snowflake-arctic-embed2": 1024,
        "llama3.2:3b": 3072,
        "nvidia/nemotron-3-embed-1b": 2048,
        "nvidia/nemotron-3-embed-1b:free": 2048,
        "nvidia/Nemotron-3-Embed-1B-BF16": 2048,
        "text-embedding-004": 768,
    }
    default_embedding_dimension: int = 768
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    embedding_provider: str = "ollama"
    ollama_model: str = "llama3.2:3b"

    # Max output tokens for LLM calls, sent as ``max_tokens`` (or
    # ``max_completion_tokens`` where the provider requires it). Omitting it is
    # NOT safe: several providers silently truncate (Cloudflare defaults to
    # 256, NVIDIA to 1024) while Groq/Cerebras/Gemini/OpenRouter run unbounded.
    llm_max_tokens: int = 2048
    # Per-purpose output caps — short-output purposes (rewrite/intent/
    # groundedness) get small budgets, long-output ones (answer/code) get more.
    purpose_max_tokens: dict[str, int] = {
        "answer": 4096,
        "code": 4096,
        "enrichment": 1536,
        "evaluation": 1536,
        "groundedness": 1024,
        "rewrite": 768,
        "intent": 768,
        "global": 2048,
    }

    # API key for the API auth middleware (pydantic-set, env-file aware).
    api_key: SecretStr = SecretStr("")

    # OpenRouter settings (LLM + Embeddings)
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_model: str = "openrouter/free"
    openrouter_embedding_model: str = "nvidia/nemotron-3-embed-1b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_rpm_limit: int = 18
    openrouter_rpd_limit: int = 900

    # NVIDIA settings (LLM + Embeddings)
    nvidia_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
    )
    nvidia_model: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_embedding_model: str = "nvidia/nemotron-3-embed-1b"
    nvidia_rpm_limit: int = 36
    # Free Developer tier is 40 RPM / 1000 RPD (per-key; chat + embeddings share
    # the same daily budget). 0 disables the daily gate entirely.
    nvidia_rpd_limit: int = Field(
        default=1000,
        validation_alias=AliasChoices("NVIDIA_RPD_LIMIT", "NVIDIA_NIM_RPD_LIMIT"),
    )

    # Groq settings (LLM only)
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_rpm_limit: int = 27
    groq_rpd_limit: int = 13000

    # Cerebras settings (LLM only)
    cerebras_api_key: SecretStr = SecretStr("")
    cerebras_model: str = "gemma-4-31b"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_rpm_limit: int = 4
    cerebras_rpd_limit: int = 2200

    # Gemini settings (LLM + Embeddings)
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_embedding_model: str = "text-embedding-001"
    gemini_rpm_limit: int = 13
    gemini_rpd_limit: int = 450

    # Cloudflare Workers AI settings (LLM only)
    cloudflare_api_key: SecretStr = SecretStr("")
    cloudflare_model: str = ""
    cloudflare_account_id: str = ""
    cloudflare_base_url: str = ""
    cloudflare_rpm_limit: int = 60
    cloudflare_rpd_limit: int = 1000

    # OpenCode Zen free models (LLM only). Free models (e.g.
    # ``deepseek-v4-flash-free``) are only served on the Zen pay-as-you-go
    # surface (``/zen/v1``), not the Go subscription surface (``/zen/go/v1``).
    # Free-tier requests are throttled via ``FreeUsageLimitError`` (429) when
    # the per-key quota is exhausted.
    opencodezen_api_key: SecretStr = SecretStr("")
    opencodezen_model: str = "deepseek-v4-flash-free"
    opencodezen_base_url: str = "https://opencode.ai/zen/v1"
    opencodezen_rpm_limit: int = 60
    opencodezen_rpd_limit: int = 5000

    # OpenCode Go subscription models (LLM only). Served on the Go surface
    # (``/zen/go/v1``) and requires an OpenCode Go subscription; the ``-free``
    # Zen models are NOT available here.
    opencodego_api_key: SecretStr = SecretStr("")
    opencodego_model: str = "deepseek-v4-flash"
    opencodego_base_url: str = "https://opencode.ai/zen/go/v1"
    opencodego_rpm_limit: int = 60
    opencodego_rpd_limit: int = 1000

    # SambaNova Cloud settings (LLM only). OpenAI-compatible
    # ``/v1/chat/completions`` surface. Meta-Llama-3.3-70B-Instruct is the
    # default (fast, non-reasoning); gpt-oss-120b, MiniMax-M2.7 and
    # DeepSeek-V3.1 are also hosted (gpt-oss-120b is a reasoning model and
    # needs a larger max_tokens budget to emit its final answer).
    sambanova_api_key: SecretStr = SecretStr("")
    sambanova_model: str = "Meta-Llama-3.3-70B-Instruct"
    sambanova_base_url: str = "https://api.sambanova.ai/v1"
    sambanova_rpm_limit: int = 30
    sambanova_rpd_limit: int = 1000

    # Mistral AI settings (LLM only). OpenAI-compatible
    # ``/v1/chat/completions`` surface. Free tier includes $10/mo API
    # credits; models are priced per million tokens (input + output).
    mistral_api_key: SecretStr = SecretStr("")
    mistral_model: str = "mistral-small-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_rpm_limit: int = 30
    mistral_rpd_limit: int = 1000

    # DeepSeek (LLM only). OpenAI-compatible ``/v1/chat/completions`` surface.
    # Very cheap pay-as-you-go ($0.14/M input for V4 Flash); concurrency-based
    # rate limits (2500 Flash, 500 Pro). No permanently free tier.
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_rpm_limit: int = 60
    deepseek_rpd_limit: int = 1000

    # Z.AI / Zhipu AI (LLM only). OpenAI-compatible surface. GLM-4.5-Flash is
    # free forever ($0 input/output); GLM-5.2 is the paid flagship.
    # Rate limits are undocumented (~60 RPM, ~1K RPD from 3rd-party reports).
    zai_api_key: SecretStr = SecretStr("")
    zai_model: str = "glm-4.7-flash"
    zai_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zai_rpm_limit: int = 60
    zai_rpd_limit: int = 1000

    # SiliconFlow (LLM only). OpenAI-compatible multi-model inference platform.
    # $1 starter credit + permanently free models (Qwen3-8B, DeepSeek-R1-Distill).
    # Use global (.com) endpoint — .cn is China-only.
    siliconflow_api_key: SecretStr = SecretStr("")
    siliconflow_model: str = "Qwen/Qwen3-8B"
    siliconflow_base_url: str = "https://api.siliconflow.com/v1"
    siliconflow_rpm_limit: int = 100
    siliconflow_rpd_limit: int = 1000

    # Together AI (LLM only). OpenAI-compatible surface. 50+ models (Llama,
    # DeepSeek, Qwen). $1 starter credit; dynamic rate limits scale with usage.
    together_api_key: SecretStr = SecretStr("")
    together_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    together_base_url: str = "https://api.together.xyz/v1"
    together_rpm_limit: int = 600
    together_rpd_limit: int = 10000

    # Fireworks AI (LLM only). OpenAI-compatible surface. 50+ models.
    # $1 starter credit; 10 RPM without payment method.
    fireworks_api_key: SecretStr = SecretStr("")
    fireworks_model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct"
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_rpm_limit: int = 10
    fireworks_rpd_limit: int = 500

    # LLM7.io (LLM only). OpenAI-compatible aggregator. Free 30 RPM (120 with
    # token registration); 1M tokens/day. Small independent provider.
    llm7_api_key: SecretStr = SecretStr("")
    llm7_model: str = "default"
    llm7_base_url: str = "https://api.llm7.io/v1"
    llm7_rpm_limit: int = 120
    llm7_rpd_limit: int = 1000

    # Agnes AI (LLM only). OpenAI-compatible surface. Free 20 RPM text;
    # agnes-2.5-flash with 512K context. Singapore-based, multimodal focus.
    agnes_api_key: SecretStr = SecretStr("")
    agnes_model: str = "agnes-2.5-flash"
    agnes_base_url: str = "https://apihub.agnes-ai.com/v1"
    agnes_rpm_limit: int = 20
    agnes_rpd_limit: int = 500

    # Helyx AI (LLM only). OpenAI-compatible aggregator. Free 100K tokens/day,
    # 50+ models (Llama 4, Qwen 3.6, DeepSeek V4, Mistral). Aggregator.
    helyx_api_key: SecretStr = SecretStr("")
    helyx_model: str = "deepseek-chat"
    helyx_base_url: str = "https://helyxai.space/v1"
    helyx_rpm_limit: int = 30
    helyx_rpd_limit: int = 1000

    # AnyAPI.ai (LLM only). OpenAI-compatible aggregator. Free 20 RPM,
    # BYOK proxy with zero markup. Aggregator.
    anyapi_api_key: SecretStr = SecretStr("")
    anyapi_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    anyapi_base_url: str = "https://api.anyapi.ai/v1"
    anyapi_rpm_limit: int = 20
    anyapi_rpd_limit: int = 500

    # Hugging Face Inference Providers (LLM + Embeddings).
    # Serverless embeddings use the native ``feature-extraction`` route (the
    # OpenAI-compatible ``/v1`` surface is chat-completions only). Free-tier
    # credits apply when routed through ``router.huggingface.co``.
    huggingface_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("HUGGINGFACE_API_KEY", "HF_TOKEN"),
    )
    huggingface_embedding_model: str = "nvidia/Nemotron-3-Embed-1B-BF16"
    huggingface_base_url: str = "https://router.huggingface.co/hf-inference"
    # Free-tier rate budget: ~270 requests/hour (~900/day per model). The
    # sliding-window limiter is per-minute, so rpm 4 caps sustained throughput
    # at ~240/hr — safely under the hourly limit while the rpd 900 is the hard
    # daily gate (fail over to the next provider when either is exhausted).
    huggingface_rpm_limit: int = 4
    huggingface_rpd_limit: int = 900

    # LLM fallback chain: ordered list of providers to try on failure
    llm_fallback_order: list[str] = Field(
        default_factory=lambda: [
            "cloudflare",
            "groq",
            "nvidia",
            "gemini",
            "cerebras",
            "sambanova",
            "mistral",
            "zai",
            "llm7",
            "agnes",
            "anyapi",
            "ollama_cloud",
            "ollama",
        ]
    )
    llm_fallback_call_timeout: int = 30  # per-attempt timeout for non-primary fallback providers

    # Embedding fallback chain: ordered list of embedding providers to try on failure
    embedding_fallback_order: list[str] = Field(default_factory=lambda: ["nvidia", "openrouter", "ollama"])

    # Provider cooldown / routing
    provider_cooldown_seconds: int = 60
    health_success_rate_weight: float = 0.6
    health_latency_weight: float = 0.2
    health_recency_weight: float = 0.2
    health_consecutive_failure_penalty: float = 0.3

    # Availability-aware priority router
    # When enabled, provider cooldowns + the cached best provider are shared
    # across processes via Redis (falls back to in-memory if Redis is down).
    router_redis_sharing: bool = True
    # All-down wait policy: keep waiting (sleeping at most
    # router_wait_max_seconds per iteration) until at least this fraction of
    # external providers have exited cooldown, up to router_deadline_seconds,
    # then degrade to the local fallback.
    router_wait_min_available_fraction: float = 0.5
    router_wait_max_seconds: float = 15.0
    router_deadline_seconds: float = 45.0
    # How long the cached "best provider" stays authoritative per purpose.
    router_best_cache_ttl_seconds: float = 15.0
    # Bonus score applied to the purpose-pinned provider so it is preferred
    # (not hard-pinned) when its recent health is competitive.
    router_purpose_preference_weight: float = 0.15

    # After this many consecutive failures the degraded Ollama fallback is
    # skipped (fail fast) instead of stalling the request on a broken local
    # model. A single success resets the counter.
    ollama_degraded_max_consecutive_failures: int = 2

    # Per-purpose LLM overrides (empty = use global llm_provider / llm_model)
    answer_llm_provider: str = "openrouter"
    answer_llm_model: str = "openrouter/free"
    rewrite_llm_provider: str = "groq"
    rewrite_llm_model: str = ""
    groundedness_llm_provider: str = "groq"
    groundedness_llm_model: str = ""
    intent_llm_provider: str = "groq"
    intent_llm_model: str = ""
    enrichment_llm_provider: str = ""
    enrichment_llm_model: str = ""
    evaluation_llm_provider: str = "groq"
    evaluation_llm_model: str = ""
    # Code-specific LLM override (optional, for code_example/api_lookup intents)
    code_llm_provider: str = ""
    code_llm_model: str = ""

    # Per-purpose embedding overrides (empty = use global embedding_provider / embedding_fallback_order)
    enrichment_embedding_provider: str = ""
    evaluation_embedding_provider: str = ""

    # Per-provider purpose model overrides (override model per-provider per-purpose)
    # When set, these take priority over the generic {purpose}_llm_model above when
    # the effective provider matches. Format: {provider}_{purpose}_llm_model.
    openrouter_answer_llm_model: str = ""
    openrouter_rewrite_llm_model: str = ""
    openrouter_groundedness_llm_model: str = ""
    openrouter_intent_llm_model: str = ""
    openrouter_enrichment_llm_model: str = ""
    openrouter_evaluation_llm_model: str = ""
    openrouter_code_llm_model: str = ""
    nvidia_answer_llm_model: str = ""
    nvidia_rewrite_llm_model: str = ""
    nvidia_groundedness_llm_model: str = ""
    nvidia_intent_llm_model: str = ""
    nvidia_enrichment_llm_model: str = ""
    nvidia_evaluation_llm_model: str = ""
    nvidia_code_llm_model: str = ""
    groq_answer_llm_model: str = ""
    groq_rewrite_llm_model: str = ""
    groq_groundedness_llm_model: str = ""
    groq_intent_llm_model: str = ""
    groq_enrichment_llm_model: str = ""
    groq_evaluation_llm_model: str = ""
    groq_code_llm_model: str = ""
    cerebras_answer_llm_model: str = ""
    cerebras_rewrite_llm_model: str = ""
    cerebras_groundedness_llm_model: str = ""
    cerebras_intent_llm_model: str = ""
    cerebras_enrichment_llm_model: str = ""
    cerebras_evaluation_llm_model: str = ""
    cerebras_code_llm_model: str = ""
    gemini_answer_llm_model: str = ""
    gemini_rewrite_llm_model: str = ""
    gemini_groundedness_llm_model: str = ""
    gemini_intent_llm_model: str = ""
    gemini_enrichment_llm_model: str = ""
    gemini_evaluation_llm_model: str = ""
    gemini_code_llm_model: str = ""
    cloudflare_answer_llm_model: str = ""
    cloudflare_rewrite_llm_model: str = ""
    cloudflare_groundedness_llm_model: str = ""
    cloudflare_intent_llm_model: str = ""
    cloudflare_enrichment_llm_model: str = ""
    cloudflare_evaluation_llm_model: str = ""
    cloudflare_code_llm_model: str = ""
    opencodezen_answer_llm_model: str = ""
    opencodezen_rewrite_llm_model: str = ""
    opencodezen_groundedness_llm_model: str = ""
    opencodezen_intent_llm_model: str = ""
    opencodezen_enrichment_llm_model: str = ""
    opencodezen_evaluation_llm_model: str = ""
    opencodezen_code_llm_model: str = ""
    opencodego_answer_llm_model: str = ""
    opencodego_rewrite_llm_model: str = ""
    opencodego_groundedness_llm_model: str = ""
    opencodego_intent_llm_model: str = ""
    opencodego_enrichment_llm_model: str = ""
    opencodego_evaluation_llm_model: str = ""
    opencodego_code_llm_model: str = ""
    sambanova_answer_llm_model: str = ""
    sambanova_rewrite_llm_model: str = ""
    sambanova_groundedness_llm_model: str = ""
    sambanova_intent_llm_model: str = ""
    sambanova_enrichment_llm_model: str = ""
    sambanova_evaluation_llm_model: str = ""
    sambanova_code_llm_model: str = ""
    mistral_answer_llm_model: str = ""
    mistral_rewrite_llm_model: str = ""
    mistral_groundedness_llm_model: str = ""
    mistral_intent_llm_model: str = ""
    mistral_enrichment_llm_model: str = ""
    mistral_evaluation_llm_model: str = ""
    mistral_code_llm_model: str = ""
    deepseek_answer_llm_model: str = ""
    deepseek_rewrite_llm_model: str = ""
    deepseek_groundedness_llm_model: str = ""
    deepseek_intent_llm_model: str = ""
    deepseek_enrichment_llm_model: str = ""
    deepseek_evaluation_llm_model: str = ""
    deepseek_code_llm_model: str = ""
    zai_answer_llm_model: str = ""
    zai_rewrite_llm_model: str = ""
    zai_groundedness_llm_model: str = ""
    zai_intent_llm_model: str = ""
    zai_enrichment_llm_model: str = ""
    zai_evaluation_llm_model: str = ""
    zai_code_llm_model: str = ""
    siliconflow_answer_llm_model: str = ""
    siliconflow_rewrite_llm_model: str = ""
    siliconflow_groundedness_llm_model: str = ""
    siliconflow_intent_llm_model: str = ""
    siliconflow_enrichment_llm_model: str = ""
    siliconflow_evaluation_llm_model: str = ""
    siliconflow_code_llm_model: str = ""
    together_answer_llm_model: str = ""
    together_rewrite_llm_model: str = ""
    together_groundedness_llm_model: str = ""
    together_intent_llm_model: str = ""
    together_enrichment_llm_model: str = ""
    together_evaluation_llm_model: str = ""
    together_code_llm_model: str = ""
    fireworks_answer_llm_model: str = ""
    fireworks_rewrite_llm_model: str = ""
    fireworks_groundedness_llm_model: str = ""
    fireworks_intent_llm_model: str = ""
    fireworks_enrichment_llm_model: str = ""
    fireworks_evaluation_llm_model: str = ""
    fireworks_code_llm_model: str = ""
    llm7_answer_llm_model: str = ""
    llm7_rewrite_llm_model: str = ""
    llm7_groundedness_llm_model: str = ""
    llm7_intent_llm_model: str = ""
    llm7_enrichment_llm_model: str = ""
    llm7_evaluation_llm_model: str = ""
    llm7_code_llm_model: str = ""
    agnes_answer_llm_model: str = ""
    agnes_rewrite_llm_model: str = ""
    agnes_groundedness_llm_model: str = ""
    agnes_intent_llm_model: str = ""
    agnes_enrichment_llm_model: str = ""
    agnes_evaluation_llm_model: str = ""
    agnes_code_llm_model: str = ""
    ollama_cloud_answer_llm_model: str = ""
    ollama_cloud_rewrite_llm_model: str = ""
    ollama_cloud_groundedness_llm_model: str = ""
    ollama_cloud_intent_llm_model: str = ""
    ollama_cloud_enrichment_llm_model: str = ""
    ollama_cloud_evaluation_llm_model: str = ""
    ollama_cloud_code_llm_model: str = ""
    helyx_answer_llm_model: str = ""
    helyx_rewrite_llm_model: str = ""
    helyx_groundedness_llm_model: str = ""
    helyx_intent_llm_model: str = ""
    helyx_enrichment_llm_model: str = ""
    helyx_evaluation_llm_model: str = ""
    helyx_code_llm_model: str = ""
    anyapi_answer_llm_model: str = ""
    anyapi_rewrite_llm_model: str = ""
    anyapi_groundedness_llm_model: str = ""
    anyapi_intent_llm_model: str = ""
    anyapi_enrichment_llm_model: str = ""
    anyapi_evaluation_llm_model: str = ""
    anyapi_code_llm_model: str = ""

    ollama_code_llm_model: str = "qwen2.5-coder:7b"

    # Chunking strategy: "fixed_size", "sentence_preserving", or "semantic"
    chunking_strategy: str = "sentence_preserving"
    chunk_size_words: int = 375
    chunk_overlap_words: int = 90
    # Semantic chunker specific settings
    min_semantic_similarity: float = 0.5
    max_chunk_words: int | None = None  # Auto: 1.5x chunk_size_words if None
    # Feature flags
    enable_semantic_chunking: bool = True  # Enable semantic chunker (requires embedding model)
    # Retrieve a broad candidate pool per query variant; reranking narrows it
    # to the final context set after dense+sparse rank fusion.
    retrieval_top_k: int = 50
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 30
    max_context_chars: int = 16000
    # Diversity cap on final context: at most this many chunks per distinct
    # source URL, applied after a one-per-source coverage guarantee. Keeps the
    # context compact (context-rot safe) while ensuring cross-source coverage.
    max_chunks_per_source: int = 2
    max_expansion_queries: int = 2
    context_compression_ratio: float = 0.8
    groundedness_threshold: float = 0.6
    confidence_threshold: float = 0.18
    # Cross-encoder sigmoid scores cluster lower than embedding/fused
    # confidence (relevant pairs commonly land ~0.10-0.15). When a reranker
    # ran for a query, the quality gate compares against this value; without a
    # reranker it falls back to ``confidence_threshold``.
    reranker_confidence_threshold: float = 0.10
    # LLM-based (cloud) reranking: when enabled, ``rerank_fallback_order`` is
    # tried before the local cross-encoder (which stays the degraded last
    # resort). All providers normalize scores to [0, 1] so the confidence gate
    # above keeps the same meaning across providers.
    llm_rerank_enabled: bool = True
    rerank_fallback_order: list[str] = Field(default_factory=lambda: ["openrouter", "nvidia", "huggingface"])
    openrouter_rerank_model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
    openrouter_rerank_url: str = "https://openrouter.ai/api/v1/rerank"
    nvidia_rerank_model: str = "nv-rerank-qa-mistral-4b:1"
    nvidia_rerank_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
    huggingface_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_cloud_timeout_seconds: int = 30
    request_timeout_seconds: int = 15
    ollama_timeout_seconds: int = 180
    ollama_connect_timeout_seconds: int = 5
    ollama_pool_timeout_seconds: int = 5
    ollama_keep_alive: str | int = "10m"
    # Native /api/chat params (num_ctx / keep_alive) are NOT honored on the
    # OpenAI-compat /v1/chat/completions endpoint — only ``max_tokens`` maps
    # to num_predict there. `num_ctx` / `keep_alive` are therefore only
    # applied when Ollama is driven through the native API.
    ollama_num_ctx: int = 4096
    # Used as ``max_tokens`` on the OpenAI-compat endpoint (enforced output cap
    # for the local model, which is CPU-bound and slow).
    ollama_num_predict: int = 512
    ollama_retry_context_ratio: float = 0.5
    ollama_retry_extra_num_predict: int = 512
    ollama_retry_max_num_predict: int = 1024
    crawl_delay_seconds: float = 0.3
    max_pages_per_source: int = 100000
    max_pages_hard_cap: int = 100000
    recovery_max_pages: int = 100000
    crawl_attempt_multiplier: int = 3
    crawl_min_attempts: int = 200
    crawl_thread_pool_size: int = 4
    ingestion_batch_chunk_size: int = 512
    processing_concurrency: int = 4  # ROLLBACK: If Ollama overloads at 4, change to 3
    enrichment_concurrency: int = 1
    # Multi-stage pipeline concurrency (isolated executor pools)
    parse_concurrency: int = 4
    chunk_concurrency: int = 4
    store_concurrency: int = 2
    # Async crawler settings
    crawl_db_url: str = ""
    # Conversational RAG (multi-turn chat). Chat sessions/messages are stored
    # durably in Postgres (chat_db_url, defaulting to crawl_db_url) with a Redis
    # hot cache for the recent turns.
    chat_enabled: bool = True
    chat_session_ttl_seconds: int = 259200  # 72h
    chat_history_max_turns: int = 10
    chat_history_max_tokens: int = 2048
    chat_db_url: str = ""
    chat_title_max_chars: int = 60
    # Chat speed tuning (Phase C): route short LLM steps (rewrite/hyde/expand/
    # scope-verify) to the local Ollama model. Default OFF: on CPU-only hosts
    # (no GPU) Ollama is ~6 tok/s, so medium-length rewrites/HyDE are SLOWER
    # than the cloud chain. Enable only on GPU-backed Ollama or when generating
    # very short outputs. The answer generation stays on the cloud chain unless
    # chat_answer_local.
    chat_rewrite_local: bool = False
    chat_scope_local: bool = False
    chat_answer_local: bool = False
    # Chat reranking: use the local cross-encoder instead of the cloud LLM
    # rerank chain. The cloud chain costs ~5s/turn; the local model
    # (bge-reranker-v2-m3) is free and near-instant on CPU. Default ON for chat;
    # the single-turn Ask pipeline is unaffected.
    chat_rerank_local: bool = True
    # Chat speed tuning (Phase F): smart-cache recall tier — reuse similar
    # cached (question→answer) pairs via local synthesis, gated by scope
    # verify. Opt-in default-off; flip on after measuring cache hit rate.
    chat_cache_recall_enabled: bool = False
    chat_cache_top_k: int = 3
    chat_cache_recall_threshold: float = 0.70
    chat_cache_max_age_seconds: int = 86400
    # Anti-hallucination / identity hardening.
    # URL substrings whose chunks must never be used as context (Claude's
    # self-identifying system prompts hijack the assistant's identity).
    chat_blocked_url_substrings: list[str] = field(default_factory=lambda: ["system-prompts.md"])
    # Optional source-name whitelist for chat retrieval (domain isolation).
    # Empty = all sources. When set, only these source_names are retrieved.
    chat_domain_sources: list[str] = field(default_factory=list)
    # Clickable follow-up suggestions (ChatGPT-style). After each assistant
    # answer the UI shows N suggested follow-up questions as chips; clicking one
    # submits it as the next turn. Mode: "llm" | "rule" | "hybrid" (LLM first,
    # rule-based fallback on failure/empty).
    chat_suggestions_enabled: bool = True
    chat_suggestions_count: int = 3
    chat_suggestions_mode: str = "hybrid"
    crawl_async_concurrency: int = 10
    crawl_async_max_concurrency: int = 40
    crawl_async_per_domain_concurrency: int = 2
    crawl_async_conditional_get: bool = True
    crawl_async_cache_url: str = ""
    crawl_async_thread_pool_size: int = 4
    # Maximum number of times a FAILED frontier URL is re-discovered before it
    # becomes terminal (stops being retried).
    frontier_max_attempts: int = 3
    logging_enabled: bool = True
    # Hybrid search
    hybrid_search_enabled: bool = True
    hybrid_rrf_k: int = 60
    # Identifier-aware hybrid profiles: when enabled, technical queries
    # (dotted identifiers, paths, versions, code/SQL) fuse hybrid retrieval with
    # weighted RRF (sparse 1.25 / dense 1.0) instead of equal weights. Off by
    # default (``equal_rrf``) until the benchmark gate (identifier recall
    # >= +0.05 with all global recall/MRR thresholds satisfied) passes.
    identifier_sparse_rrf_enabled: bool = False
    # Namespace-aware BM25 tokenizer (``namespace-v1``). Off by default until
    # the technical-query benchmark gate (identifier recall >= +0.05, generic
    # recall <= -0.01, MRR <= -0.02) passes. Enabling it invalidates every
    # legacy BM25 cache: a new generation must be built and activated.
    namespace_bm25_enabled: bool = False
    # Semantic cache
    semantic_cache_threshold: float = 0.95
    semantic_cache_ttl: int = 3600
    # Cache toggles (per-type enable/disable). `query_cache_enabled` is the
    # master switch for the two-tier RAG query cache; the exact/semantic flags
    # disable individual tiers. `embedding_cache_enabled` and
    # `crawl_cache_enabled` control the embedder and crawler caches.
    query_cache_enabled: bool = True
    query_cache_exact_enabled: bool = True
    query_cache_semantic_enabled: bool = True
    embedding_cache_enabled: bool = True
    crawl_cache_enabled: bool = True
    # Index generation identity and validation. `index_generation` is empty for
    # legacy operation; when set it identifies a reproducible corpus build.
    index_generation: str = ""
    index_require_hybrid: bool = True
    index_validation_min_points: int = 1
    # Query rewriting / grounding
    query_rewrite_enabled: bool = True
    # Deterministic HyDE policy (``default_hyde_policy``): suppress HyDE for
    # non-factual intents and for identifier/version-qualified, code, or
    # debugging queries. Off by default (HyDE runs as before) until the
    # benchmark gate (API/code/debugging provider-call reduction >= 20% within
    # fixed recall/MRR thresholds) passes.
    hyde_policy_enabled: bool = False
    groundedness_enabled: bool = True
    # Post-answer topic-scope gate (fail-open): refuses answers when the retrieved
    # context does not cover the question's topic, converting them to
    # INSUFFICIENT_CONTEXT. Reuses the groundedness-purpose LLM client.
    scope_check_enabled: bool = True
    intent_classification_llm_enabled: bool = False  # Enable LLM fallback for intent classification
    # Context management
    context_compression_enabled: bool = False
    max_context_tokens: int = 4096
    # Post-processing toggles
    contextual_enrichment_enabled: bool = True
    api_extraction_enabled: bool = True
    code_block_parsing_enabled: bool = True
    chunk_filtering_enabled: bool = True
    # PII redaction
    pii_redaction_enabled: bool = True
    pii_redaction_mode: str = "full"  # full | masked | none
    # Indirect prompt injection guard for retrieved documents
    input_guardrails_enabled: bool = True
    # RBAC (document-level access control)
    rbac_enabled: bool = False
    rbac_users_json: str = ""  # inline JSON string: {"key_prefix": {"allowed_sources": [...], "role": "reader"}}
    # Drift detection
    drift_detection_enabled: bool = True
    drift_eval_history_path: str = "data/eval_history.jsonl"
    drift_window_days: int = 7
    sources: tuple[DocumentationSource, ...] = ()

    @model_validator(mode="after")
    def _load_sources_from_json(self) -> AppSettings:
        if not self.sources:
            object.__setattr__(self, "sources", load_documentation_sources(self.documentation_sources_path))
        return self

    @model_validator(mode="after")
    def _validate_index_generation(self) -> AppSettings:
        import re

        generation = (self.index_generation or "").strip()
        object.__setattr__(self, "index_generation", generation)
        if generation and not re.fullmatch(r"[A-Za-z0-9_.:-]+", generation):
            raise ValueError(
                "index_generation must match [A-Za-z0-9_.:-]+ and must not contain whitespace or path separators"
            )
        if self.index_validation_min_points < 0:
            raise ValueError("index_validation_min_points must be >= 0")

        active_generation = (self.active_index_generation or "").strip()
        object.__setattr__(self, "active_index_generation", active_generation)
        if active_generation and not re.fullmatch(r"[A-Za-z0-9_.:-]+", active_generation):
            raise ValueError(
                "active_index_generation must match [A-Za-z0-9_.:-]+ and must not contain whitespace or path separators"
            )
        active_collection = (self.active_collection_name or "").strip()
        object.__setattr__(self, "active_collection_name", active_collection)
        if active_collection and not re.fullmatch(r"[A-Za-z0-9_.:-]+", active_collection):
            raise ValueError("active_collection_name must match [A-Za-z0-9_.:-]+ and must not contain whitespace")
        return self

    @model_validator(mode="after")
    def _resolve_cloudflare_base_url(self) -> AppSettings:
        if not self.cloudflare_base_url and self.cloudflare_account_id:
            object.__setattr__(
                self,
                "cloudflare_base_url",
                f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/v1",
            )
        return self

    @model_validator(mode="after")
    def _validate_provider_api_keys(self) -> AppSettings:
        if self.skip_provider_check:
            return self
        all_llm_providers = {
            p
            for p in [
                self.llm_provider,
                self.answer_llm_provider,
                self.rewrite_llm_provider,
                self.groundedness_llm_provider,
                self.intent_llm_provider,
                self.enrichment_llm_provider,
                self.evaluation_llm_provider,
                self.code_llm_provider,
            ]
            if p
        }
        # Each API-key-gated provider is checked: if referenced by any LLM purpose
        # or as the embedding provider, its API key must be set.
        provider_api_key_map: dict[str, tuple[str, str]] = {
            "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
            "nvidia": ("nvidia_api_key", "NVIDIA_API_KEY"),
            "groq": ("groq_api_key", "GROQ_API_KEY"),
            "cerebras": ("cerebras_api_key", "CEREBRAS_API_KEY"),
            "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
            "cloudflare": ("cloudflare_api_key", "CLOUDFLARE_API_KEY"),
            "opencodezen": ("opencodezen_api_key", "OPENCODEZEN_API_KEY"),
            "opencodego": ("opencodego_api_key", "OPENCODEGO_API_KEY"),
            "sambanova": ("sambanova_api_key", "SAMBANOVA_API_KEY"),
            "mistral": ("mistral_api_key", "MISTRAL_API_KEY"),
            "deepseek": ("deepseek_api_key", "DEEPSEEK_API_KEY"),
            "zai": ("zai_api_key", "ZAI_API_KEY"),
            "siliconflow": ("siliconflow_api_key", "SILICONFLOW_API_KEY"),
            "together": ("together_api_key", "TOGETHER_API_KEY"),
            "fireworks": ("fireworks_api_key", "FIREWORKS_API_KEY"),
            "llm7": ("llm7_api_key", "LLM7_API_KEY"),
            "agnes": ("agnes_api_key", "AGNES_API_KEY"),
            "ollama_cloud": ("ollama_cloud_api_key", "OLLAMA_API_KEY"),
            "helyx": ("helyx_api_key", "HELYX_API_KEY"),
            "anyapi": ("anyapi_api_key", "ANYAPI_API_KEY"),
            "huggingface": ("huggingface_api_key", "HUGGINGFACE_API_KEY or HF_TOKEN"),
        }
        for provider, (field_name, env_var) in provider_api_key_map.items():
            key_value = getattr(self, field_name).get_secret_value()
            if provider in all_llm_providers and not key_value:
                raise ValueError(f"{env_var} is required when any LLM provider is '{provider}'")
            if self.embedding_provider == provider and not key_value:
                raise ValueError(f"{env_var} is required when EMBEDDING_PROVIDER='{provider}'")
        return self

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension for the active model.

        Dimension is model-dependent — looks up the configured model in
        ``embedding_model_dimensions`` and falls back to
        ``default_embedding_dimension`` if the model is unrecognised.
        """
        provider = self.embedding_provider.lower()
        if provider == "openrouter":
            model_name = self.openrouter_embedding_model
        elif provider == "nvidia":
            model_name = self.nvidia_embedding_model
        elif provider == "gemini":
            model_name = self.gemini_embedding_model
        elif provider == "local-hf":
            model_name = self.local_hf_embedding_model
        elif provider == "huggingface":
            model_name = self.huggingface_embedding_model
        else:
            model_name = self.embedding_model_name
        return self.embedding_model_dimensions.get(model_name, self.default_embedding_dimension)

    def validate_all(self) -> None:
        """Cross-field consistency checks. Raises ``ValidationError`` on conflicts."""
        from pydantic import ValidationError

        errors: list[str] = []

        if self.reranker_enabled and self.reranker_top_k > self.retrieval_top_k:
            errors.append(
                f"reranker_top_k ({self.reranker_top_k}) must be <= retrieval_top_k ({self.retrieval_top_k}) "
                "— the reranker can only narrow the retrieved set."
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(f"confidence_threshold ({self.confidence_threshold}) must be within [0.0, 1.0]")
        if not 0.0 <= self.reranker_confidence_threshold <= 1.0:
            errors.append(
                f"reranker_confidence_threshold ({self.reranker_confidence_threshold}) must be within [0.0, 1.0]"
            )
        _known_rerank_providers = {"openrouter", "nvidia", "huggingface", "local"}
        unknown_rerank = [p for p in self.rerank_fallback_order if p.lower() not in _known_rerank_providers]
        if unknown_rerank:
            errors.append(
                f"rerank_fallback_order contains unknown provider(s): {unknown_rerank}. "
                f"Known providers: {sorted(_known_rerank_providers)}"
            )
        if self.rerank_cloud_timeout_seconds < 1:
            errors.append(f"rerank_cloud_timeout_seconds ({self.rerank_cloud_timeout_seconds}) must be >= 1")
        if self.max_pages_per_source < 0:
            errors.append(f"max_pages_per_source ({self.max_pages_per_source}) must be >= 0")
        if self.max_pages_hard_cap < 1:
            errors.append(f"max_pages_hard_cap ({self.max_pages_hard_cap}) must be >= 1")
        if self.crawl_attempt_multiplier < 1:
            errors.append(f"crawl_attempt_multiplier ({self.crawl_attempt_multiplier}) must be >= 1")
        if self.recovery_max_pages < 1:
            errors.append(f"recovery_max_pages ({self.recovery_max_pages}) must be >= 1")
        if self.chat_session_ttl_seconds < 60:
            errors.append(f"chat_session_ttl_seconds ({self.chat_session_ttl_seconds}) must be >= 60")
        if not 1 <= self.chat_history_max_turns <= 100:
            errors.append(f"chat_history_max_turns ({self.chat_history_max_turns}) must be within [1, 100]")
        if self.chat_history_max_tokens < 128:
            errors.append(f"chat_history_max_tokens ({self.chat_history_max_tokens}) must be >= 128")
        if self.chat_title_max_chars < 10:
            errors.append(f"chat_title_max_chars ({self.chat_title_max_chars}) must be >= 10")
        if not 0.0 < self.chat_cache_recall_threshold <= 1.0:
            errors.append(f"chat_cache_recall_threshold ({self.chat_cache_recall_threshold}) must be within (0.0, 1.0]")
        if not 1 <= self.chat_cache_top_k <= 20:
            errors.append(f"chat_cache_top_k ({self.chat_cache_top_k}) must be within [1, 20]")
        if self.chat_cache_max_age_seconds < 60:
            errors.append(f"chat_cache_max_age_seconds ({self.chat_cache_max_age_seconds}) must be >= 60")
        if any(not isinstance(s, str) or not s.strip() for s in self.chat_blocked_url_substrings):
            errors.append("chat_blocked_url_substrings must be non-empty strings")
        if any(not isinstance(s, str) or not s.strip() for s in self.chat_domain_sources):
            errors.append("chat_domain_sources must be non-empty strings")
        if not 1 <= self.chat_suggestions_count <= 5:
            errors.append(f"chat_suggestions_count ({self.chat_suggestions_count}) must be within [1, 5]")
        if self.chat_suggestions_mode not in ("llm", "rule", "hybrid"):
            errors.append(
                f"chat_suggestions_mode ({self.chat_suggestions_mode}) must be one of 'llm', 'rule', 'hybrid'"
            )
        configured_providers = {
            p
            for p in [
                self.llm_provider,
                self.answer_llm_provider,
                self.rewrite_llm_provider,
                self.groundedness_llm_provider,
                self.intent_llm_provider,
                self.enrichment_llm_provider,
                self.evaluation_llm_provider,
                self.code_llm_provider,
                self.embedding_provider,
            ]
            if p
        }
        if not configured_providers:
            errors.append("At least one LLM or embedding provider must be configured")

        if errors:
            from typing import cast

            from pydantic_core import InitErrorDetails

            line_errors = [
                {
                    "type": "value_error",
                    "loc": ("settings",),
                    "msg": msg,
                    "input": None,
                    "ctx": {"error": ValueError(msg)},
                }
                for msg in errors
            ]
            raise ValidationError.from_exception_data(
                title="AppSettings.validate_all",
                line_errors=cast(list[InitErrorDetails], line_errors),
            )


# Module-load seam ONLY: skip the API-key validator so importing this module
# never hard-fails on missing provider API keys (e.g. in CI or a fresh clone
# where .env.secrets is absent).  Provider builders in factory.py still
# validate API keys and raise clear errors when a non-Ollama provider is
# built.  NOT a test escape hatch — tests use tests.conftest.make_settings()
# (which is hermetic and never reads .env) and must not set this flag.
settings = AppSettings(skip_provider_check=True)
