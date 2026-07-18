"""
Tests for chunking factory and configuration.

Tests cover:
- Factory function behavior
- Strategy selection and configuration
- Settings validation
- Fallback mechanisms
"""

import pytest
from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.factory import build_chunker
from data_engineering_copilot.services.chunker import DocumentChunker, ChunkingStrategy
from data_engineering_copilot.services.semantic_chunker import SemanticChunker


class TestSettingsValidation:
    """Tests for AppSettings chunking configuration."""

    def test_default_settings_use_sentence_preserving(self):
        """Test that default settings use sentence_preserving strategy."""
        assert AppSettings.chunking_strategy == "sentence_preserving"

    def test_default_min_semantic_similarity(self):
        """Test that default similarity threshold is set."""
        assert AppSettings.min_semantic_similarity == 0.5
        assert 0.0 <= AppSettings.min_semantic_similarity <= 1.0

    def test_semantic_chunking_flag_has_default(self):
        """Test that enable_semantic_chunking has a boolean default."""
        default = AppSettings.enable_semantic_chunking
        assert isinstance(default, bool)

    def test_chunk_size_words_default(self):
        """Test default chunk size is positive and matches settings."""
        default = AppSettings.chunk_size_words
        assert isinstance(default, int)
        assert default > 0

    def test_chunk_overlap_words_default(self):
        """Test default overlap is non-negative and less than chunk size."""
        default = AppSettings.chunk_overlap_words
        assert isinstance(default, int)
        assert 0 <= default < AppSettings.chunk_size_words

    def test_max_chunk_words_defaults_to_none(self):
        """Test that max_chunk_words defaults to None (auto-calculated)."""
        assert AppSettings.max_chunk_words is None


class TestFactoryFunctionBehavior:
    """Tests for build_chunker factory function."""

    def test_build_chunker_with_sentence_preserving_strategy(self):
        """Test building chunker with sentence_preserving strategy."""
        settings = AppSettings(chunking_strategy="sentence_preserving")
        chunker = build_chunker(settings)

        assert isinstance(chunker, DocumentChunker)
        assert chunker.strategy == ChunkingStrategy.SENTENCE_PRESERVING

    def test_build_chunker_with_fixed_size_strategy(self):
        """Test building chunker with fixed_size strategy."""
        settings = AppSettings(chunking_strategy="fixed_size")
        chunker = build_chunker(settings)

        assert isinstance(chunker, DocumentChunker)
        assert chunker.strategy == ChunkingStrategy.FIXED_SIZE

    def test_build_chunker_with_semantic_disabled_falls_back(self):
        """Test that semantic chunking falls back to sentence_preserving when disabled."""
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=False,  # Disabled
        )
        chunker = build_chunker(settings)

        # Should fall back to sentence_preserving
        assert isinstance(chunker, DocumentChunker)
        assert chunker.strategy == ChunkingStrategy.SENTENCE_PRESERVING

    def test_build_chunker_with_semantic_enabled(self):
        """Test that semantic chunker is built when enabled."""
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,  # Enabled
        )
        chunker = build_chunker(settings)

        # Should build semantic chunker
        assert isinstance(chunker, SemanticChunker)

    def test_build_chunker_with_invalid_strategy_defaults(self):
        """Test that invalid strategy defaults to sentence_preserving."""
        settings = AppSettings(chunking_strategy="invalid_strategy_xyz")
        chunker = build_chunker(settings)

        # Should default to sentence_preserving
        assert isinstance(chunker, DocumentChunker)
        assert chunker.strategy == ChunkingStrategy.SENTENCE_PRESERVING

    def test_build_chunker_respects_chunk_size(self):
        """Test that chunker is configured with correct chunk size."""
        settings = AppSettings(
            chunk_size_words=300,
            chunk_overlap_words=60,
        )
        chunker = build_chunker(settings)

        assert chunker.chunk_size_words == 300
        assert chunker.overlap_words == 60

    def test_build_chunker_respects_similarity_threshold(self):
        """Test that semantic chunker uses configured similarity threshold."""
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            min_semantic_similarity=0.7,
        )
        chunker = build_chunker(settings)

        assert isinstance(chunker, SemanticChunker)
        assert chunker.min_semantic_similarity == 0.7

    def test_build_chunker_calculates_max_chunk_words_auto(self):
        """Test that max_chunk_words is auto-calculated when None for semantic chunker."""
        settings = AppSettings(
            chunk_size_words=250,
            max_chunk_words=None,  # Auto-calculate
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
        )
        chunker = build_chunker(settings)

        # Max should be 1.5x target (for semantic chunker only)
        expected_max = int(250 * 1.5)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.max_chunk_words == expected_max

    def test_build_chunker_respects_custom_max_chunk_words(self):
        """Test that custom max_chunk_words is respected for semantic chunker."""
        settings = AppSettings(
            chunk_size_words=250,
            max_chunk_words=400,  # Custom value
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
        )
        chunker = build_chunker(settings)

        assert isinstance(chunker, SemanticChunker)
        assert chunker.max_chunk_words == 400

    def test_build_chunker_min_chunk_words_is_percentage_of_target(self):
        """Test that min_chunk_words is calculated as 10% of target."""
        settings = AppSettings(chunk_size_words=250)
        chunker = build_chunker(settings)

        expected_min = int(250 * 0.1)  # 10% of target
        assert chunker.min_chunk_words == expected_min

    def test_build_chunker_with_default_settings_produces_valid_chunker(self):
        """Test that factory creates valid chunker with default settings."""
        settings = AppSettings()
        chunker = build_chunker(settings)

        # Should be valid and usable
        assert chunker is not None
        assert hasattr(chunker, "chunk")
        assert callable(chunker.chunk)


class TestStrategySelection:
    """Tests for strategy selection logic."""

    def test_case_insensitive_strategy_selection(self):
        """Test that strategy selection is case-insensitive."""
        for strategy in ["SENTENCE_PRESERVING", "Sentence_Preserving", "sentence_preserving"]:
            settings = AppSettings(chunking_strategy=strategy)
            chunker = build_chunker(settings)
            assert isinstance(chunker, DocumentChunker)

    def test_semantic_strategy_case_insensitive(self):
        """Test that semantic strategy is case-insensitive."""
        for strategy in ["SEMANTIC", "Semantic", "semantic"]:
            settings = AppSettings(
                chunking_strategy=strategy,
                enable_semantic_chunking=True,
            )
            chunker = build_chunker(settings)
            assert isinstance(chunker, SemanticChunker)

    def test_strategy_priority_semantic_over_others(self):
        """Test that semantic is correctly distinguished from others."""
        # semantic should create SemanticChunker
        settings_semantic = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
        )
        chunker_semantic = build_chunker(settings_semantic)
        assert isinstance(chunker_semantic, SemanticChunker)

        # fixed_size should create DocumentChunker
        settings_fixed = AppSettings(chunking_strategy="fixed_size")
        chunker_fixed = build_chunker(settings_fixed)
        assert isinstance(chunker_fixed, DocumentChunker)
        assert not isinstance(chunker_fixed, SemanticChunker)


class TestSemanticChunkerConfiguration:
    """Tests for semantic chunker specific configuration."""

    def test_semantic_chunker_receives_embedding_model(self):
        """Test that semantic chunker is initialized with embedding model."""
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            embedding_model_name="nomic-embed-text",
        )
        chunker = build_chunker(settings)

        assert isinstance(chunker, SemanticChunker)
        assert chunker.embedding_model is not None

    def test_semantic_chunker_with_custom_parameters(self):
        """Test semantic chunker with custom configuration."""
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            chunk_size_words=200,
            chunk_overlap_words=40,
            min_semantic_similarity=0.6,
            max_chunk_words=300,
        )
        chunker = build_chunker(settings)

        assert isinstance(chunker, SemanticChunker)
        assert chunker.chunk_size_words == 200
        assert chunker.overlap_words == 40
        assert chunker.min_semantic_similarity == 0.6
        assert chunker.max_chunk_words == 300


class TestDocumentChunkerConfiguration:
    """Tests for DocumentChunker specific configuration."""

    def test_document_chunker_receives_strategy(self):
        """Test that DocumentChunker is initialized with correct strategy."""
        settings = AppSettings(chunking_strategy="fixed_size")
        chunker = build_chunker(settings)

        assert isinstance(chunker, DocumentChunker)
        assert chunker.strategy == ChunkingStrategy.FIXED_SIZE

    def test_document_chunker_with_sentence_preserving(self):
        """Test DocumentChunker with sentence_preserving strategy."""
        settings = AppSettings(chunking_strategy="sentence_preserving")
        chunker = build_chunker(settings)

        assert isinstance(chunker, DocumentChunker)
        assert chunker.strategy == ChunkingStrategy.SENTENCE_PRESERVING

    def test_document_chunker_min_chunk_words_set_correctly(self):
        """Test that DocumentChunker min_chunk_words is set."""
        settings = AppSettings(chunk_size_words=300)
        chunker = build_chunker(settings)

        # Should be 10% of target
        assert chunker.min_chunk_words == int(300 * 0.1)


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_default_settings_produce_sentence_preserving_chunker(self):
        """Test that default behavior is sentence_preserving (backward compatible)."""
        settings = AppSettings()
        chunker = build_chunker(settings)

        # Default should be sentence_preserving
        assert isinstance(chunker, DocumentChunker)
        assert chunker.strategy == ChunkingStrategy.SENTENCE_PRESERVING

    def test_existing_code_unaffected_by_semantic_flag(self):
        """Test that existing code isn't affected by semantic chunking flag."""
        # Code that doesn't specify semantic should work as before
        settings = AppSettings()
        chunker = build_chunker(settings)

        # Should work without errors
        assert hasattr(chunker, "chunk")
        assert callable(chunker.chunk)

    def test_language_compatibility_with_factory(self):
        """Test that Python typing is backward compatible."""
        from data_engineering_copilot.config.settings import AppSettings
        from data_engineering_copilot.factory import build_chunker

        # This should work without type errors
        settings = AppSettings()
        chunker = build_chunker(settings)
        assert chunker is not None


class TestIntegration:
    """Integration tests for factory and settings."""

    def test_all_default_settings_build_successfully(self):
        """Test that default AppSettings builds a valid chunker."""
        from data_engineering_copilot.config.settings import settings as default_settings

        chunker = build_chunker(default_settings)
        assert chunker is not None
        assert hasattr(chunker, "chunk")

    def test_multiple_strategy_configurations_work(self):
        """Test that different strategy configurations all work."""
        strategies = [
            "fixed_size",
            "sentence_preserving",
        ]

        for strategy in strategies:
            settings = AppSettings(chunking_strategy=strategy)
            chunker = build_chunker(settings)
            assert chunker is not None
            assert hasattr(chunker, "chunk")

    def test_factory_with_production_settings(self):
        """Test factory with realistic production settings."""
        settings = AppSettings(
            chunking_strategy="sentence_preserving",
            chunk_size_words=250,
            chunk_overlap_words=50,
            enable_semantic_chunking=False,
        )
        chunker = build_chunker(settings)

        assert isinstance(chunker, DocumentChunker)
        assert chunker.chunk_size_words == 250
        assert chunker.overlap_words == 50
