"""Tests for chunking factory and configuration."""

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.factory import build_chunker
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.semantic_chunker import SemanticChunker
from data_engineering_copilot.services.sentence_preserving_chunker import SentencePreservingChunker


class TestSettingsValidation:
    def test_default_settings_use_sentence_preserving(self):
        assert AppSettings().chunking_strategy == "sentence_preserving"

    def test_default_min_semantic_similarity(self):
        s = AppSettings()
        assert s.min_semantic_similarity == 0.5
        assert 0.0 <= s.min_semantic_similarity <= 1.0

    def test_semantic_chunking_flag_has_default(self):
        default = AppSettings().enable_semantic_chunking
        assert isinstance(default, bool)

    def test_chunk_size_words_default(self):
        default = AppSettings().chunk_size_words
        assert isinstance(default, int)
        assert default > 0

    def test_chunk_overlap_words_default(self):
        s = AppSettings()
        assert isinstance(s.chunk_overlap_words, int)
        assert 0 <= s.chunk_overlap_words < s.chunk_size_words

    def test_max_chunk_words_defaults_to_none(self):
        assert AppSettings().max_chunk_words is None


class TestFactoryFunctionBehavior:
    def test_build_chunker_with_sentence_preserving_strategy(self):
        settings = AppSettings(chunking_strategy="sentence_preserving")
        chunker = build_chunker(settings)
        assert isinstance(chunker, SentencePreservingChunker)

    def test_build_chunker_with_fixed_size_strategy(self):
        settings = AppSettings(chunking_strategy="fixed_size")
        chunker = build_chunker(settings)
        assert isinstance(chunker, DocumentChunker)

    def test_build_chunker_with_semantic_disabled_falls_back(self):
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=False,
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, SentencePreservingChunker)

    def test_build_chunker_with_semantic_enabled(self):
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            embedding_provider="local-hf",
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, SemanticChunker)

    def test_build_chunker_with_invalid_strategy_defaults(self):
        settings = AppSettings(chunking_strategy="invalid_strategy_xyz")
        chunker = build_chunker(settings)
        assert isinstance(chunker, SentencePreservingChunker)

    def test_build_chunker_respects_chunk_size(self):
        settings = AppSettings(
            chunk_size_words=300,
            chunk_overlap_words=60,
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, SentencePreservingChunker)
        assert chunker.max_chars == 300 * 5

    def test_build_chunker_respects_similarity_threshold(self):
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            min_semantic_similarity=0.7,
            embedding_provider="local-hf",
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.min_semantic_similarity == 0.7

    def test_build_chunker_calculates_max_chunk_words_auto(self):
        settings = AppSettings(
            chunk_size_words=250,
            max_chunk_words=None,
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            embedding_provider="local-hf",
        )
        chunker = build_chunker(settings)
        expected_max = int(250 * 1.5)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.max_chunk_words == expected_max

    def test_build_chunker_respects_custom_max_chunk_words(self):
        settings = AppSettings(
            chunk_size_words=250,
            max_chunk_words=400,
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            embedding_provider="local-hf",
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.max_chunk_words == 400

    def test_build_chunker_with_default_settings_produces_valid_chunker(self):
        settings = AppSettings()
        chunker = build_chunker(settings)
        assert chunker is not None
        assert hasattr(chunker, "chunk")
        assert callable(chunker.chunk)


class TestStrategySelection:
    def test_case_insensitive_strategy_selection(self):
        for strategy in ["SENTENCE_PRESERVING", "Sentence_Preserving", "sentence_preserving"]:
            settings = AppSettings(chunking_strategy=strategy)
            chunker = build_chunker(settings)
            assert isinstance(chunker, SentencePreservingChunker)

    def test_semantic_strategy_case_insensitive(self):
        for strategy in ["SEMANTIC", "Semantic", "semantic"]:
            settings = AppSettings(
                chunking_strategy=strategy,
                enable_semantic_chunking=True,
                embedding_provider="local-hf",
            )
            chunker = build_chunker(settings)
            assert isinstance(chunker, SemanticChunker)

    def test_strategy_priority_semantic_over_others(self):
        settings_semantic = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            embedding_provider="local-hf",
        )
        chunker_semantic = build_chunker(settings_semantic)
        assert isinstance(chunker_semantic, SemanticChunker)

        settings_fixed = AppSettings(chunking_strategy="fixed_size")
        chunker_fixed = build_chunker(settings_fixed)
        assert isinstance(chunker_fixed, DocumentChunker)
        assert not isinstance(chunker_fixed, SemanticChunker)


class TestSemanticChunkerConfiguration:
    def test_semantic_chunker_receives_embedding_model(self):
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            local_hf_embedding_model="nvidia/Nemotron-3-Embed-1B-BF16",
            embedding_provider="local-hf",
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.embedding_model is not None

    def test_semantic_chunker_with_custom_parameters(self):
        settings = AppSettings(
            chunking_strategy="semantic",
            enable_semantic_chunking=True,
            chunk_size_words=200,
            chunk_overlap_words=40,
            min_semantic_similarity=0.6,
            max_chunk_words=300,
            embedding_provider="local-hf",
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.chunk_size_words == 200
        assert chunker.overlap_words == 40
        assert chunker.min_semantic_similarity == 0.6
        assert chunker.max_chunk_words == 300


class TestDocumentChunkerConfiguration:
    def test_document_chunker_receives_correct_chunk_size(self):
        settings = AppSettings(chunking_strategy="fixed_size", chunk_size_words=200, chunk_overlap_words=40)
        chunker = build_chunker(settings)
        assert isinstance(chunker, DocumentChunker)
        assert chunker.chunk_size_chars == 200 * 5
        assert chunker.chunk_overlap_chars == 40 * 5

    def test_document_chunker_with_sentence_preserving(self):
        settings = AppSettings(chunking_strategy="sentence_preserving", chunk_size_words=300, chunk_overlap_words=60)
        chunker = build_chunker(settings)
        assert isinstance(chunker, SentencePreservingChunker)
        assert chunker.max_chars == 300 * 5


class TestBackwardCompatibility:
    def test_default_settings_produce_sentence_preserving_chunker(self):
        settings = AppSettings()
        chunker = build_chunker(settings)
        assert isinstance(chunker, SentencePreservingChunker)

    def test_existing_code_unaffected_by_semantic_flag(self):
        settings = AppSettings()
        chunker = build_chunker(settings)
        assert hasattr(chunker, "chunk")
        assert callable(chunker.chunk)

    def test_language_compatibility_with_factory(self):
        from data_engineering_copilot.config.settings import AppSettings
        from data_engineering_copilot.factory import build_chunker

        settings = AppSettings()
        chunker = build_chunker(settings)
        assert chunker is not None


class TestIntegration:
    def test_all_default_settings_build_successfully(self):
        from data_engineering_copilot.config.settings import settings as default_settings

        chunker = build_chunker(default_settings)
        assert chunker is not None
        assert hasattr(chunker, "chunk")

    def test_multiple_strategy_configurations_work(self):
        strategies = ["fixed_size", "sentence_preserving"]
        for strategy in strategies:
            settings = AppSettings(chunking_strategy=strategy)
            chunker = build_chunker(settings)
            assert chunker is not None
            assert hasattr(chunker, "chunk")

    def test_factory_with_production_settings(self):
        settings = AppSettings(
            chunking_strategy="sentence_preserving",
            chunk_size_words=250,
            chunk_overlap_words=50,
            enable_semantic_chunking=False,
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, SentencePreservingChunker)
        assert chunker.max_chars == 250 * 5
