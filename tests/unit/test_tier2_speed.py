from pathlib import Path


def test_tune_tier2_speed_exists() -> None:
    assert Path("scripts/tune_tier2_speed.py").exists()


def test_settings_has_tier2_candidates() -> None:
    from data_engineering_copilot.config.settings import settings

    assert hasattr(settings, "retrieval_top_k")
    assert hasattr(settings, "mrl_multistage_enabled")
