"""Tests for core/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.core.config import AssistantConfig, get_config


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:  # type: ignore[return]
    yield
    get_config.cache_clear()


class TestDefaults:
    def test_llm_defaults(self) -> None:
        cfg = AssistantConfig()
        assert cfg.llm_model == "openai/gpt-4o-mini"
        assert cfg.llm_timeout == 30
        assert cfg.llm_max_tokens == 1024
        assert cfg.llm_base_url == "https://openrouter.ai/api/v1"

    def test_audio_defaults(self) -> None:
        cfg = AssistantConfig()
        assert cfg.audio_sample_rate == 16000
        assert cfg.audio_channels == 1
        assert cfg.stt_model == "medium"
        assert cfg.stt_language == "ru"

    def test_system_defaults(self) -> None:
        cfg = AssistantConfig()
        assert cfg.log_level == "INFO"
        assert cfg.data_dir == "data"

    def test_wake_word_defaults(self) -> None:
        cfg = AssistantConfig()
        assert cfg.porcupine_access_key == ""
        assert cfg.wake_word_sensitivity == 0.5


class TestEnvOverride:
    def test_llm_model_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL", "anthropic/claude-3-haiku")
        cfg = AssistantConfig()
        assert cfg.llm_model == "anthropic/claude-3-haiku"

    def test_log_level_overridden_and_uppercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        cfg = AssistantConfig()
        assert cfg.log_level == "DEBUG"

    def test_audio_sample_rate_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_SAMPLE_RATE", "22050")
        cfg = AssistantConfig()
        assert cfg.audio_sample_rate == 22050

    def test_get_config_uses_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STT_MODEL", "large-v3")
        cfg = get_config()
        assert cfg.stt_model == "large-v3"


class TestDataPath:
    def test_single_part(self) -> None:
        cfg = AssistantConfig()
        assert cfg.data_path("models") == Path("data") / "models"

    def test_multiple_parts(self) -> None:
        cfg = AssistantConfig()
        assert cfg.data_path("models", "whisper") == Path("data") / "models" / "whisper"

    def test_custom_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATA_DIR", "/tmp/assistant")
        cfg = AssistantConfig()
        assert cfg.data_path("chroma") == Path("/tmp/assistant") / "chroma"


class TestTypes:
    def test_log_level_is_str(self) -> None:
        cfg = AssistantConfig()
        assert isinstance(cfg.log_level, str)

    def test_audio_sample_rate_is_int(self) -> None:
        cfg = AssistantConfig()
        assert isinstance(cfg.audio_sample_rate, int)

    def test_wake_word_sensitivity_is_float(self) -> None:
        cfg = AssistantConfig()
        assert isinstance(cfg.wake_word_sensitivity, float)

    def test_telegram_api_id_is_int(self) -> None:
        cfg = AssistantConfig()
        assert isinstance(cfg.telegram_api_id, int)


class TestLruCache:
    def test_same_object_returned(self) -> None:
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_new_object_after_cache_clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg1 = get_config()
        get_config.cache_clear()
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        cfg2 = get_config()
        assert cfg1 is not cfg2
        assert cfg2.log_level == "WARNING"
