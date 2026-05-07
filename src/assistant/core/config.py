"""Centralised application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AssistantConfig", "get_config"]


class AssistantConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    openrouter_api_key: str = ""
    llm_model: str = "x-ai/grok-4-fast"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout: int = 30
    llm_max_tokens: int = 1024
    llm_max_history: int = 20

    # --- Wake Word ---
    porcupine_access_key: str = ""
    porcupine_keyword: str = "jarvis"
    porcupine_keyword_path: str = ""
    wake_word_sensitivity: float = 0.5

    # --- Audio / STT ---
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    stt_model: str = "medium"
    stt_language: str = "ru"

    # --- TTS / Voice ---
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_language: str = "ru"
    tts_emotion: str = "neutral"
    tts_voice: str = "ru-RU-SvetlanaNeural"
    voice_sample_path: str = ""

    # --- Telegram ---
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_phone: str = ""

    # --- Weather ---
    openweather_api_key: str = ""

    # --- System ---
    log_level: str = "INFO"
    data_dir: str = "data"
    assistant_name: str = "Reze"

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, v: str) -> str:
        return v.upper()

    def data_path(self, *parts: str) -> Path:
        """Return a path under the configured data directory."""
        return Path(self.data_dir).joinpath(*parts)


@lru_cache(maxsize=1)
def get_config() -> AssistantConfig:
    """Return the application config (cached after first call)."""
    return AssistantConfig()
