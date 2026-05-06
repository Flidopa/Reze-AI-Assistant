"""Core infrastructure: event bus, event types, config, and logging."""

from assistant.core.config import AssistantConfig, get_config
from assistant.core.event_bus import EventBus
from assistant.core.events import (
    AssistantShutdown,
    AssistantStarted,
    AudioPlaybackRequested,
    Event,
    LLMResponseReady,
    SpeechSynthesisRequested,
    TelegramMessageReceived,
    TelegramReplySent,
    ToolCallCompleted,
    ToolCallRequested,
    UserSpeechDetected,
    WakeWordDetected,
)
from assistant.core.logging import setup_logging

__all__ = [
    "AssistantConfig",
    "get_config",
    "setup_logging",
    "EventBus",
    "Event",
    "WakeWordDetected",
    "UserSpeechDetected",
    "SpeechSynthesisRequested",
    "AudioPlaybackRequested",
    "LLMResponseReady",
    "ToolCallRequested",
    "ToolCallCompleted",
    "TelegramMessageReceived",
    "TelegramReplySent",
    "AssistantStarted",
    "AssistantShutdown",
]
