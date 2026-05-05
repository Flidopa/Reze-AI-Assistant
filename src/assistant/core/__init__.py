"""Core infrastructure: event bus and event types."""

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

__all__ = [
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
