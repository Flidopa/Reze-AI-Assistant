"""All event dataclasses used across the system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "Event",
    # Audio
    "WakeWordDetected",
    "RecordingStarted",
    "RecordingStopped",
    "UserSpeechDetected",
    "SpeechSynthesisRequested",
    "AudioPlaybackRequested",
    # Brain
    "LLMResponseReady",
    "ToolCallRequested",
    "ToolCallCompleted",
    # Telegram
    "TelegramMessageReceived",
    "TelegramReplySent",
    # System
    "AssistantStarted",
    "AssistantShutdown",
]


@dataclass
class Event:
    pass


# ---------------------------------------------------------------------------
# Audio events
# ---------------------------------------------------------------------------


@dataclass
class WakeWordDetected(Event):
    pass


@dataclass
class RecordingStarted(Event):
    pass


@dataclass
class RecordingStopped(Event):
    pass


@dataclass
class UserSpeechDetected(Event):
    text: str = ""


@dataclass
class SpeechSynthesisRequested(Event):
    text: str = ""
    emotion: str = "neutral"


@dataclass
class AudioPlaybackRequested(Event):
    audio_data: bytes = b""


# ---------------------------------------------------------------------------
# Brain events
# ---------------------------------------------------------------------------


@dataclass
class LLMResponseReady(Event):
    text: str = ""


@dataclass
class ToolCallRequested(Event):
    tool_name: str = ""
    arguments: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.arguments is None:
            self.arguments = {}


@dataclass
class ToolCallCompleted(Event):
    tool_name: str = ""
    result: str = ""
    success: bool = True


# ---------------------------------------------------------------------------
# Telegram events
# ---------------------------------------------------------------------------


@dataclass
class TelegramMessageReceived(Event):
    sender: str = ""
    text: str = ""
    chat_id: int = 0


@dataclass
class TelegramReplySent(Event):
    chat_id: int = 0
    text: str = ""


# ---------------------------------------------------------------------------
# System events
# ---------------------------------------------------------------------------


@dataclass
class AssistantStarted(Event):
    pass


@dataclass
class AssistantShutdown(Event):
    reason: str = "normal"
