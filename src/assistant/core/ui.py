"""Rich-powered terminal UI: shared console, banner, dialog rendering, spinner."""

from __future__ import annotations

import logging

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.text import Text

from assistant.core.event_bus import EventBus
from assistant.core.events import (
    AudioPlaybackRequested,
    LLMResponseReady,
    SpeechSynthesisRequested,
    ToolCallCompleted,
    ToolCallRequested,
    UserSpeechDetected,
    WakeWordDetected,
)

__all__ = ["ConsoleUI", "get_console", "print_banner"]

logger = logging.getLogger(__name__)

# --- color palette --------------------------------------------------------

_USER_PREFIX_STYLE = "bold cyan"
_USER_TEXT_STYLE = "cyan"
_ASSIST_PREFIX_STYLE = "bold magenta"
_ASSIST_TEXT_STYLE = "magenta"
_SYS_STYLE = "dim yellow"
_RULE_STYLE = "grey39"
_THINK_STYLE = "magenta"
_TOOL_STYLE = "yellow"
_TTS_STYLE = "bright_blue"

# --- per-tool spinner labels (Russian) ------------------------------------

_TOOL_LABELS_RU: dict[str, str] = {
    "web_search": "Ищу в интернете...",
    "weather": "Смотрю погоду...",
    "currency_rate": "Уточняю курс...",
    "datetime": "Сверяюсь со временем...",
    "current_datetime": "Сверяюсь со временем...",
}

_console: Console | None = None


def get_console() -> Console:
    """Return the singleton rich Console shared by logging and UI."""
    global _console
    if _console is None:
        _console = Console(highlight=False, soft_wrap=False)
    return _console


def print_banner(name: str) -> None:
    """Print a stylised welcome banner with the assistant's name."""
    console = get_console()
    title = Text(f"⚡  {name.upper()}  ⚡", style="bold magenta", justify="center")
    subtitle = Text("voice assistant · ready when you are", style="italic dim", justify="center")
    body = Align.center(Group(title, Text(""), subtitle))
    console.print()
    console.print(Panel(body, border_style="magenta", padding=(1, 6)))
    console.print()


class ConsoleUI:
    """Renders the dialog and spinners by subscribing to bus events."""

    def __init__(self, console: Console, assistant_name: str) -> None:
        self._console = console
        self._name = assistant_name
        self._status: Status | None = None

    def start(self, event_bus: EventBus) -> None:
        """Subscribe to bus events. Call once at startup."""
        event_bus.subscribe(WakeWordDetected, self._on_wake)
        event_bus.subscribe(UserSpeechDetected, self._on_user_speech)
        event_bus.subscribe(ToolCallRequested, self._on_tool_requested)
        event_bus.subscribe(ToolCallCompleted, self._on_tool_completed)
        event_bus.subscribe(LLMResponseReady, self._on_llm_response)
        event_bus.subscribe(SpeechSynthesisRequested, self._on_synthesis_requested)
        event_bus.subscribe(AudioPlaybackRequested, self._on_playback_requested)

    # --- spinner control -------------------------------------------------

    def _start_spinner(self, text: str, style: str = _THINK_STYLE) -> None:
        self._stop_spinner()
        self._status = self._console.status(f"[{style}]{text}[/]", spinner="dots")
        self._status.start()

    def _update_spinner(self, text: str, style: str = _THINK_STYLE) -> None:
        if self._status is None:
            self._start_spinner(text, style)
            return
        self._status.update(f"[{style}]{text}[/]")

    def _stop_spinner(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    # --- event handlers --------------------------------------------------

    async def _on_wake(self, _event: WakeWordDetected) -> None:
        self._console.print(Text("🎙   слышу тебя...", style=_SYS_STYLE))

    async def _on_user_speech(self, event: UserSpeechDetected) -> None:
        self._console.print(Rule(style=_RULE_STYLE))
        line = Text("🎤  Ты: ", style=_USER_PREFIX_STYLE)
        line.append(event.text, style=_USER_TEXT_STYLE)
        self._console.print(line)
        self._start_spinner("Думаю...")

    async def _on_tool_requested(self, event: ToolCallRequested) -> None:
        label = _TOOL_LABELS_RU.get(event.tool_name, f"Использую {event.tool_name}...")
        self._update_spinner(label, style=_TOOL_STYLE)

    async def _on_tool_completed(self, event: ToolCallCompleted) -> None:
        marker = "✓" if event.success else "✗"
        color = "green" if event.success else "red"
        self._console.print(Text(f"   {marker} {event.tool_name}", style=f"dim {color}"))
        self._update_spinner("Думаю...")

    async def _on_llm_response(self, event: LLMResponseReady) -> None:
        self._stop_spinner()
        line = Text(f"🤖  {self._name}: ", style=_ASSIST_PREFIX_STYLE)
        line.append(event.text, style=_ASSIST_TEXT_STYLE)
        self._console.print(line)

    async def _on_synthesis_requested(self, _event: SpeechSynthesisRequested) -> None:
        self._start_spinner("Синтезирую речь...", style=_TTS_STYLE)

    async def _on_playback_requested(self, _event: AudioPlaybackRequested) -> None:
        self._stop_spinner()
