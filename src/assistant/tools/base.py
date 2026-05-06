"""Base Tool ABC and ToolResult dataclass for function-calling tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = ["Tool", "ToolResult"]


@dataclass
class ToolResult:
    content: str
    success: bool = True


class Tool(ABC):
    """Abstract base class for all assistant tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier used in function calling."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema object describing the tool's parameters."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a result."""

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
