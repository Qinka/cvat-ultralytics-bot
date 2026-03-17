"""Base classes and registry helpers for annotation tools.

This module provides the core abstractions for annotation tools:
- :class:`AnnotationTool`: Protocol defining the interface for all tools.
- :class:`AnnotationToolFactory`: Protocol for constructing tools.
- :class:`AnnotationToolRegistration`: Dataclass holding tool metadata.

Example:
    >>> from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
    >>> # Register a custom tool
    >>> register_tool(AnnotationToolRegistration(
    ...     name="my_tool",
    ...     factory=my_factory,
    ...     description="My custom annotation tool",
    ... ))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cvat_ultralytics_bot.types import AnnotationTool


class AnnotationToolFactory(Protocol):
    """Callable protocol for constructing annotation tools from config.

    A factory is any callable that takes a configuration dictionary
    and returns an object implementing the :class:`AnnotationTool` protocol.
    """

    def __call__(self, config: dict[str, Any]) -> AnnotationTool:
        """Build an annotation tool instance.

        Args:
            config: Tool-specific configuration dictionary.

        Returns:
            An object implementing the :class:`AnnotationTool` protocol.
        """


@dataclass(frozen=True)
class AnnotationToolRegistration:
    """Registered annotation tool metadata.

    Attributes:
        name: Unique identifier for the tool.
        factory: Callable that constructs the tool from config.
        description: Human-readable description of the tool.
        use_polygon: Whether the tool produces polygon outputs.
    """

    name: str
    factory: AnnotationToolFactory
    description: str
    use_polygon: bool = False
