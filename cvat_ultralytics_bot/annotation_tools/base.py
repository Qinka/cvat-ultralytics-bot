"""Base classes and registry helpers for annotation tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cvat_ultralytics_bot.types import AnnotationTool


class AnnotationToolFactory(Protocol):
    """Callable protocol for constructing annotation tools from config."""

    def __call__(self, config: dict[str, Any]) -> AnnotationTool:
        """Build an annotation tool instance."""


@dataclass(frozen=True)
class AnnotationToolRegistration:
    """Registered annotation tool metadata."""

    name: str
    factory: AnnotationToolFactory
    description: str
    use_polygon: bool = False
