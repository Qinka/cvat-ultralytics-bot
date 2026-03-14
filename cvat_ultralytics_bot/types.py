"""Shared types for annotation tools and predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class PredictedObject:
    """A single prediction result from a model (detection or segmentation)."""

    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    polygon_xy: list[float] | None = field(default=None)


class AnnotationTool(Protocol):
    """Protocol implemented by all annotation tools."""

    @property
    def tool_name(self) -> str:
        """Unique tool identifier."""

    def predict(self, image: "Image", conf: float = 0.25) -> list[PredictedObject]:
        """Run inference and return predictions."""
