"""YOLO instance segmentation annotation tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import Detection

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class YoloSegmentTool:
    """Ultralytics YOLO instance segmentor."""

    weights: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.weights)
        self.tool_name = "yolo_segment"

    def predict(self, image: "Image", conf: float = 0.25) -> list[Detection]:
        results = self._model.predict(image, conf=conf, device=self.device, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            masks = result.masks
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                polygon: list[float] | None = None
                if masks is not None and index < len(masks.xy):
                    polygon = masks.xy[index].flatten().tolist()
                detections.append(
                    Detection(
                        class_name=self._model.names[cls_id],
                        confidence=float(boxes.conf[index].item()),
                        bbox_xyxy=boxes.xyxy[index].tolist(),
                        polygon_xy=polygon,
                    )
                )
        return detections


def build_tool(config: dict[str, Any]):
    return YoloSegmentTool(weights=str(config["weights"]), device=str(config.get("device", "cpu")))


register_tool(
    AnnotationToolRegistration(
        name="yolo_segment",
        factory=build_tool,
        description="Ultralytics YOLO segmentation tool",
        use_polygon=True,
    )
)
