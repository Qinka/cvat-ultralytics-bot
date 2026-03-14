"""YOLO detection annotation tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import Detection

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class YoloDetectTool:
    """Ultralytics YOLO object detector."""

    weights: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.weights)
        self.tool_name = "yolo_detect"

    def predict(self, image: "Image", conf: float = 0.25) -> list[Detection]:
        results = self._model.predict(image, conf=conf, device=self.device, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                detections.append(
                    Detection(
                        class_name=self._model.names[cls_id],
                        confidence=float(boxes.conf[index].item()),
                        bbox_xyxy=boxes.xyxy[index].tolist(),
                    )
                )
        return detections


def build_tool(config: dict[str, Any]):
    return YoloDetectTool(weights=str(config["weights"]), device=str(config.get("device", "cpu")))


register_tool(
    AnnotationToolRegistration(
        name="yolo_detect",
        factory=build_tool,
        description="Ultralytics YOLO detection tool",
        use_polygon=False,
    )
)
