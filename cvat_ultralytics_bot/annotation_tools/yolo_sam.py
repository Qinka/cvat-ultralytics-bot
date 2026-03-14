"""YOLO + SAM annotation tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import Detection

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class YoloSamTool:
    """YOLO detection followed by SAM refinement."""

    yolo_weights: str
    sam_weights: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        from ultralytics import SAM, YOLO

        self._yolo = YOLO(self.yolo_weights)
        self._sam = SAM(self.sam_weights)
        self.tool_name = "yolo_sam"

    def predict(self, image: "Image", conf: float = 0.25) -> list[Detection]:
        yolo_results = self._yolo.predict(image, conf=conf, device=self.device, verbose=False)
        detections: list[Detection] = []
        for result in yolo_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            bboxes = result.boxes.xyxy.tolist()
            cls_ids = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()
            sam_results = self._sam(image, bboxes=bboxes, device=self.device, verbose=False)
            for index, (sam_result, cls_id, confidence) in enumerate(zip(sam_results, cls_ids, confs)):
                polygon: list[float] | None = None
                if sam_result.masks is not None and len(sam_result.masks.xy) > 0:
                    pts: np.ndarray = sam_result.masks.xy[0]
                    polygon = pts.flatten().tolist()
                detections.append(
                    Detection(
                        class_name=self._yolo.names[cls_id],
                        confidence=float(confidence),
                        bbox_xyxy=bboxes[index],
                        polygon_xy=polygon,
                    )
                )
        return detections


def build_tool(config: dict[str, Any]):
    return YoloSamTool(
        yolo_weights=str(config["yolo_weights"]),
        sam_weights=str(config["sam_weights"]),
        device=str(config.get("device", "cpu")),
    )


register_tool(
    AnnotationToolRegistration(
        name="yolo_sam",
        factory=build_tool,
        description="Ultralytics YOLO + SAM tool",
        use_polygon=True,
    )
)
