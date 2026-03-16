"""YOLO + SAM annotation tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import PredictedObject

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class YoloSamTool:
    """YOLO detection followed by SAM refinement."""

    yolo_weights: str
    sam_weights: str
    device: str = "cpu"
    label_map: dict[str, str] | None = None
    use_polygon: bool = True

    def __post_init__(self) -> None:
        from ultralytics import SAM, YOLO

        self._yolo = YOLO(self.yolo_weights)
        self._sam = SAM(self.sam_weights)
        self.tool_name = "yolo_sam"

    def _apply_label_map(self, class_name: str) -> str:
        """Apply label mapping if configured."""
        if self.label_map and class_name in self.label_map:
            return self.label_map[class_name]
        return class_name

    def predict(self, image: "Image", conf: float = 0.25) -> list[PredictedObject]:
        yolo_results = self._yolo.predict(image, conf=conf, device=self.device, verbose=False)
        predictions: list[PredictedObject] = []

        # Debug: print YOLO model class names
        # print(f"[DEBUG] YOLO class names: {self._yolo.names}")

        for result in yolo_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            bboxes = result.boxes.xyxy.tolist()
            cls_ids = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()
            sam_results = self._sam(image, bboxes=bboxes, device=self.device, verbose=False)

            # breakpoint()  # Debug: inspect sam_results and corresponding bboxes/cls_ids/confs

            # for sam_result, bbox, cls_id, confidence in zip(sam_results, bboxes, cls_ids, confs):


            # Debug: print class info
            # print(f"[DEBUG] cls_id={cls_id}, cls_id={cls_id}, label_map={self.label_map}")

            # Use polygon (instance segmentation)

            for mask_xy, cls_id, bbox, confidence in zip(sam_results[0].masks.xy, cls_ids, bboxes, confs):
                pts: np.ndarray = mask_xy

                if self.use_polygon:
                    polygon = pts.flatten().tolist()
                else:
                    polygon = None

                class_name = self._yolo.names[cls_id]
                predictions.append(
                    PredictedObject(
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox_xyxy=bbox,
                        polygon_xy=polygon,
                    )
                )

        return predictions


def build_tool(config: dict[str, Any]):
    label_map = config.get("label_map")
    return YoloSamTool(
        yolo_weights=str(config["yolo_weights"]),
        sam_weights=str(config["sam_weights"]),
        device=str(config.get("device", "cpu")),
        label_map=dict(label_map) if label_map else None,
        use_polygon=bool(config.get("use_polygon", True)),
    )


register_tool(
    AnnotationToolRegistration(
        name="yolo_sam",
        factory=build_tool,
        description="Ultralytics YOLO + SAM tool",
        use_polygon=True,
    )
)
