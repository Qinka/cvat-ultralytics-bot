"""YOLO detection annotation tool.

This module provides a wrapper around Ultralytics YOLO models for
object detection tasks. The tool takes images and returns bounding
box predictions with class labels and confidence scores.

Example:
    >>> tool = YoloDetectTool(weights="yolov8n.pt", device="cpu")
    >>> predictions = tool.predict(image, conf=0.25)
    >>> for pred in predictions:
    ...     print(pred.class_name, pred.confidence, pred.bbox_xyxy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import PredictedObject

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class YoloDetectTool:
    """Ultralytics YOLO object detector.

    This tool wraps an Ultralytics YOLO model for object detection.
    It provides a simple interface to run inference on images and
    returns predictions as :class:`PredictedObject` instances.

    Attributes:
        weights: Path to YOLO weights file (e.g., ``yolov8n.pt``).
        device: Device to run inference on (e.g., ``"cpu"``, ``"cuda:0"``).

    Note:
        The model is loaded lazily on first prediction to avoid
        blocking the main thread during initialization.
    """

    weights: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.weights)
        self.tool_name = "yolo_detect"

    def predict(self, image: "Image", conf: float = 0.25) -> list[PredictedObject]:
        """Run object detection on an image.

        Args:
            image: Input image as PIL Image.
            conf: Confidence threshold for predictions.

        Returns:
            List of :class:`PredictedObject` instances, each containing:
            - ``class_name``: Predicted class label.
            - ``confidence``: Prediction confidence score.
            - ``bbox_xyxy``: Bounding box in xyxy format.
        """


        # print("yolo_detect annotation tool running...")

        results = self._model.predict(image, conf=conf, device=self.device, verbose=False)
        predictions: list[PredictedObject] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                predictions.append(
                    PredictedObject(
                        class_name=self._model.names[cls_id],
                        confidence=float(boxes.conf[index].item()),
                        bbox_xyxy=boxes.xyxy[index].tolist(),
                    )
                )
        return predictions


def build_tool(config: dict[str, Any]) -> YoloDetectTool:
    """Build a YoloDetectTool from configuration dictionary.

    Args:
        config: Dictionary with ``weights`` and optional ``device`` keys.

    Returns:
        Configured YoloDetectTool instance.
    """
    return YoloDetectTool(weights=str(config["weights"]), device=str(config.get("device", "cpu")))


register_tool(
    AnnotationToolRegistration(
        name="yolo_detect",
        factory=build_tool,
        description="Ultralytics YOLO detection tool",
        use_polygon=False,
    )
)
