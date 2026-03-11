"""Model wrappers for YOLO (detection/segmentation) and SAM."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


class ModelType(str, Enum):
    """Supported model types."""

    YOLO_DETECT = "yolo-detect"
    YOLO_SEGMENT = "yolo-segment"
    YOLO_SAM = "yolo-sam"


@dataclass
class Detection:
    """A single detection result from a model.

    Attributes:
        class_name: Model class name string.
        confidence: Detection confidence score in [0, 1].
        bbox_xyxy: Bounding box as [x1, y1, x2, y2] in pixel coordinates.
        polygon_xy: Optional polygon points as a flat list [x1, y1, x2, y2, ...].
    """

    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    polygon_xy: list[float] | None = field(default=None)


class YoloDetector:
    """Wrapper around ultralytics YOLO for object detection.

    Produces :class:`Detection` instances with ``bbox_xyxy`` only.
    """

    def __init__(self, weights: str, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._device = device

    @property
    def class_names(self) -> dict[int, str]:
        return self._model.names  # type: ignore[return-value]

    def predict(self, image: "Image", conf: float = 0.25) -> list[Detection]:
        """Run detection inference on *image*.

        Args:
            image: A PIL Image to run inference on.
            conf: Confidence threshold.

        Returns:
            List of :class:`Detection` results.
        """
        results = self._model.predict(image, conf=conf, device=self._device, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                detections.append(
                    Detection(
                        class_name=self._model.names[cls_id],
                        confidence=float(boxes.conf[i].item()),
                        bbox_xyxy=boxes.xyxy[i].tolist(),
                    )
                )
        return detections


class YoloSegmentor:
    """Wrapper around ultralytics YOLO for instance segmentation.

    Produces :class:`Detection` instances with both ``bbox_xyxy`` and
    ``polygon_xy``.
    """

    def __init__(self, weights: str, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._device = device

    @property
    def class_names(self) -> dict[int, str]:
        return self._model.names  # type: ignore[return-value]

    def predict(self, image: "Image", conf: float = 0.25) -> list[Detection]:
        """Run segmentation inference on *image*.

        Args:
            image: A PIL Image to run inference on.
            conf: Confidence threshold.

        Returns:
            List of :class:`Detection` results. Each result includes
            ``polygon_xy`` with the instance mask contour points.
        """
        results = self._model.predict(image, conf=conf, device=self._device, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            masks = result.masks
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                polygon: list[float] | None = None
                if masks is not None and i < len(masks.xy):
                    pts = masks.xy[i]  # numpy array, shape (N, 2)
                    polygon = pts.flatten().tolist()
                detections.append(
                    Detection(
                        class_name=self._model.names[cls_id],
                        confidence=float(boxes.conf[i].item()),
                        bbox_xyxy=boxes.xyxy[i].tolist(),
                        polygon_xy=polygon,
                    )
                )
        return detections


class YoloSamSegmentor:
    """Combined YOLO detection + SAM segmentation pipeline.

    First runs YOLO to detect objects, then uses SAM to produce precise
    segmentation masks guided by the YOLO bounding boxes.
    """

    def __init__(
        self,
        yolo_weights: str,
        sam_weights: str,
        device: str = "cpu",
    ) -> None:
        from ultralytics import SAM, YOLO

        self._yolo = YOLO(yolo_weights)
        self._sam = SAM(sam_weights)
        self._device = device

    @property
    def class_names(self) -> dict[int, str]:
        return self._yolo.names  # type: ignore[return-value]

    def predict(self, image: "Image", conf: float = 0.25) -> list[Detection]:
        """Run YOLO+SAM inference on *image*.

        First detects objects with YOLO, then refines each detection into a
        precise polygon using SAM.

        Args:
            image: A PIL Image to run inference on.
            conf: Confidence threshold for the YOLO stage.

        Returns:
            List of :class:`Detection` results with polygon masks.
        """
        yolo_results = self._yolo.predict(image, conf=conf, device=self._device, verbose=False)
        detections: list[Detection] = []

        for result in yolo_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            bboxes = result.boxes.xyxy.tolist()
            cls_ids = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()

            sam_results = self._sam(
                image,
                bboxes=bboxes,
                device=self._device,
                verbose=False,
            )
            for idx, (sam_res, cls_id, confidence) in enumerate(
                zip(sam_results, cls_ids, confs)
            ):
                polygon: list[float] | None = None
                if sam_res.masks is not None and len(sam_res.masks.xy) > 0:
                    pts: np.ndarray = sam_res.masks.xy[0]
                    polygon = pts.flatten().tolist()
                detections.append(
                    Detection(
                        class_name=self._yolo.names[cls_id],
                        confidence=float(confidence),
                        bbox_xyxy=bboxes[idx],
                        polygon_xy=polygon,
                    )
                )
        return detections
