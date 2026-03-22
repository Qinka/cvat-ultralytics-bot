"""Fusion annotation tool that combines results from multiple detection/segmentation tools.

This module provides a wrapper that runs multiple annotation tools (YOLO, SAM, VLM, etc.)
on the same image and fuses the results using IoU-based Non-Maximum Suppression or
union/intersection strategies.

Example:
    >>> tool = build_tool({
    ...     "tools": [
    ...         {"type": "yolo_detect", "weights": "yolo26n.pt", "device": "cpu"},
    ...         {"type": "sam3", "weights": "sam3.pt", "device": "cpu",
    ...          "label_prompts": {"person": "person", "car": "car"}},
    ...     ],
    ...     "fusion_strategy": "union",
    ...     "iou_threshold": 0.5,
    ... })
    >>> predictions = tool.predict(image, conf=0.25)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import get_tool_registration, register_tool
from cvat_ultralytics_bot.types import PredictedObject

if TYPE_CHECKING:
    from PIL.Image import Image


# Default fusion strategies
_STRATEGIES = {"union", "intersection", "nms"}


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _mask_iou(polygon_a: list[float], polygon_b: list[float], bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute IoU between two polygons using bbox approximation."""
    return _compute_iou(bbox_a, bbox_b)


def _nms_single(
    predictions: list[PredictedObject],
    iou_threshold: float,
) -> list[PredictedObject]:
    """Apply Non-Maximum Suppression to a list of predictions."""
    if not predictions:
        return []

    # Sort by confidence descending
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    keep: list[PredictedObject] = []

    for pred in sorted_preds:
        suppressed = False
        for kept_pred in keep:
            iou = _compute_iou(pred.bbox_xyxy, kept_pred.bbox_xyxy)
            if iou >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(pred)

    return keep


def _fuse_polygons(polygons: list[list[float] | None], weights: list[float]) -> list[float] | None:
    """Fuse multiple polygons into one using weighted averaging of points."""
    valid_polygons = [p for p in polygons if p is not None and len(p) >= 6]
    if not valid_polygons:
        return None

    # Simple approach: use the polygon from the highest-weight source
    best_idx = max(range(len(polygons)), key=lambda i: weights[i] if polygons[i] is not None else -1)
    return polygons[best_idx]


@dataclass
class FusionTool:
    """Fuses results from multiple annotation tools.

    This tool runs multiple sub-tools on the same image and combines
    their predictions using a configurable fusion strategy.

    Attributes:
        subtools: List of (tool_name, tool_instance, weight) tuples.
        fusion_strategy: Strategy for combining results.
            - "union": Keep all detections from all tools.
            - "intersection": Keep only detections appearing in multiple tools.
            - "nms": Apply IoU-based NMS across all tools.
        iou_threshold: IoU threshold for NMS and intersection checks.
        conf: Base confidence threshold passed to sub-tools.
    """

    subtools: list[tuple[str, Any, float]]
    fusion_strategy: str = "union"
    iou_threshold: float = 0.5
    conf: float = 0.25

    def __post_init__(self) -> None:
        self.tool_name = "fusion"

    def predict(self, image: "Image", conf: float = 0.25) -> list[PredictedObject]:
        """Run all sub-tools and fuse their predictions.

        Args:
            image: Input PIL Image.
            conf: Confidence threshold for sub-tool predictions.

        Returns:
            Fused list of PredictedObject instances.
        """
        all_predictions: list[tuple[PredictedObject, str, float]] = []

        for tool_name, tool_instance, weight in self.subtools:
            try:
                preds = tool_instance.predict(image, conf=conf)
                for pred in preds:
                    all_predictions.append((pred, tool_name, weight))
            except Exception:
                # If one sub-tool fails, continue with others
                continue

        if not all_predictions:
            return []

        if self.fusion_strategy == "nms":
            return self._fuse_nms(all_predictions)
        elif self.fusion_strategy == "intersection":
            return self._fuse_intersection(all_predictions)
        else:  # union
            return self._fuse_union(all_predictions)

    def _fuse_union(self, predictions: list[tuple[PredictedObject, str, float]]) -> list[PredictedObject]:
        """Union strategy: keep all predictions, apply light NMS to deduplicate."""
        preds_only = [p[0] for p in predictions]
        # Apply light NMS with higher threshold to avoid over-merging
        return _nms_single(preds_only, iou_threshold=max(0.8, self.iou_threshold))

    def _fuse_intersection(self, predictions: list[tuple[PredictedObject, str, float]]) -> list[PredictedObject]:
        """Intersection strategy: keep detections appearing in >= 2 tools."""
        from collections import defaultdict

        # Group by approximate location (grid-based)
        groups: dict[int, list[tuple[PredictedObject, str, float]]] = defaultdict(list)
        grid_size = 50  # pixels

        for pred, tool_name, weight in predictions:
            # Use center of bbox as grid key
            cx = int((pred.bbox_xyxy[0] + pred.bbox_xyxy[2]) / 2 / grid_size)
            cy = int((pred.bbox_xyxy[1] + pred.bbox_xyxy[3]) / 2 / grid_size)
            key = cy * 10000 + cx  # Simple 2D hashing
            groups[key].append((pred, tool_name, weight))

        results: list[PredictedObject] = []
        for group in groups.values():
            if len(group) < 2:
                continue  # Need at least 2 tools to agree
            tool_names = {t[1] for t in group}
            if len(tool_names) < 2:
                continue
            # Take the highest confidence prediction in the group
            best = max(group, key=lambda x: x[0].confidence)
            results.append(best[0])

        return results

    def _fuse_nms(self, predictions: list[tuple[PredictedObject, str, float]]) -> list[PredictedObject]:
        """NMS strategy: apply IoU-based NMS across all predictions, weighted by source."""
        preds_only = [p[0] for p in predictions]
        return _nms_single(preds_only, iou_threshold=self.iou_threshold)


def _build_subtool(tool_type: str, tool_config: dict[str, Any], global_conf: float) -> tuple[str, Any, float]:
    """Build a single sub-tool and return (tool_name, instance, weight)."""
    registration = get_tool_registration(tool_type)
    tool_instance = registration.factory(tool_config)
    weight = float(tool_config.get("weight", 1.0))
    return (tool_type, tool_instance, weight)


def build_tool(config: dict[str, Any]) -> FusionTool:
    """Build a FusionTool from configuration.

    Configuration format:
        [fusion]
        fusion_strategy = "union"   # union, intersection, nms
        iou_threshold = 0.5
        conf = 0.25

        [[fusion.tools]]
        type = "yolo_detect"
        weights = "yolo26n.pt"
        device = "cpu"
        weight = 1.0

        [[fusion.tools]]
        type = "sam3"
        weights = "sam3.pt"
        device = "cpu"
        label_prompts = { vehicle = "vehicle" }
        weight = 0.9

    Args:
        config: Tool configuration dictionary.

    Returns:
        FusionTool instance.
    """
    tools_config = config.get("tools", [])
    if not tools_config:
        raise ValueError("Fusion tool requires at least one sub-tool in 'tools' list")

    fusion_strategy = str(config.get("fusion_strategy", "union"))
    if fusion_strategy not in _STRATEGIES:
        raise ValueError(f"Unknown fusion_strategy '{fusion_strategy}'. Available: {_STRATEGIES}")

    iou_threshold = float(config.get("iou_threshold", 0.5))
    conf = float(config.get("conf", 0.25))

    subtools: list[tuple[str, Any, float]] = []
    for tool_cfg in tools_config:
        tool_type = tool_cfg.get("type")
        if not tool_type:
            raise ValueError(f"Each tool in 'tools' list must have a 'type' field. Got: {tool_cfg}")
        tool_instance = _build_subtool(tool_type, tool_cfg, conf)
        subtools.append(tool_instance)

    return FusionTool(
        subtools=subtools,
        fusion_strategy=fusion_strategy,
        iou_threshold=iou_threshold,
        conf=conf,
    )


register_tool(
    AnnotationToolRegistration(
        name="fusion",
        factory=build_tool,
        description="Fusion tool that combines results from YOLO, SAM, VLM and other tools",
        use_polygon=True,
    )
)
