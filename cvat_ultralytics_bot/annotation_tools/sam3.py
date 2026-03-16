"""SAM3 annotation tool using ModelScope/Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import PredictedObject

if TYPE_CHECKING:
    from PIL.Image import Image


# TODO: 已知问题，标注会存在一个目标多个框子的情况，这个应该是分割转


@dataclass
class Sam3Tool:
    """SAM3 image segmentation with automatic mask generation."""

    weights: str
    device: str = "cpu"
    label_prompts: dict[str, str] | None = None
    use_polygon: bool = True
    threshold: float = 0.5

    def __post_init__(self) -> None:
        from modelscope import Sam3Processor, Sam3Model

        self._device = self.device if self.device != "mps" else "cpu"
        self._model = Sam3Model.from_pretrained(self.weights).to(self._device)
        self._processor = Sam3Processor.from_pretrained(self.weights)
        self.tool_name = "sam3"


    def _apply_label_map(self, class_name: str) -> str:
        """Apply label mapping if configured."""
        return class_name

    def predict_text(self, image_size, vision_embeds, prompt: str, conf: float = 0.25) -> list[dict]:
        import torch

        # print(f"[DEBUG] Predicting for prompt: '{prompt}' with confidence threshold: {conf}")
        text_inputs = self._processor(text=prompt, return_tensors="pt").to(self._device)

        # breakpoint()

        with torch.no_grad():
            outputs = self._model(vision_embeds=vision_embeds, **text_inputs)

        # Post-process to get instance segmentation results
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=conf,
            mask_threshold=0.5,
            target_sizes=[image_size[::-1]],  # (height, width) -> (width, height)
        )

        # if prompt == 'vehicle':
        #     breakpoint()
        #     print(f"[DEBUG] Raw results for prompt '{prompt}': {results}")

        return results[0]


    def predict(self, image: "Image", conf: float = 0.25) -> list[PredictedObject]:
        import torch

        img_inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            vision_embeds = self._model.get_vision_features(pixel_values=img_inputs.pixel_values)

        results = dict([ (label, self.predict_text(image.size, vision_embeds, self.label_prompts[label], conf)) for label in self.label_prompts])
        # breakpoint()

        predictions: list[PredictedObject] = []

        for label, result in results.items():


            if "masks" in result and result["masks"] is not None:
                masks = result["masks"]
                boxes = result.get("boxes", [])
                scores = result.get("scores", torch.ones(len(masks)))

                # print(boxes)
                # breakpoint()

                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    # Get bounding box from mask
                    mask_np = mask.cpu().numpy()
                    if len(mask_np.shape) == 2:
                        rows = np.any(mask_np, axis=1)
                        cols = np.any(mask_np, axis=0)
                    else:
                        rows = np.any(mask_np[0], axis=1)
                        cols = np.any(mask_np[0], axis=0)

                    if not np.any(rows) or not np.any(cols):
                        continue

                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    # Convert mask to polygon
                    if self.use_polygon:
                        polygon = self._mask_to_polygon(mask_np, bbox)
                    else:
                        polygon = None

                    confidence = float(scores[i].item()) if i < len(scores) else 1.0

                    predictions.append(
                        PredictedObject(
                            class_name=label,
                            confidence=confidence,
                            bbox_xyxy=bbox,
                            polygon_xy=polygon,
                        )
                    )

        return predictions

    def _mask_to_polygon(self, mask_np: np.ndarray, bbox: list[float]) -> list[float]:
        """Convert mask to polygon contour."""
        import cv2

        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]

        mask_uint8 = (mask_np * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Use the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) >= 3:
                return largest_contour.squeeze().flatten().tolist()

        # Fallback: return bbox as polygon
        return [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]


def build_tool(config: dict[str, Any]):
    # label_map = config.get("label_map")
    return Sam3Tool(
        weights=str(config["weights"]),
        device=str(config.get("device", "cpu")),
        label_prompts=dict(config.get("label_prompts", None)),
        # label_map=dict(label_map) if label_map else None,
        use_polygon=bool(config.get("use_polygon", True)),
        threshold=float(config.get("threshold", 0.5)),
    )


register_tool(
    AnnotationToolRegistration(
        name="sam3",
        factory=build_tool,
        description="SAM3 image segmentation tool using ModelScope/Transformers",
        use_polygon=True,
    )
)

