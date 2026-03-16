"""DINOv3-based object detection tool using ModelScope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import PredictedObject

if TYPE_CHECKING:
    from PIL.Image import Image


# Grounding DINO task name in ModelScope
GROUNDING_DINO_TASK = "object-detection" # "grounding_dino"


@dataclass
class Dinov3DetectTool:
    """DINOv3-based object detector using ModelScope."""

    model_id: str
    device: str = "cpu"
    conf_threshold: float = 0.25

    def __post_init__(self) -> None:
        from modelscope import pipeline



        self._pipeline = pipeline(
            task=GROUNDING_DINO_TASK,
            model=self.model_id,
            device=self.device,
        )

        self.tool_name = "dinov3_detect"

    def predict(self, image: "Image", conf: float = 0.25) -> list[PredictedObject]:
        """Run inference and return predictions."""
        import numpy as np

        # Convert PIL Image to numpy array for ModelScope
        img_array = np.array(image)

        # Run detection with text prompt
        # Using empty prompt to detect all objects, or customize as needed
        breakpoint()
        result = self._pipeline(
            img_array,
            text_prompt="object",
        )

        predictions: list[PredictedObject] = []

        # Parse results - format depends on ModelScope output
        if result is not None:
            # ModelScope grounding dino typically returns boxes and labels
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            labels = result.get("labels", [])

            for box, score, label in zip(boxes, scores, labels):
                if score < conf:
                    continue

                # Box format: [x1, y1, x2, y2]
                if len(box) >= 4:
                    predictions.append(
                        PredictedObject(
                            class_name=str(label) if label else "object",
                            confidence=float(score),
                            bbox_xyxy=[float(b) for b in box[:4]],
                        )
                    )

        return predictions


def build_tool(config: dict[str, Any]) -> Dinov3DetectTool:
    """Build a DINOv3 detection tool from configuration."""
    return Dinov3DetectTool(
        model_id=str(config.get("model_id", "facebook/dinov3-vit7b16-pretrain-lvd1689m")),
        device=str(config.get("device", "cpu")),
        conf_threshold=float(config.get("conf_threshold", 0.25)),
    )


register_tool(
    AnnotationToolRegistration(
        name="dinov3_detect",
        factory=build_tool,
        description="DINOv3-based object detection tool using ModelScope",
        use_polygon=False,
    )
)
