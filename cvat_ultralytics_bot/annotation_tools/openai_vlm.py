"""OpenAI-compatible VLM annotation tool."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import PredictedObject

if TYPE_CHECKING:
    from PIL.Image import Image


SYSTEM_PROMPT = """You are an expert object detector. Analyze the image and return detected objects as bounding boxes.
Respond in the following JSON format:
```json
{
  "objects": [
    {"class_name": "object_class", "bbox_xyxy": [x1, y1, x2, y2], "confidence": 0.95},
    ...
  ]
}
```
IMPORTANT: bbox_xyxy must be in PIXEL COORDINATES (normalized 0-1), where (x1, y1) is top-left and (x2, y2) is bottom-right corner.
The image has specific dimensions - use the actual pixel coordinates based on the image size.
Be precise and only detect actual objects, do not hallucinate."""


@dataclass
class OpenAIVLMTool:
    """OpenAI-compatible VLM object detector."""

    api_base: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    timeout: int = 120
    label_map: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self.tool_name = "openai_vlm"

    def _encode_image(self, image: "Image") -> str:
        """Encode PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _call_api(self, image: "Image") -> list[dict[str, Any]]:
        """Call the VLM API with the image."""
        import json

        base64_image = self._encode_image(image)
        width, height = image.size

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Detect all objects in this image. The image size is {width}x{height} pixels. Return bounding boxes in pixel coordinates.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                },
            ],
            "temperature": self.temperature,
        }

        response = requests.post(
            f"{self.api_base.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        print("API response:", response.text)

        content = response.json()["choices"][0]["message"]["content"]

        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())
        print(image.size, '\n', data)
        return data.get("objects", [])

    def predict(self, image: "Image", conf: float = 0.5) -> list[PredictedObject]:
        """Run inference and return predictions."""
        width, height = image.size
        raw_objects = self._call_api(image)
        predictions: list[PredictedObject] = []

        w = (width + 31) // 32 * 32
        h = (height + 31) // 32 * 32
        # breakpoint()

        for obj in raw_objects:
            confidence = obj.get("confidence", 1.0)
            if confidence < conf:
                continue

            class_name = obj.get("class_name", "unknown")
            bbox = obj.get("bbox_xyxy", [])

            # Apply label map if provided
            if self.label_map and class_name in self.label_map:
                class_name = self.label_map[class_name]

            if len(bbox) == 4:
                # Check if coordinates are normalized (0-1 range) and convert to pixels
                x1, y1, x2, y2 = bbox
                # if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                #     # Convert from normalized (0-1) to pixel coordinates
                #     x1 = int(x1 * width)
                #     y1 = int(y1 * height)
                #     x2 = int(x2 * width)
                #     y2 = int(y2 * height)
                # else:
                #     # Already in pixel coordinates, ensure integers
                x1, y1, x2, y2 = int(x1/1000*w), int(y1/1000*h), int(x2/1000*w), int(y2/1000*h)

                predictions.append(
                    PredictedObject(
                        class_name=class_name,
                        confidence=confidence,
                        bbox_xyxy=[x1, y1, x2, y2],
                    )
                )

        return predictions


def build_tool(config: dict[str, Any]) -> OpenAIVLMTool:
    """Build an OpenAI VLM tool from configuration."""
    label_map = config.get("label_map")
    return OpenAIVLMTool(
        api_base=str(config["api_base"]),
        model=str(config["model"]),
        api_key=config.get("api_key"),
        temperature=float(config.get("temperature", 0.0)),
        timeout=int(config.get("timeout", 120)),
        label_map=dict(label_map) if label_map else None,
    )


register_tool(
    AnnotationToolRegistration(
        name="openai_vlm",
        factory=build_tool,
        description="OpenAI-compatible VLM object detection tool",
        use_polygon=False,
    )
)
