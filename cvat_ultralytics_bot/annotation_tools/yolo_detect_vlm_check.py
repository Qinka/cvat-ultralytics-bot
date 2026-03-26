"""YOLO detection with VLM verification annotation tool.

This module provides a wrapper around Ultralytics YOLO models for
object detection tasks, followed by VLM-based verification of each
detection to filter out false positives.

The tool works in two stages:
1. Run YOLO detection with an extremely low confidence threshold to
   catch as many potential objects as possible.
2. For each detected bounding box, query a VLM (e.g., GPT-4V, LLaVA)
   to verify whether the detection is correct.

Example:
    >>> tool = YoloDetectVLMTool(
    ...     weights="yolo26n.pt",
    ...     yolo_conf=0.01,
    ...     api_base="http://localhost:11434/v1",
    ...     model="llava",
    ... )
    >>> predictions = tool.predict(image, conf=0.25)
    >>> for pred in predictions:
    ...     print(pred.class_name, pred.confidence, pred.bbox_xyxy)
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration
from cvat_ultralytics_bot.annotation_tools.registry import register_tool
from cvat_ultralytics_bot.types import PredictedObject

if TYPE_CHECKING:
    import requests
    from PIL.Image import Image


SYSTEM_PROMPT = """
# 身份
你是一个专业的图片描述者和目标检测专家，深刻理解各种场景中的任务和车辆的特征。

# 任务
给定图像中的检测框列表，每个检测框包含类别、边界框坐标和原始置信度。你的任务是：
1. 检查每个检测框是否正确包含了对应类别的对象
2. 如果检测不正确（空框、误标、包含多个对象等），返回 null
3. 如果检测正确但类别或位置需要调整，返回修正后的类别、边界框和置信度
4. 如果漏检了某个对象，在结果中补充新的检测（可选）

# 响应格式

请严格按以下JSON格式返回结果（所有检测框按原始顺序）：
```json
{
  "results": [
    {
      "index": 0,
      "is_correct": true或false,
      "corrected": {
        "class_name": "修正后的类别，如果没有修正则与原类别相同",
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": 0.0到1.0之间的置信度
      },
      "reason": "简要说明"
    }
  ]
}
```

如果 is_correct 为 false 且不需要补充检测，则 corrected 为 null。
如果 is_correct 为 true 但类别或位置有误，corrected 包含修正后的值。
"""


@dataclass
class YoloDetectVLMTool:
    """YOLO object detector with VLM verification.

    This tool combines YOLO detection (run with low threshold) and
    VLM verification to filter out false positives. Each detection from
    YOLO is verified by the VLM before being included in the final results.

    Attributes:
        weights: Path to YOLO weights file (e.g., ``yolo26n.pt``).
        yolo_conf: YOLO detection confidence threshold (set very low, e.g., 0.01).
        device: Device to run YOLO inference on (e.g., ``"cpu"``, ``"cuda:0"``).
        api_base: OpenAI-compatible API base URL for VLM (e.g., ``http://localhost:11434/v1``).
        model: VLM model name (e.g., ``"llava"``, ``"gpt-4o"``).
        api_key: Optional API key for VLM authentication.
        temperature: Temperature for VLM generation.
        timeout: Timeout in seconds for VLM API calls.
        label_map: Optional mapping from YOLO class names to desired labels.
    """

    weights: str
    yolo_conf: float = 0.01
    device: str = "cpu"
    api_base: str = "http://localhost:11434/v1"
    model: str = "llava"
    api_key: str | None = None
    temperature: float = 0.0
    timeout: int = 120
    label_map: dict[str, str] | None = None

    def __post_init__(self) -> None:
        from ultralytics import YOLO

        self._yolo = YOLO(self.weights)
        self.tool_name = "yolo_detect_vlm_check"

    def _encode_image(self, image: "Image") -> str:
        """Encode PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _verify_batch_with_vlm(
        self,
        image: "Image",
        detections: list[dict],
    ) -> list[dict | None]:
        """Verify and correct multiple detections with VLM in a single call.

        Args:
            image: Input image as PIL Image.
            detections: List of dicts with keys: bbox_xyxy, class_name, yolo_confidence.

        Returns:
            List of dicts with keys: is_correct, corrected (dict with class_name, bbox_xyxy, confidence, or None if incorrect).
        """
        import requests

        if not detections:
            return []

        base64_image = self._encode_image(image)
        width, height = image.size

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Build prompt with all detections
        detection_list_text = "\n".join([
            f"- 检测框 {i+1}: 类别='{d['class_name']}', "
            f"坐标=[{d['bbox_xyxy'][0]:.0f}, {d['bbox_xyxy'][1]:.0f}, "
            f"{d['bbox_xyxy'][2]:.0f}, {d['bbox_xyxy'][3]:.0f}], "
            f"YOLO置信度={d['yolo_confidence']:.3f}"
            for i, d in enumerate(detections)
        ])

        # Build JSON format example (avoiding f-string brace issues)
        json_example = (
            '{\n'
            '  "results": [\n'
            '    {"index": 0, "is_correct": true, "corrected": {"class_name": "类别", "bbox_xyxy": [x1,y1,x2,y2], "confidence": 0.0}},\n'
            '    {"index": 1, "is_correct": false, "corrected": null, "reason": "误标"},\n'
            '    ...\n'
            '  ]\n'
            '}'
        )

        verification_prompt = (
            f"图像尺寸为 {width}x{height} 像素。\n"
            f"YOLO 模型在图像中检测到了 {len(detections)} 个目标：\n"
            f"{detection_list_text}\n\n"
            f"请逐一检查每个边界框，判断其中是否真的包含对应类别的对象。\n"
            f"如果检测不正确（空框、误标、包含多个对象等），返回 is_correct=false 且 corrected=null。\n"
            f"如果检测正确但类别或位置需要调整，在 corrected 中返回修正后的值。\n\n"
            f"请严格按以下JSON格式返回结果（所有检测框按原始顺序）：\n"
            + json_example
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": verification_prompt,
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

        # Determine the correct endpoint based on api_base
        base = self.api_base.rstrip("/")
        if "/v1" in base:
            endpoint = f"{base}/chat/completions"
        else:
            endpoint = f"{base}/api/chat"

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())
            results = data.get("results", [])

            # Map results back to original detection order
            result_map = {r["index"]: r for r in results}
            verified = []
            for i, det in enumerate(detections):
                r = result_map.get(i, {"is_correct": False, "corrected": None, "reason": "No result returned"})
                is_correct = r.get("is_correct", False)
                corrected = r.get("corrected")
                reason = r.get("reason", "")

                if is_correct and corrected:
                    print(f"[VLM] bbox={det['bbox_xyxy']} class={det['class_name']} conf={det['yolo_confidence']:.3f} -> ACCEPTED (corrected: class={corrected.get('class_name')}, conf={corrected.get('confidence')})")
                elif is_correct and not corrected:
                    print(f"[VLM] bbox={det['bbox_xyxy']} class={det['class_name']} conf={det['yolo_confidence']:.3f} -> ACCEPTED (original)")
                else:
                    print(f"[VLM] bbox={det['bbox_xyxy']} class={det['class_name']} conf={det['yolo_confidence']:.3f} -> REJECTED: {reason}")

                verified.append({
                    "is_correct": is_correct,
                    "corrected": corrected,
                })
            return verified
        except Exception as e:
            print(f"[VLM] Verification failed: {e}")
            # On error, reject all detections
            return [{"is_correct": False, "corrected": None} for _ in detections]

    def _compute_iou(self, bbox_a: list[float], bbox_b: list[float]) -> float:
        """Compute IoU between two bboxes in xyxy format."""
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _nms(self, predictions: list[PredictedObject], iou_threshold: float = 0.5) -> list[PredictedObject]:
        """Apply Non-Maximum Suppression to predictions."""
        if not predictions:
            return []

        # Sort by confidence descending
        sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
        keep: list[PredictedObject] = []

        for pred in sorted_preds:
            suppressed = False
            for kept_pred in keep:
                iou = self._compute_iou(pred.bbox_xyxy, kept_pred.bbox_xyxy)
                if iou >= iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(pred)

        return keep

    def predict(self, image: "Image", conf: float = 0.25) -> list[PredictedObject]:
        """Run YOLO detection with VLM verification and correction.

        Workflow:
        1. Run YOLO detection with low threshold
        2. Filter detections by label_map (only keep matching classes)
        3. Separate into high-confidence (keep) and low-confidence (send to VLM)
        4. VLM returns corrected class, bbox, confidence, or null if wrong
        5. Merge results and apply NMS

        Args:
            image: Input image as PIL Image.
            conf: Final confidence threshold.

        Returns:
            List of :class:`PredictedObject` instances after VLM correction and NMS.
        """
        # Get image size for YOLO input
        # Use the longer edge to maintain aspect ratio, rounded to nearest multiple of 32
        orig_width, orig_height = image.size
        max_dim = max(orig_width, orig_height)
        imgsz = (max_dim // 32) * 32  # Round down to nearest multiple of 32

        # Stage 1: Run YOLO with low threshold, using actual image size
        results = self._yolo.predict(image, conf=self.yolo_conf, device=self.device, imgsz=imgsz, verbose=False)

        # Collect all detections after label_map filtering
        all_detections: list[dict] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                yolo_confidence = float(boxes.conf[index].item())
                bbox_xyxy = boxes.xyxy[index].tolist()

                class_name = self._yolo.names[cls_id]

                # Apply label_map filtering - skip if not in label_map
                if self.label_map and class_name not in self.label_map:
                    continue
                # If label_map has values, use mapped class name
                if self.label_map and class_name in self.label_map:
                    class_name = self.label_map[class_name]

                all_detections.append({
                    "bbox_xyxy": bbox_xyxy,
                    "class_name": class_name,
                    "yolo_confidence": yolo_confidence,
                    "original_class": class_name,
                })

        # Separate into high and low confidence for VLM processing
        vlm_conf_threshold = 0.5  # Above this keep without VLM, below send to VLM
        high_conf_detections = [d for d in all_detections if d["yolo_confidence"] >= vlm_conf_threshold]
        low_conf_detections = [d for d in all_detections if d["yolo_confidence"] < vlm_conf_threshold]

        print(f"[YOLO] Total: {len(all_detections)}, High conf (keep): {len(high_conf_detections)}, Low conf (VLM): {len(low_conf_detections)}")

        # Stage 2: Keep high confidence detections directly
        predictions: list[PredictedObject] = [
            PredictedObject(
                class_name=d["class_name"],
                confidence=d["yolo_confidence"],
                bbox_xyxy=d["bbox_xyxy"],
            )
            for d in high_conf_detections
        ]

        # Stage 3: Send low confidence detections to VLM for correction
        if low_conf_detections:
            vlm_results = self._verify_batch_with_vlm(image=image, detections=low_conf_detections)

            for det, vlm_result in zip(low_conf_detections, vlm_results):
                if not vlm_result["is_correct"]:
                    # Detection is wrong, skip
                    continue

                corrected = vlm_result.get("corrected")
                if corrected is None:
                    # VLM returned null, detection is wrong
                    continue

                # Use corrected values or original
                final_class = corrected.get("class_name", det["class_name"])
                final_bbox = corrected.get("bbox_xyxy", det["bbox_xyxy"])
                final_conf = corrected.get("confidence", det["yolo_confidence"])

                predictions.append(
                    PredictedObject(
                        class_name=final_class,
                        confidence=final_conf,
                        bbox_xyxy=final_bbox,
                    )
                )

        # Stage 4: Apply NMS to merge overlapping detections
        predictions = self._nms(predictions, iou_threshold=0.5)

        return predictions


def build_tool(config: dict[str, Any]) -> YoloDetectVLMTool:
    """Build a YoloDetectVLMTool from configuration.

    Args:
        config: Dictionary with tool configuration.

    Returns:
        Configured YoloDetectVLMTool instance.
    """
    # The tool_config is a flat dict from load_annotation_config, not nested.
    # Keys like "weights", "yolo_conf", "device" are at the top level.
    # VLM config shares the "openai_vlm" namespace in the original TOML.

    # label_map may come from the tool section or the root
    label_map = config.get("label_map")

    return YoloDetectVLMTool(
        weights=str(config["weights"]),
        yolo_conf=float(config.get("yolo_conf", 0.01)),
        device=str(config.get("device", "cpu")),
        api_base=str(config.get("api_base", "http://localhost:11434/v1")),
        model=str(config.get("model", "llava")),
        api_key=config.get("api_key"),
        temperature=float(config.get("temperature", 0.0)),
        timeout=int(config.get("timeout", 120)),
        label_map=dict(label_map) if label_map else None,
    )


register_tool(
    AnnotationToolRegistration(
        name="yolo_detect_vlm_check",
        factory=build_tool,
        description="YOLO detection with VLM verification - uses low-threshold YOLO followed by VLM-based false positive filtering",
        use_polygon=False,
    )
)
