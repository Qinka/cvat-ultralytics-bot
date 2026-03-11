"""Annotation orchestration: download frames → run inference → upload results."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .cvat_utils import build_label_map, detections_to_shapes, upload_annotations
from .models import ModelType, YoloDetector, YoloSamSegmentor, YoloSegmentor

if TYPE_CHECKING:
    from cvat_sdk.core.proxies.tasks import Task


class ModelPredictor(Protocol):
    """Protocol for objects that can predict detections from a PIL Image."""

    def predict(self, image, conf: float) -> list:  # noqa: D102
        ...


def build_model(
    model_type: ModelType,
    model_weights: str,
    sam_weights: str | None,
    device: str,
) -> ModelPredictor:
    """Instantiate the requested model.

    Args:
        model_type: One of :class:`~cvat_ultralytics_bot.models.ModelType`.
        model_weights: Path or hub name for the primary (YOLO) weights.
        sam_weights: Path or hub name for SAM weights. Required when
            *model_type* is :attr:`ModelType.YOLO_SAM`.
        device: Inference device string (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns:
        An object with a ``predict(image, conf)`` method.

    Raises:
        ValueError: If *model_type* is ``YOLO_SAM`` but *sam_weights* is not
            provided.
    """
    if model_type == ModelType.YOLO_DETECT:
        return YoloDetector(model_weights, device=device)
    if model_type == ModelType.YOLO_SEGMENT:
        return YoloSegmentor(model_weights, device=device)
    if model_type == ModelType.YOLO_SAM:
        if not sam_weights:
            raise ValueError(
                "--sam-weights must be specified when --model-type is yolo-sam"
            )
        return YoloSamSegmentor(model_weights, sam_weights, device=device)
    raise ValueError(f"Unknown model type: {model_type}")  # pragma: no cover


def annotate_task(
    task: "Task",
    model: ModelPredictor,
    model_type: ModelType,
    conf: float,
    user_label_map: dict[str, str] | None,
    replace: bool,
    frame_ids: list[int] | None = None,
    progress_callback=None,
) -> int:
    """Annotate all frames of *task* using *model*.

    Downloads each frame, runs inference, converts predictions to CVAT shapes,
    and uploads the resulting annotations.

    Args:
        task: CVAT task proxy object.
        model: Loaded model implementing ``predict(image, conf)``.
        model_type: Used to decide whether to prefer polygon output.
        conf: Confidence threshold passed to the model.
        user_label_map: Optional mapping ``{model_class: cvat_label_name}``.
            When ``None``, model class names are matched to CVAT label names
            case-insensitively.
        replace: When ``True`` existing annotations are replaced; when
            ``False`` new shapes are appended to existing ones.
        frame_ids: Specific frame indices to annotate. When ``None`` all
            frames in the task are annotated.
        progress_callback: Optional callable invoked after each frame with
            ``(frame_index, total_frames)`` as arguments.

    Returns:
        Total number of shapes uploaded.
    """
    use_polygon = model_type in {ModelType.YOLO_SEGMENT, ModelType.YOLO_SAM}

    # Build label map from CVAT labels
    cvat_labels = task.get_labels()
    label_map = build_label_map(cvat_labels, user_label_map)

    frames_info = task.get_frames_info()
    if frame_ids is None:
        frame_ids = list(range(len(frames_info)))

    all_shapes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        total = len(frame_ids)
        for idx, frame_id in enumerate(frame_ids):
            # Download single frame as PIL Image
            frame_images = task.download_frames(
                [frame_id],
                outdir=tmpdir,
                quality="original",
            )

            if not frame_images:
                # Fallback: read downloaded file from disk
                import glob as _glob

                files = _glob.glob(str(Path(tmpdir) / "*.jpg")) + _glob.glob(
                    str(Path(tmpdir) / "*.png")
                )
                if not files:
                    if progress_callback:
                        progress_callback(idx + 1, total)
                    continue
                from PIL import Image

                pil_image = Image.open(files[-1]).convert("RGB")
            else:
                pil_image = frame_images[0]

            detections = model.predict(pil_image, conf=conf)
            shapes = detections_to_shapes(detections, frame_id, label_map, use_polygon)
            all_shapes.extend(shapes)

            if progress_callback:
                progress_callback(idx + 1, total)

    upload_annotations(task, all_shapes, replace=replace)
    return len(all_shapes)
