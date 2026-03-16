"""Annotation orchestration: fetch frames → run inference → upload results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from .annotation_tools.registry import get_tool_registration
from .cvat_utils import build_label_map, detections_to_shapes, upload_annotations
from .types import AnnotationTool

if TYPE_CHECKING:
    from cvat_sdk.core.proxies.tasks import Task


def build_model(
    tool_name: str,
    tool_config: dict,
    device: str,
) -> AnnotationTool:
    """Instantiate the requested annotation tool.

    Args:
        tool_name: Registered annotation tool name.
        tool_config: Tool-specific config loaded from annotation TOML.
        device: Inference device string (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns:
        An object with a ``predict(image, conf)`` method.
    """
    registration = get_tool_registration(tool_name)
    merged_config = dict(tool_config)
    merged_config.setdefault("device", device)
    return registration.factory(merged_config)


def resolve_task_frame_ids(
    task: "Task",
    frame_ids: list[int] | None = None,
) -> list[int]:
    """Resolve the actual frame ids that should be processed for a task."""
    if frame_ids is not None:
        return list(frame_ids)

    frames_info = task.get_frames_info()
    is_video_task = (
        len(frames_info) == 1
        and hasattr(frames_info[0], "name")
        and frames_info[0].name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    )

    if not is_video_task:
        return list(range(len(frames_info)))

    try:
        task_meta = task.get_meta()
        start_frame = task_meta.get("start_frame", 0)
        stop_frame = task_meta.get("stop_frame", 0)
        if isinstance(start_frame, int) and isinstance(stop_frame, int) and stop_frame >= start_frame:
            return list(range(start_frame, stop_frame + 1))
    except Exception:
        pass

    return [0]


def annotate_task(
    task: "Task",
    model: AnnotationTool,
    use_polygon: bool,
    conf: float,
    user_label_map: dict[str, str] | None,
    replace: bool,
    frame_ids: list[int] | None = None,
    progress_callback=None,
    frame_result_callback=None,
) -> int:
    """Annotate all frames of *task* using *model*.

    Fetches each frame, runs inference, converts predictions to CVAT shapes,
    and uploads the resulting annotations frame by frame.

    Args:
        task: CVAT task proxy object.
        model: Loaded model implementing ``predict(image, conf)``.
        use_polygon: Whether polygon outputs should be preferred when present.
        conf: Confidence threshold passed to the model.
        user_label_map: Optional mapping ``{model_class: cvat_label_name}``.
            When ``None``, model class names are matched to CVAT label names
            case-insensitively.
        replace: When ``True`` existing annotations are cleared once before
            uploading frame-by-frame results; when ``False`` new shapes are
            appended to existing ones.
        frame_ids: Specific frame indices to annotate. When ``None`` all
            frames in the task are annotated.
        progress_callback: Optional callable invoked after each frame with
            ``(frame_index, total_frames)`` as arguments.
        frame_result_callback: Optional callable invoked after each frame with
            ``(frame_id, detection_count, uploaded_count)``.

    Returns:
        Total number of shapes uploaded.
    """
    # Build label map from CVAT labels
    cvat_labels = task.get_labels()
    label_map = build_label_map(cvat_labels, user_label_map)
    frame_ids = resolve_task_frame_ids(task, frame_ids)

    uploaded_shapes = 0

    if replace:
        task.remove_annotations()

    total = len(frame_ids)
    for idx, frame_id in enumerate(frame_ids):
        frame_bytes = task.get_frame(frame_id, quality="original")
        try:
            pil_image = Image.open(frame_bytes).convert("RGB")
            try:
                detections = model.predict(pil_image, conf=conf)
            finally:
                pil_image.close()
        finally:
            frame_bytes.close()

        shapes = detections_to_shapes(detections, frame_id, label_map, use_polygon)
        uploaded_count = len(shapes)
        if shapes:
            upload_annotations(task, shapes, replace=False)
            uploaded_shapes += uploaded_count

        if frame_result_callback:
            frame_result_callback(frame_id, len(detections), uploaded_count)

        if progress_callback:
            progress_callback(idx + 1, total)

    return uploaded_shapes
