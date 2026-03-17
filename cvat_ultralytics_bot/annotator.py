"""Annotation orchestration: fetch frames → run inference → upload results."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PIL import Image

from .annotation_tools.registry import get_tool_registration
from .cvat_utils import build_label_map, detections_to_shapes, upload_annotations
from .logging_config import get_logger
from .types import AnnotationTool

if TYPE_CHECKING:
    from cvat_sdk.core.proxies.tasks import Task


# Module-level logger
logger = get_logger(__name__)


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

    Raises:
        ValueError: If the tool name is not registered.
    """
    logger.info("Building annotation tool: name=%s, device=%s", tool_name, device)
    registration = get_tool_registration(tool_name)
    merged_config = dict(tool_config)
    merged_config.setdefault("device", device)
    model = registration.factory(merged_config)
    logger.debug("Model config: %s", merged_config)
    return model


def resolve_task_frame_ids(
    task: "Task",
    frame_ids: list[int] | None = None,
) -> list[int]:
    """Resolve the actual frame ids that should be processed for a task.

    Args:
        task: CVAT task proxy object.
        frame_ids: User-specified frame IDs. If None, auto-detect.

    Returns:
        List of frame IDs to process.

    Note:
        For video tasks, this tries to read start_frame/stop_frame from
        task metadata. For image tasks, it returns all frame indices.
    """
    if frame_ids is not None:
        logger.debug("Using user-specified frame IDs: %s", frame_ids)
        return list(frame_ids)

    frames_info = task.get_frames_info()
    is_video_task = (
        len(frames_info) == 1
        and hasattr(frames_info[0], "name")
        and frames_info[0].name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    )

    if not is_video_task:
        frame_count = len(frames_info)
        logger.debug("Detected image task with %d frames", frame_count)
        return list(range(len(frames_info)))

    try:
        task_meta = task.get_meta()
        start_frame = task_meta.get("start_frame", 0)
        stop_frame = task_meta.get("stop_frame", 0)
        if isinstance(start_frame, int) and isinstance(stop_frame, int) and stop_frame >= start_frame:
            frame_range = list(range(start_frame, stop_frame + 1))
            logger.debug("Detected video task: frames %d to %d", start_frame, stop_frame)
            return frame_range
    except Exception as e:
        logger.warning("Failed to read video task metadata: %s", e)

    logger.debug("Using default frame ID: 0")
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
            ``(frame_index, total_frames, elapsed_seconds, eta_seconds)`` as arguments.
        frame_result_callback: Optional callable invoked after each frame with
            ``(frame_id, detection_count, uploaded_count)``.

    Returns:
        Total number of shapes uploaded.

    Note:
        This function logs at INFO level for each frame processed and at
        DEBUG level for detailed timing information.
    """
    # Build label map from CVAT labels
    logger.info("Starting annotation for task ID: %s", task.id)
    logger.debug("Annotation parameters: conf=%.2f, use_polygon=%s, replace=%s",
                 conf, use_polygon, replace)

    cvat_labels = task.get_labels()
    label_map = build_label_map(cvat_labels, user_label_map)
    logger.debug("Built label map: %s", label_map)

    frame_ids = resolve_task_frame_ids(task, frame_ids)
    total_frames = len(frame_ids)
    logger.info("Processing %d frames", total_frames)

    uploaded_shapes = 0

    if replace:
        logger.info("Clearing existing annotations before upload")
        task.remove_annotations()

    total = len(frame_ids)
    total_start_time = time.perf_counter()
    for idx, frame_id in enumerate(frame_ids):
        logger.debug("Processing frame %d (%d/%d)", frame_id, idx + 1, total)
        frame_start_time = time.perf_counter()

        frame_bytes = task.get_frame(frame_id, quality="original")
        try:
            pil_image = Image.open(frame_bytes).convert("RGB")
            try:
                infer_start_time = time.perf_counter()
                detections = model.predict(pil_image, conf=conf)
                infer_elapsed = time.perf_counter() - infer_start_time
                logger.debug("Inference completed: %d detections in %.3fs",
                             len(detections), infer_elapsed)
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

        # Calculate timing statistics
        elapsed_total = time.perf_counter() - total_start_time
        avg_time_per_frame = elapsed_total / (idx + 1)
        remaining_frames = total - (idx + 1)
        eta_seconds = avg_time_per_frame * remaining_frames

        if progress_callback:
            progress_callback(idx + 1, total, elapsed_total, eta_seconds)

        # Log progress with timing info every 10 frames or at completion
        if (idx + 1) % 10 == 0 or idx + 1 == total:
            logger.info(
                "Progress: %d/%d frames (%.1f%%) | elapsed: %s | ETA: %s",
                idx + 1, total,
                (idx + 1) * 100 / total,
                _format_duration(elapsed_total),
                _format_duration(eta_seconds)
            )

        frame_elapsed = time.perf_counter() - frame_start_time
        logger.debug("Frame %d processed in %.3fs: %d detections, %d shapes uploaded",
                      frame_id, frame_elapsed, len(detections), uploaded_count)

    total_elapsed = time.perf_counter() - total_start_time
    logger.info("Annotation completed: %d shapes uploaded to task %s in %s",
                uploaded_shapes, task.id, _format_duration(total_elapsed))
    return uploaded_shapes


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1h 23m 45s", "5m 30s", or "45s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
