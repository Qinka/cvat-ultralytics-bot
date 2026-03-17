"""CVAT SDK helpers: client creation, label resolution, annotation upload."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cvat_sdk import make_client
from cvat_sdk.core.proxies.annotations import AnnotationUpdateAction
from cvat_sdk.models import (
    LabeledDataRequest,
    LabeledShapeRequest,
    PatchedLabeledDataRequest,
    ShapeType,
)

from .logging_config import get_logger

if TYPE_CHECKING:
    from cvat_sdk import Client
    from cvat_sdk.core.proxies.tasks import Task
    from cvat_sdk.models import IFrameMeta, ILabel

    from .types import PredictedObject


# Module-level logger
logger = get_logger(__name__)


def create_client(
    host: str,
    username: str,
    password: str,
) -> "Client":
    """Create and authenticate a CVAT SDK client.

    Args:
        host: Base URL of the CVAT instance (e.g. ``http://localhost:8080``).
        username: CVAT account username.
        password: CVAT account password.

    Returns:
        An authenticated :class:`cvat_sdk.Client`.

    Raises:
        Exception: If authentication fails.
    """
    logger.debug("Creating CVAT client for host: %s", host)
    client = make_client(host=host, credentials=(username, password))
    logger.info("Successfully connected to CVAT at %s", host)
    return client


def get_task(client: "Client", task_id: int) -> "Task":
    """Retrieve a task by its numeric ID.

    Args:
        client: An authenticated CVAT client.
        task_id: Numeric ID of the task.

    Returns:
        The :class:`~cvat_sdk.core.proxies.tasks.Task` proxy object.

    Raises:
        Exception: If the task cannot be retrieved.
    """
    logger.debug("Retrieving task with ID: %d", task_id)
    task = client.tasks.retrieve(task_id)
    logger.debug("Retrieved task: name=%s, id=%d", task.name, task.id)
    return task


def build_label_map(
    cvat_labels: "list[ILabel]",
    user_map: dict[str, str] | None,
) -> dict[str, int]:
    """Build a mapping from model class name to CVAT label ID.

    If *user_map* is provided (``{model_class: cvat_label_name}``), only the
    mapped classes are included. Otherwise every CVAT label is matched
    case-insensitively by name.

    Args:
        cvat_labels: Labels returned by :py:meth:`Task.get_labels`.
        user_map: Optional explicit mapping from model class name to CVAT
            label name.

    Returns:
        Dict mapping model class name → CVAT label ID.

    Raises:
        ValueError: If a label name in *user_map* does not exist in *cvat_labels*.
    """
    name_to_id: dict[str, int] = {lbl.name: lbl.id for lbl in cvat_labels}
    lower_name_to_id: dict[str, int] = {k.lower(): v for k, v in name_to_id.items()}

    if user_map:
        result: dict[str, int] = {}
        for model_cls, cvat_name in user_map.items():
            if cvat_name in name_to_id:
                result[model_cls] = name_to_id[cvat_name]
            elif cvat_name.lower() in lower_name_to_id:
                result[model_cls] = lower_name_to_id[cvat_name.lower()]
            else:
                raise ValueError(
                    f"CVAT label '{cvat_name}' not found in task labels: "
                    f"{list(name_to_id.keys())}"
                )
        return result

    # Auto-map: model class name -> CVAT label with same name (case-insensitive)
    return {k: v for k, v in lower_name_to_id.items()}


def detections_to_shapes(
    predictions: "list[PredictedObject]",
    frame_id: int,
    label_map: dict[str, int],
    use_polygon: bool,
) -> list[LabeledShapeRequest]:
    """Convert model predictions to CVAT :class:`LabeledShapeRequest` objects.

    Args:
        predictions: List of predictions for a single frame.
        frame_id: Zero-based CVAT frame index.
        label_map: Mapping from model class name (lower-case) to CVAT label ID.
        use_polygon: If ``True``, prefer polygon shapes when the prediction
            carries ``polygon_xy``; otherwise always produce rectangles.

    Returns:
        List of :class:`LabeledShapeRequest` ready for upload.
    """
    shapes: list[LabeledShapeRequest] = []
    for det in predictions:
        label_id = label_map.get(det.class_name) or label_map.get(det.class_name.lower())
        if label_id is None:
            continue  # unmapped class – skip silently

        if use_polygon and det.polygon_xy and len(det.polygon_xy) >= 6:
            shape = LabeledShapeRequest(
                type=ShapeType("polygon"),
                frame=frame_id,
                label_id=label_id,
                points=det.polygon_xy,
                occluded=False,
                source="semi-auto",
            )
        else:
            x1, y1, x2, y2 = det.bbox_xyxy
            shape = LabeledShapeRequest(
                type=ShapeType("rectangle"),
                frame=frame_id,
                label_id=label_id,
                points=[x1, y1, x2, y2],
                occluded=False,
                source="semi-auto",
            )
        shapes.append(shape)
    return shapes


def upload_annotations(
    task: "Task",
    shapes: list[LabeledShapeRequest],
    *,
    replace: bool = False,
) -> None:
    """Upload *shapes* as annotations to *task*.

    Args:
        task: The CVAT task proxy.
        shapes: Shapes to upload.
        replace: If ``True``, the existing annotations are fully replaced.
            If ``False`` (default), new shapes are appended.

    Note:
        This function logs at DEBUG level for each upload operation.
    """
    action = "replace" if replace else "create"
    logger.debug("Uploading %d shapes to task %d (action=%s)",
                 len(shapes), task.id, action)
    if replace:
        data = LabeledDataRequest(shapes=shapes)
        task.set_annotations(data)
    else:
        data = PatchedLabeledDataRequest(shapes=shapes)
        task.update_annotations(data, action=AnnotationUpdateAction.CREATE)
    logger.debug("Successfully uploaded annotations to task %d", task.id)
