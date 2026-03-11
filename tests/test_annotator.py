"""Tests for annotator orchestration logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cvat_ultralytics_bot.annotator import annotate_task, build_model
from cvat_ultralytics_bot.models import Detection, ModelType


class TestBuildModel:
    def test_yolo_detect(self):
        with patch("ultralytics.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock(names={0: "person"})
            model = build_model(ModelType.YOLO_DETECT, "yolov8n.pt", None, "cpu")
            assert hasattr(model, "predict")

    def test_yolo_segment(self):
        with patch("ultralytics.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock(names={0: "person"})
            model = build_model(ModelType.YOLO_SEGMENT, "yolov8n-seg.pt", None, "cpu")
            assert hasattr(model, "predict")

    def test_yolo_sam_requires_sam_weights(self):
        with pytest.raises(ValueError, match="--sam-weights"):
            build_model(ModelType.YOLO_SAM, "yolov8n.pt", None, "cpu")

    def test_yolo_sam_creates_segmentor(self):
        with (
            patch("ultralytics.YOLO") as mock_yolo,
            patch("ultralytics.SAM") as mock_sam,
        ):
            mock_yolo.return_value = MagicMock(names={0: "person"})
            mock_sam.return_value = MagicMock()
            model = build_model(ModelType.YOLO_SAM, "yolov8n.pt", "sam2.pt", "cpu")
            assert hasattr(model, "predict")


class TestAnnotateTask:
    """Integration-style tests for annotate_task using mocked CVAT + model."""

    def _make_task(self, labels, frames_info):
        task = MagicMock()
        task.get_labels.return_value = labels
        task.get_frames_info.return_value = frames_info

        def fake_download(frame_ids, outdir, quality="original"):
            from PIL import Image

            imgs = []
            for _ in frame_ids:
                img = Image.new("RGB", (640, 480), color=(0, 0, 0))
                imgs.append(img)
            return imgs

        task.download_frames.side_effect = fake_download
        return task

    def _make_label(self, name, label_id):
        lbl = MagicMock()
        lbl.name = name
        lbl.id = label_id
        return lbl

    def _make_frame(self):
        f = MagicMock()
        f.width = 640
        f.height = 480
        return f

    def test_annotate_detections(self):
        labels = [self._make_label("person", 1), self._make_label("car", 2)]
        frames_info = [self._make_frame(), self._make_frame()]
        task = self._make_task(labels, frames_info)

        class FakeModel:
            def predict(self, image, conf):
                return [
                    Detection("person", 0.9, [10.0, 20.0, 100.0, 200.0]),
                    Detection("car", 0.8, [30.0, 30.0, 200.0, 200.0]),
                ]

        progress_calls = []
        n = annotate_task(
            task=task,
            model=FakeModel(),
            model_type=ModelType.YOLO_DETECT,
            conf=0.25,
            user_label_map=None,
            replace=False,
            frame_ids=[0, 1],
            progress_callback=lambda done, total: progress_calls.append((done, total)),
        )

        # 2 frames × 2 detections = 4 shapes
        assert n == 4
        assert progress_calls == [(1, 2), (2, 2)]
        task.update_annotations.assert_called_once()

    def test_annotate_replace_calls_set_annotations(self):
        labels = [self._make_label("person", 1)]
        frames_info = [self._make_frame()]
        task = self._make_task(labels, frames_info)

        class FakeModel:
            def predict(self, image, conf):
                return [Detection("person", 0.9, [0.0, 0.0, 10.0, 10.0])]

        annotate_task(
            task=task,
            model=FakeModel(),
            model_type=ModelType.YOLO_DETECT,
            conf=0.25,
            user_label_map=None,
            replace=True,
        )
        task.set_annotations.assert_called_once()
        task.update_annotations.assert_not_called()

    def test_annotate_empty_detections(self):
        labels = [self._make_label("person", 1)]
        frames_info = [self._make_frame()]
        task = self._make_task(labels, frames_info)

        class FakeModel:
            def predict(self, image, conf):
                return []

        n = annotate_task(
            task=task,
            model=FakeModel(),
            model_type=ModelType.YOLO_DETECT,
            conf=0.25,
            user_label_map=None,
            replace=False,
        )
        assert n == 0

    def test_annotate_polygon_model(self):
        labels = [self._make_label("person", 1)]
        frames_info = [self._make_frame()]
        task = self._make_task(labels, frames_info)

        class FakeModel:
            def predict(self, image, conf):
                return [
                    Detection(
                        "person",
                        0.9,
                        [0.0, 0.0, 100.0, 100.0],
                        polygon_xy=[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0],
                    )
                ]

        annotate_task(
            task=task,
            model=FakeModel(),
            model_type=ModelType.YOLO_SEGMENT,
            conf=0.25,
            user_label_map=None,
            replace=False,
        )
        # Verify the shape type uploaded is a polygon
        call_args = task.update_annotations.call_args
        labeled_data = call_args[0][0]
        assert str(labeled_data.shapes[0].type) == "polygon"
