"""Tests for annotator orchestration logic."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from cvat_sdk.models import PatchedLabeledDataRequest
from PIL import Image

from cvat_ultralytics_bot.annotator import annotate_task, build_model
from cvat_ultralytics_bot.types import Detection


class TestBuildModel:
    def test_yolo_detect(self):
        with patch("ultralytics.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock(names={0: "person"})
            model = build_model("yolo_detect", {"weights": "yolo26n.pt"}, "cpu")
            assert hasattr(model, "predict")

    def test_yolo_segment(self):
        with patch("ultralytics.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock(names={0: "person"})
            model = build_model("yolo_segment", {"weights": "yolo26n-seg.pt"}, "cpu")
            assert hasattr(model, "predict")

    def test_yolo_sam_creates_segmentor(self):
        with (
            patch("ultralytics.YOLO") as mock_yolo,
            patch("ultralytics.SAM") as mock_sam,
        ):
            mock_yolo.return_value = MagicMock(names={0: "person"})
            mock_sam.return_value = MagicMock()
            model = build_model(
                "yolo_sam",
                {"yolo_weights": "yolo26n.pt", "sam_weights": "sam2.pt"},
                "cpu",
            )
            assert hasattr(model, "predict")


class TestAnnotateTask:
    """Integration-style tests for annotate_task using mocked CVAT + model."""

    def _make_task(self, labels, frames_info):
        task = MagicMock()
        task.get_labels.return_value = labels
        task.get_frames_info.return_value = frames_info
        return task

    def _set_frame_images(self, task, colors):
        def fake_get_frame(frame_id, quality="original"):
            image = Image.new("RGB", (640, 480), color=colors[frame_id])
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer

        task.get_frame.side_effect = fake_get_frame

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
        self._set_frame_images(task, [(0, 0, 0), (1, 1, 1)])

        class FakeModel:
            def predict(self, image, conf):
                return [
                    Detection("person", 0.9, [10.0, 20.0, 100.0, 200.0]),
                    Detection("car", 0.8, [30.0, 30.0, 200.0, 200.0]),
                ]

        progress_calls = []
        frame_result_calls = []
        n = annotate_task(
            task=task,
            model=FakeModel(),
            use_polygon=False,
            conf=0.25,
            user_label_map=None,
            replace=False,
            frame_ids=[0, 1],
            progress_callback=lambda done, total: progress_calls.append((done, total)),
            frame_result_callback=lambda frame_id, dets, uploaded: frame_result_calls.append(
                (frame_id, dets, uploaded)
            ),
        )

        # 2 frames × 2 detections = 4 shapes
        assert n == 4
        assert progress_calls == [(1, 2), (2, 2)]
        assert frame_result_calls == [(0, 2, 2), (1, 2, 2)]
        assert task.update_annotations.call_count == 2
        task.remove_annotations.assert_not_called()

    def test_annotate_replace_calls_set_annotations(self):
        labels = [self._make_label("person", 1)]
        frames_info = [self._make_frame()]
        task = self._make_task(labels, frames_info)
        self._set_frame_images(task, [(0, 0, 0)])

        class FakeModel:
            def predict(self, image, conf):
                return [Detection("person", 0.9, [0.0, 0.0, 10.0, 10.0])]

        annotate_task(
            task=task,
            model=FakeModel(),
            use_polygon=False,
            conf=0.25,
            user_label_map=None,
            replace=True,
        )
        task.remove_annotations.assert_called_once_with()
        task.update_annotations.assert_called_once()
        task.set_annotations.assert_not_called()

    def test_annotate_empty_detections(self):
        labels = [self._make_label("person", 1)]
        frames_info = [self._make_frame()]
        task = self._make_task(labels, frames_info)
        self._set_frame_images(task, [(0, 0, 0)])

        class FakeModel:
            def predict(self, image, conf):
                return []

        n = annotate_task(
            task=task,
            model=FakeModel(),
            use_polygon=False,
            conf=0.25,
            user_label_map=None,
            replace=False,
        )
        assert n == 0
        task.update_annotations.assert_not_called()

    def test_annotate_polygon_model(self):
        labels = [self._make_label("person", 1)]
        frames_info = [self._make_frame()]
        task = self._make_task(labels, frames_info)
        self._set_frame_images(task, [(0, 0, 0)])

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
            use_polygon=True,
            conf=0.25,
            user_label_map=None,
            replace=False,
        )
        # Verify the shape type uploaded is a polygon
        call_args = task.update_annotations.call_args
        labeled_data = call_args[0][0]
        assert isinstance(labeled_data, PatchedLabeledDataRequest)
        assert str(labeled_data.shapes[0].type) == "polygon"

    def test_annotate_replace_clears_even_when_no_detections(self):
        labels = [self._make_label("person", 1)]
        frames_info = [self._make_frame()]
        task = self._make_task(labels, frames_info)
        self._set_frame_images(task, [(0, 0, 0)])

        class FakeModel:
            def predict(self, image, conf):
                return []

        n = annotate_task(
            task=task,
            model=FakeModel(),
            use_polygon=False,
            conf=0.25,
            user_label_map=None,
            replace=True,
        )

        assert n == 0
        task.remove_annotations.assert_called_once_with()
        task.update_annotations.assert_not_called()

    def test_annotate_reads_distinct_frames(self):
        labels = [self._make_label("person", 1)]
        frames_info = [self._make_frame(), self._make_frame()]
        task = self._make_task(labels, frames_info)
        self._set_frame_images(task, [(255, 0, 0), (0, 255, 0)])

        seen_pixels = []

        class FakeModel:
            def predict(self, image, conf):
                seen_pixels.append(image.getpixel((0, 0)))
                return [Detection("person", 0.9, [0.0, 0.0, 10.0, 10.0])]

        annotate_task(
            task=task,
            model=FakeModel(),
            use_polygon=False,
            conf=0.25,
            user_label_map=None,
            replace=False,
            frame_ids=[0, 1],
        )

        assert seen_pixels == [(255, 0, 0), (0, 255, 0)]
