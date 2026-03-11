"""Tests for cvat_utils helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cvat_ultralytics_bot.cvat_utils import build_label_map, detections_to_shapes
from cvat_ultralytics_bot.models import Detection


def _make_label(name: str, label_id: int) -> MagicMock:
    lbl = MagicMock()
    lbl.name = name
    lbl.id = label_id
    return lbl


class TestBuildLabelMap:
    def test_auto_map_case_insensitive(self):
        labels = [_make_label("Person", 1), _make_label("Car", 2)]
        result = build_label_map(labels, user_map=None)
        assert result["person"] == 1
        assert result["car"] == 2

    def test_user_map_exact(self):
        labels = [_make_label("Person", 1), _make_label("Car", 2)]
        result = build_label_map(labels, user_map={"human": "Person", "vehicle": "Car"})
        assert result["human"] == 1
        assert result["vehicle"] == 2

    def test_user_map_case_insensitive_match(self):
        labels = [_make_label("Person", 10)]
        result = build_label_map(labels, user_map={"human": "person"})
        assert result["human"] == 10

    def test_user_map_missing_label_raises(self):
        labels = [_make_label("Person", 1)]
        with pytest.raises(ValueError, match="'Unknown'"):
            build_label_map(labels, user_map={"x": "Unknown"})


class TestDetectionsToShapes:
    def test_rectangle_from_bbox(self):
        det = Detection(
            class_name="person",
            confidence=0.9,
            bbox_xyxy=[10.0, 20.0, 100.0, 200.0],
        )
        shapes = detections_to_shapes([det], frame_id=0, label_map={"person": 1}, use_polygon=False)
        assert len(shapes) == 1
        s = shapes[0]
        assert str(s.type) == "rectangle"
        assert s.frame == 0
        assert s.label_id == 1
        assert s.points == [10.0, 20.0, 100.0, 200.0]

    def test_polygon_preferred_when_available(self):
        det = Detection(
            class_name="person",
            confidence=0.8,
            bbox_xyxy=[10.0, 20.0, 100.0, 200.0],
            polygon_xy=[10.0, 20.0, 100.0, 20.0, 100.0, 200.0],
        )
        shapes = detections_to_shapes([det], frame_id=1, label_map={"person": 5}, use_polygon=True)
        assert len(shapes) == 1
        s = shapes[0]
        assert str(s.type) == "polygon"
        assert s.points == [10.0, 20.0, 100.0, 20.0, 100.0, 200.0]

    def test_falls_back_to_rectangle_when_polygon_too_short(self):
        det = Detection(
            class_name="car",
            confidence=0.7,
            bbox_xyxy=[5.0, 5.0, 50.0, 50.0],
            polygon_xy=[5.0, 5.0],  # only 2 coords, not a valid polygon
        )
        shapes = detections_to_shapes([det], frame_id=0, label_map={"car": 2}, use_polygon=True)
        assert len(shapes) == 1
        assert str(shapes[0].type) == "rectangle"

    def test_unmapped_class_skipped(self):
        det = Detection(
            class_name="unknown_class",
            confidence=0.9,
            bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
        )
        shapes = detections_to_shapes([det], frame_id=0, label_map={"person": 1}, use_polygon=False)
        assert shapes == []

    def test_class_name_matched_case_insensitively(self):
        det = Detection(
            class_name="Person",
            confidence=0.9,
            bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
        )
        shapes = detections_to_shapes(
            [det], frame_id=0, label_map={"person": 1}, use_polygon=False
        )
        assert len(shapes) == 1

    def test_multiple_detections(self):
        dets = [
            Detection("person", 0.9, [0.0, 0.0, 10.0, 10.0]),
            Detection("car", 0.8, [20.0, 20.0, 60.0, 60.0]),
        ]
        shapes = detections_to_shapes(dets, frame_id=2, label_map={"person": 1, "car": 2}, use_polygon=False)
        assert len(shapes) == 2
        assert shapes[0].label_id == 1
        assert shapes[1].label_id == 2
