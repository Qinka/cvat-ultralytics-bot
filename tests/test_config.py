"""Tests for config loading and tool registry."""

from __future__ import annotations

from pathlib import Path

from cvat_ultralytics_bot.annotation_tools.registry import discover_tools, get_tool_registration
from cvat_ultralytics_bot.config import dump_connection_config, load_annotation_config, load_connection_config


def test_connection_config_roundtrip(tmp_path: Path):
    path = tmp_path / "connection.json"
    dump_connection_config(
        path,
        host="http://localhost:8080",
        username="admin",
        password="secret",
    )

    config = load_connection_config(path)
    assert config.host == "http://localhost:8080"
    assert config.username == "admin"
    assert config.password == "secret"


def test_annotation_config_load(tmp_path: Path):
    path = tmp_path / "annotation.toml"
    path.write_text(
        "tool = \"yolo_detect\"\n\n[yolo_detect]\nweights = \"demo.pt\"\nconf = 0.5\ndevice = \"cuda:0\"\nreplace = true\nframes = [1, 2, 3]\n\n[yolo_detect.label_map]\nperson = \"person\"\n",
        encoding="utf-8",
    )

    config = load_annotation_config(path)
    assert config.tool == "yolo_detect"
    assert config.conf == 0.5
    assert config.device == "cuda:0"
    assert config.replace is True
    assert config.frame_ids == [1, 2, 3]
    assert config.label_map == {"person": "person"}
    assert config.tool_config == {"weights": "demo.pt"}


def test_annotation_config_load_uses_top_level_fallback(tmp_path: Path):
    path = tmp_path / "annotation.toml"
    path.write_text(
        "tool = \"yolo_detect\"\nconf = 0.4\ndevice = \"cpu\"\nreplace = true\nframes = [7]\n\n[label_map]\ncar = \"vehicle\"\n\n[yolo_detect]\nweights = \"demo.pt\"\n",
        encoding="utf-8",
    )

    config = load_annotation_config(path)
    assert config.conf == 0.4
    assert config.device == "cpu"
    assert config.replace is True
    assert config.frame_ids == [7]
    assert config.label_map == {"car": "vehicle"}
    assert config.tool_config == {"weights": "demo.pt"}


def test_registry_discovers_yolo_tools():
    registry = discover_tools(force=True)
    assert "yolo_detect" in registry
    assert "yolo_segment" in registry
    assert "yolo_sam" in registry
    assert get_tool_registration("yolo_detect").name == "yolo_detect"
