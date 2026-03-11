"""Tests for the CLI layer."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cvat_ultralytics_bot.cli import app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from *text*."""
    return re.sub(r"\x1b\[[0-9;]*[mK]", "", text)


class TestCliHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "CVAT" in plain or "annotate" in plain

    def test_annotate_help(self):
        result = runner.invoke(app, ["annotate", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "--host" in plain
        assert "--model-type" in plain
        assert "--conf" in plain


class TestParseLabelMap:
    def test_parse_json_string(self):
        from cvat_ultralytics_bot.cli import _parse_label_map

        result = _parse_label_map('{"person": "行人"}')
        assert result == {"person": "行人"}

    def test_parse_none(self):
        from cvat_ultralytics_bot.cli import _parse_label_map

        assert _parse_label_map(None) is None

    def test_parse_invalid_raises(self):
        import typer

        from cvat_ultralytics_bot.cli import _parse_label_map

        with pytest.raises(typer.BadParameter):
            _parse_label_map("not_json")

    def test_parse_file(self, tmp_path):
        import json

        from cvat_ultralytics_bot.cli import _parse_label_map

        f = tmp_path / "map.json"
        f.write_text(json.dumps({"cat": "猫"}))
        result = _parse_label_map(str(f))
        assert result == {"cat": "猫"}


class TestAnnotateCommand:
    """Integration test: invoke the annotate command with mocked internals."""

    def _patch_all(self, task_mock):
        return [
            patch("cvat_ultralytics_bot.cvat_utils.create_client", return_value=MagicMock()),
            patch("cvat_ultralytics_bot.cvat_utils.get_task", return_value=task_mock),
            patch("cvat_ultralytics_bot.annotator.build_model", return_value=MagicMock()),
            patch("cvat_ultralytics_bot.annotator.annotate_task", return_value=3),
        ]

    def _make_task(self):
        task = MagicMock()
        frame = MagicMock()
        frame.width = 640
        frame.height = 480
        task.get_frames_info.return_value = [frame]
        return task

    def test_annotate_success(self):
        task_mock = self._make_task()
        patches = self._patch_all(task_mock)
        with patches[0], patches[1], patches[2], patches[3]:
            result = runner.invoke(
                app,
                [
                    "42",
                    "--host", "http://localhost:8080",
                    "--username", "admin",
                    "--password", "secret",
                    "--model-type", "yolo-detect",
                    "--model-weights", "yolov8n.pt",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "42" in result.output

    def test_annotate_missing_required_arg(self):
        # Missing TASK_ID (positional) → typer should error
        result = runner.invoke(
            app,
            [
                "--host", "http://localhost:8080",
                "--username", "admin",
                "--password", "secret",
            ],
        )
        assert result.exit_code != 0

    def test_annotate_invalid_frames_arg(self):
        task_mock = self._make_task()
        patches = self._patch_all(task_mock)
        with patches[0], patches[1], patches[2], patches[3]:
            result = runner.invoke(
                app,
                [
                    "1",
                    "--host", "http://localhost:8080",
                    "--username", "admin",
                    "--password", "secret",
                    "--frames", "a,b,c",
                ],
            )
        assert result.exit_code == 1
