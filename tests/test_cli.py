"""Tests for the CLI layer."""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cvat_ultralytics_bot.cli import app
from cvat_ultralytics_bot.config import dump_connection_config

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
        assert "--connection-config" in plain
        assert "--annotation-config" in plain
        assert "--secret" not in plain

    def test_create_connection_config_help(self):
        result = runner.invoke(app, ["create-connection-config", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "--host" in plain
        assert "--username" in plain
        assert "--password" not in plain


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

    def _write_annotation_config(self, tmp_path):
        config_path = tmp_path / "annotation.toml"
        config_path.write_text(
            "tool = \"yolo_detect\"\nconf = 0.25\ndevice = \"cpu\"\nreplace = false\n\n[yolo_detect]\nweights = \"yolo26n.pt\"\n",
            encoding="utf-8",
        )
        return config_path

    def _write_connection_config(self, tmp_path):
        connection_path = tmp_path / "connection.json"
        dump_connection_config(
            connection_path,
            host="http://localhost:8080",
            username="admin",
            password="secret",
        )
        return connection_path

    def test_annotate_success(self, tmp_path):
        task_mock = self._make_task()
        patches = self._patch_all(task_mock)
        annotation_config = self._write_annotation_config(tmp_path)
        connection_config = self._write_connection_config(tmp_path)
        with patches[0], patches[1], patches[2], patches[3]:
            result = runner.invoke(
                app,
                [
                    "annotate",
                    "42",
                    "--connection-config", str(connection_config),
                    "--annotation-config", str(annotation_config),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "42" in result.output

    def test_annotate_passes_frame_result_callback(self, tmp_path):
        task_mock = self._make_task()
        patches = self._patch_all(task_mock)
        annotate_mock = patches[3]
        annotation_config = self._write_annotation_config(tmp_path)
        connection_config = self._write_connection_config(tmp_path)
        with patches[0], patches[1], patches[2], annotate_mock as mocked_annotate:
            result = runner.invoke(
                app,
                [
                    "annotate",
                    "42",
                    "--connection-config", str(connection_config),
                    "--annotation-config", str(annotation_config),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "frame_result_callback" in mocked_annotate.call_args.kwargs
        assert callable(mocked_annotate.call_args.kwargs["frame_result_callback"])

    def test_annotate_accepts_builtin_preset_name_without_suffix(self, tmp_path):
        task_mock = self._make_task()
        patches = self._patch_all(task_mock)
        connection_config = self._write_connection_config(tmp_path)
        with patches[0], patches[1], patches[2], patches[3]:
            result = runner.invoke(
                app,
                [
                    "annotate",
                    "42",
                    "--connection-config", str(connection_config),
                    "--annotation-config", "yolo_detect",
                ],
            )
        assert result.exit_code == 0, result.output

    def test_annotate_missing_required_arg(self):
        result = runner.invoke(
            app,
            ["annotate"],
        )
        assert result.exit_code != 0

    def test_create_connection_config(self, tmp_path):
        output = tmp_path / "conn.json"
        result = runner.invoke(
            app,
            [
                "create-connection-config",
                "--output", str(output),
                "--host", "http://localhost:8080",
                "--username", "admin",
            ],
            input="secret\n",
        )
        assert result.exit_code == 0, result.output
        assert output.exists()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["version"] == 2
        assert payload["encryption"] == "sha256-xor-base64"

    def test_write_presets(self, tmp_path):
        result = runner.invoke(
            app,
            ["write-presets", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "yolo_detect.toml").exists()
        assert (tmp_path / "yolo_detect.toml").read_text(encoding="utf-8").startswith('tool = "yolo_detect"')
