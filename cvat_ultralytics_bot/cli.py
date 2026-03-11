"""Command-line interface for cvat-ultralytics-bot."""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import typer

from .models import ModelType

app = typer.Typer(
    name="cvat-bot",
    help="基于 ultralytics YOLO/SAM 对 CVAT 任务进行自动标注。",
    add_completion=False,
)


def _parse_label_map(value: str | None) -> dict[str, str] | None:
    """Parse *value* as a JSON label map.

    Accepts a raw JSON string or the path to a JSON file.

    Args:
        value: JSON string, file path, or ``None``.

    Returns:
        Parsed dict or ``None`` if *value* is ``None``.

    Raises:
        typer.BadParameter: If the string cannot be parsed.
    """
    if value is None:
        return None
    # Try as a file path first
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as fh:
            return json.load(fh)
    # Try as a raw JSON string
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"--label-map must be a valid JSON string or path to a JSON file. Error: {exc}"
        ) from exc


@app.command()
def annotate(
    task_id: int = typer.Argument(
        ...,
        help="CVAT 任务 ID（数字）。",
    ),
    host: str = typer.Option(
        ...,
        "--host",
        "-H",
        help="CVAT 服务器地址，例如 http://localhost:8080。",
        envvar="CVAT_HOST",
    ),
    username: str = typer.Option(
        ...,
        "--username",
        "-u",
        help="CVAT 账号用户名。",
        envvar="CVAT_USERNAME",
    ),
    password: str = typer.Option(
        ...,
        "--password",
        "-p",
        help="CVAT 账号密码。",
        envvar="CVAT_PASSWORD",
        hide_input=True,
    ),
    model_type: ModelType = typer.Option(
        ModelType.YOLO_DETECT,
        "--model-type",
        "-t",
        help=(
            "模型类型：\n"
            "  yolo-detect  — YOLO 目标检测（输出矩形框）\n"
            "  yolo-segment — YOLO 实例分割（输出多边形）\n"
            "  yolo-sam     — YOLO 检测 + SAM 精细分割（输出多边形）"
        ),
        case_sensitive=False,
    ),
    model_weights: str = typer.Option(
        "yolov8n.pt",
        "--model-weights",
        "-w",
        help="YOLO 模型权重文件路径或 ultralytics hub 模型名（如 yolov8n.pt）。",
    ),
    sam_weights: Optional[str] = typer.Option(
        None,
        "--sam-weights",
        help="SAM 模型权重文件路径（仅 --model-type yolo-sam 时需要）。",
    ),
    conf: float = typer.Option(
        0.25,
        "--conf",
        "-c",
        min=0.0,
        max=1.0,
        help="置信度阈值，范围 [0, 1]。",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="推理设备，例如 cpu、cuda:0。",
    ),
    label_map: Optional[str] = typer.Option(
        None,
        "--label-map",
        "-l",
        help=(
            "模型类别名称到 CVAT 标签名称的映射，JSON 字符串或 JSON 文件路径。\n"
            '例如 \'{"person": "行人", "car": "车辆"}\'\n'
            "若不指定，则按名称（大小写不敏感）自动匹配。"
        ),
    ),
    frame_ids: Optional[str] = typer.Option(
        None,
        "--frames",
        "-f",
        help=(
            "要标注的帧编号列表（逗号分隔，0 起始），例如 0,1,2,10。\n"
            "若不指定，则标注任务中的所有帧。"
        ),
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        "-r",
        help="若指定，则清空任务现有标注后再写入；否则追加到现有标注。",
    ),
) -> None:
    """对 CVAT 任务 TASK_ID 中的图像帧执行自动标注。

    程序会连接到指定的 CVAT 服务器，逐帧下载图像并使用所选模型进行推理，
    然后将检测/分割结果作为标注写回到 CVAT 任务中。
    """
    from .annotator import annotate_task, build_model
    from .cvat_utils import create_client, get_task

    # ---------- Parse optional parameters ----------
    parsed_label_map = _parse_label_map(label_map)

    parsed_frame_ids: list[int] | None = None
    if frame_ids is not None:
        try:
            parsed_frame_ids = [int(x.strip()) for x in frame_ids.split(",") if x.strip()]
        except ValueError:
            typer.echo(
                f"[ERROR] --frames 参数格式无效：'{frame_ids}'，应为逗号分隔的整数列表。",
                err=True,
            )
            raise typer.Exit(code=1)

    # ---------- Load model ----------
    typer.echo(f"[INFO] 加载模型：type={model_type.value}  weights={model_weights}")
    try:
        model = build_model(
            model_type=model_type,
            model_weights=model_weights,
            sam_weights=sam_weights,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[ERROR] 模型加载失败：{exc}", err=True)
        raise typer.Exit(code=1)

    # ---------- Connect to CVAT ----------
    typer.echo(f"[INFO] 连接 CVAT 服务器：{host}")
    try:
        client = create_client(host=host, username=username, password=password)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[ERROR] 连接 CVAT 失败：{exc}", err=True)
        raise typer.Exit(code=1)

    try:
        typer.echo(f"[INFO] 获取任务 ID={task_id}")
        task = get_task(client, task_id)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[ERROR] 获取任务失败：{exc}", err=True)
        raise typer.Exit(code=1)

    # ---------- Annotate ----------
    frames_info = task.get_frames_info()
    total_frames = (
        len(parsed_frame_ids) if parsed_frame_ids is not None else len(frames_info)
    )
    typer.echo(
        f"[INFO] 开始标注任务 ID={task_id}，共 {total_frames} 帧"
        f"{'（全量）' if parsed_frame_ids is None else '（指定帧）'}。"
    )

    def _progress(done: int, total: int) -> None:
        pct = done * 100 // total
        typer.echo(f"[INFO]   帧 {done}/{total}  ({pct}%)")

    try:
        n_shapes = annotate_task(
            task=task,
            model=model,
            model_type=model_type,
            conf=conf,
            user_label_map=parsed_label_map,
            replace=replace,
            frame_ids=parsed_frame_ids,
            progress_callback=_progress,
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[ERROR] 标注过程出错：{exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"[INFO] 标注完成：共生成 {n_shapes} 个标注形状，已写入任务 ID={task_id}。"
    )


def main() -> None:
    """Entry point for the ``cvat-bot`` console script."""
    app()


if __name__ == "__main__":
    main()
