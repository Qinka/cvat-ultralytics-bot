"""Command-line interface for cvat-ultralytics-bot."""

from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import typer
from tqdm import tqdm

from .logging_config import get_logger, setup_logging

logger = get_logger(__name__)

app = typer.Typer(
    name="cvat-bot",
    help="基于 ultralytics YOLO/SAM 对 CVAT 任务进行自动标注。",
    # add_completion=False,
)


# Global verbose flag
VERBOSE = False


def _handle_error(message: str, exc: Exception) -> None:
    """Print error message with optional traceback."""
    logger.error("%s: %s", message, exc)
    if VERBOSE:
        logger.exception(message)


def format_duration(seconds: float) -> str:
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


@app.command("create-connection-config")


def _builtin_preset_dir() -> Path:
    return Path(__file__).resolve().parent / "preset_annotation_configs"


def _resolve_annotation_config_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate

    preset_dir = _builtin_preset_dir()
    preset_candidates = [preset_dir / value]
    if candidate.suffix != ".toml":
        preset_candidates.append(preset_dir / f"{value}.toml")

    for preset_candidate in preset_candidates:
        if preset_candidate.exists():
            return preset_candidate

    raise ValueError(f"Annotation config not found: {value}")


@app.command("create-connection-config")
def create_connection_config(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="输出的加密连接配置文件路径。",
    ),
    host: str = typer.Option(
        ...,
        "--host",
        "-H",
        help="CVAT 服务器地址，例如 http://localhost:8080。",
    ),
    username: str = typer.Option(
        ...,
        "--username",
        "-u",
        help="CVAT 账号用户名。",
    ),
) -> None:
    """根据用户提供的信息生成连接配置文件。"""
    from .config import dump_connection_config

    password = typer.prompt("CVAT 账号密码", hide_input=True)
    path = dump_connection_config(output, host=host, username=username, password=password)
    typer.echo(f"[INFO] 已生成连接配置文件：{path}")


@app.command("write-presets")
def write_presets(
    output_dir: Path = typer.Option(
        Path("./annotation_presets"),
        "--output-dir",
        "-o",
        help="输出预设 TOML 配置的目录。",
    )
) -> None:
    """复制内置的标注配置预设文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for preset_path in sorted(_builtin_preset_dir().glob("*.toml")):
        target = output_dir / preset_path.name
        shutil.copy2(preset_path, target)
        written.append(target)
    typer.echo(f"[INFO] 已写出 {len(written)} 个预设配置到：{output_dir}")
    for path in written:
        typer.echo(f"[INFO]   {path}")


@app.command()
def annotate(
    task_ids: list[int] = typer.Argument(
        None,
        help="CVAT 任务 ID（数字），可指定多个。",
    ),
    connection_config: Path = typer.Option(
        ...,
        "--connection-config",
        "-C",
        help="加密的连接配置文件路径。",
    ),
    annotation_config: str = typer.Option(
        ...,
        "--annotation-config",
        "-A",
        help="标注配置 TOML 文件路径，或内置预设文件名。",
    ),
    project_id: int | None = typer.Option(
        None,
        "--project-id",
        "-p",
        help="CVAT Project ID，指定后自动获取其下所有任务进行标注。",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        "-r",
        help="在标注前清除旧标注。",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="显示详细的调试信息（包含调用栈）。",
    ),
) -> None:
    """对 CVAT 任务中的图像帧执行自动标注。

    程序会读取连接配置和标注配置，连接到 CVAT，逐帧获取图像并通过指定工具执行推理，
    然后将结果作为标注写回到任务中。
    支持一次指定多个任务 ID 进行批量处理，或通过 --project-id / -p 选项指定
    一个 Project，自动获取其下所有任务进行标注。
    """
    global VERBOSE
    VERBOSE = verbose

    # Setup logging based on verbose flag
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    logger.info("Starting cvat-bot annotate command")
    logger.debug("Verbose mode enabled")

    from .annotator import annotate_task, build_model, resolve_task_frame_ids
    from .annotation_tools import discover_tools, get_tool_registration
    from .config import load_annotation_config, load_connection_config
    from .cvat_utils import create_client, get_project_task_ids, get_task

    discover_tools()

    try:
        annotation_config_path = _resolve_annotation_config_path(annotation_config)
        conn = load_connection_config(connection_config)
        ann = load_annotation_config(annotation_config_path)
    except Exception as exc:  # noqa: BLE001
        _handle_error("配置加载失败", exc)
        raise typer.Exit(code=1)

    # ---------- Connect to CVAT ----------
    tqdm.write(f"[INFO] 连接 CVAT 服务器：{conn.host}")
    try:
        client = create_client(host=conn.host, username=conn.username, password=conn.password)
    except Exception as exc:  # noqa: BLE001
        _handle_error("连接 CVAT 失败", exc)
        raise typer.Exit(code=1)

    # ---------- Resolve task IDs from project or arguments ----------
    if project_id is not None:
        try:
            tqdm.write(f"[INFO] 获取 Project ID={project_id} 下的所有任务")
            resolved_task_ids = get_project_task_ids(client, project_id)
            tqdm.write(f"[INFO] Project ID={project_id} 共包含 {len(resolved_task_ids)} 个任务")
        except Exception as exc:  # noqa: BLE001
            _handle_error("获取 Project 任务列表失败", exc)
            raise typer.Exit(code=1)
    elif task_ids:
        resolved_task_ids = task_ids
    else:
        typer.secho("[ERROR] 必须指定 --project-id 或提供至少一个任务 ID", fg="red")
        raise typer.Exit(code=1)

    # ---------- Load model ----------
    tqdm.write(f"[INFO] 加载标注工具：tool={ann.tool}")
    try:
        registration = get_tool_registration(ann.tool)
        model = build_model(
            tool_name=ann.tool,
            tool_config=ann.tool_config,
            device=ann.device,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_error("标注工具加载失败", exc)
        raise typer.Exit(code=1)

    # ---------- Annotate each task ----------
    total_shapes = 0
    task_pbar = tqdm(
        resolved_task_ids,
        desc="任务进度",
        unit="任务",
        position=0,
        leave=True,
    )
    for task_idx, task_id in enumerate(task_pbar):
        task_pbar.set_postfix_str(f"当前任务: {task_id}")
        try:
            typer.echo(f"[INFO] 获取任务 ID={task_id}")
            task = get_task(client, task_id)
        except Exception as exc:  # noqa: BLE001
            _handle_error("获取任务失败", exc)
            raise typer.Exit(code=1)

        frame_ids = resolve_task_frame_ids(task, ann.frame_ids)
        n_frames = len(frame_ids)
        tqdm.write(
            f"[INFO] 开始标注任务 ID={task_id}，共 {n_frames} 帧"
            f"{'（全量）' if ann.frame_ids is None else '（指定帧）'}。"
        )

        frame_pbar = tqdm(
            total=n_frames,
            desc=f"帧进度",
            unit="帧",
            position=1,
            leave=False,
        )

        def _progress(done: int, total: int, elapsed: float, eta: float) -> None:
            frame_pbar.n = done
            frame_pbar.set_postfix({
                "已用时": format_duration(elapsed),
                "预计剩余": format_duration(eta),
            })
            frame_pbar.refresh()

        def _frame_result(frame_id: int, detection_count: int, uploaded_count: int) -> None:
            tqdm.write(
                f"[INFO]   frame_id={frame_id} detected={detection_count} uploaded={uploaded_count}"
            )
            frame_pbar.set_postfix({
                "frame_id": frame_id,
                "detected": detection_count,
                "uploaded": uploaded_count,
            })

        try:
            n_shapes = annotate_task(
                task=task,
                model=model,
                use_polygon=registration.use_polygon,
                conf=ann.conf,
                user_label_map=ann.label_map,
                replace=replace,
                frame_ids=frame_ids,
                progress_callback=_progress,
                frame_result_callback=_frame_result,
            )
        except Exception as exc:  # noqa: BLE001
            _handle_error("标注过程出错", exc)
            raise typer.Exit(code=1)

        frame_pbar.close()
        total_shapes += n_shapes
        typer.echo(
            f"[INFO] 任务 ID={task_id} 标注完成：共生成 {n_shapes} 个标注形状。"
        )

    typer.echo(
        f"[INFO] 全部标注完成：共处理 {len(resolved_task_ids)} 个任务，生成 {total_shapes} 个标注形状。"
    )


def main() -> None:
    """Entry point for the ``cvat-bot`` console script."""
    app()


if __name__ == "__main__":
    main()
