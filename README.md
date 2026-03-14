# cvat-ultralytics-bot

基于 ultralytics YOLO/SAM 的 CVAT 自动标注工具。

当前版本使用两类配置文件：

1. 连接配置文件：通过哈希和 base64 做轻量混淆，保存 CVAT 的 host、username、password。
2. 标注配置文件：TOML，顶层只保留 `tool`，其余参数放在对应的工具段里。

## 生成连接配置文件

```shell
cvat-bot create-connection-config \
    --output ./configs/cvat.connection.json \
    --host http://localhost:8080 \
    --username admin
```

命令执行后会交互式提示输入 CVAT 密码，不通过命令行参数传递。

## 导出内置标注预设

```shell
cvat-bot write-presets --output-dir ./configs
```

这会直接复制包内现有的 TOML 预设文件，例如：

- `yolo_detect.toml`
- `yolo_segment.toml`
- `yolo_sam.toml`

## 标注配置文件示例

```toml
tool = "yolo_detect"

[yolo_detect]
weights = "yolo26n.pt"
conf = 0.25
device = "cpu"
replace = false

[yolo_detect.label_map]
person = "person"
```

规则如下：

- 顶层的 `tool` 指定要使用的标注工具。
- 与 `tool` 同名的 TOML 段同时承载公共参数和工具专属参数。
- 例如 `tool = "yolo_detect"` 时，`conf`、`device`、`replace`、`frames`、`label_map` 以及工具自定义参数都放在 `[yolo_detect]` 下。
- 标签映射使用嵌套表，例如 `[yolo_detect.label_map]`。
- 工具通过 `cvat_ultralytics_bot/annotation_tools` 目录自动扫描注册。

## 执行标注

```shell
cvat-bot annotate 42 \
    --connection-config ./configs/cvat.connection.json \
    --annotation-config ./configs/yolo_detect.toml
```

也可以直接使用内置预设名：

```shell
cvat-bot annotate 42 \
    --connection-config ./configs/cvat.connection.json \
    --annotation-config yolo_detect
```

## 标注工具目录

内置工具位于：

- [cvat_ultralytics_bot/annotation_tools/yolo_detect.py](cvat_ultralytics_bot/annotation_tools/yolo_detect.py)
- [cvat_ultralytics_bot/annotation_tools/yolo_segment.py](cvat_ultralytics_bot/annotation_tools/yolo_segment.py)
- [cvat_ultralytics_bot/annotation_tools/yolo_sam.py](cvat_ultralytics_bot/annotation_tools/yolo_sam.py)

新增工具时，只需要在该目录中添加模块并完成注册，配置文件即可通过 `tool = "your_tool_name"` 使用它。