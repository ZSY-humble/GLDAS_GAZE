"""使用 GLDAS-Gaze 预训练权重对单张图片生成搜索轨迹。"""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web_demo.engine import InferenceEngine, MAX_STEPS, TASKS


def parse_args():
    parser = argparse.ArgumentParser(description="GLDAS-Gaze 单图推理")
    parser.add_argument("--image", type=Path, required=True, help="输入图片路径")
    parser.add_argument("--task", choices=TASKS, required=True,
                        help="搜索目标，含空格的类别需加引号")
    parser.add_argument("--condition", choices=MAX_STEPS, default="TP",
                        help="预训练权重的训练条件")
    parser.add_argument("--device", default="auto",
                        help="auto、cpu、cuda 或 cuda:设备编号")
    parser.add_argument("--decoding", choices=("greedy", "sample"),
                        default="greedy")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def save_overlay(image, result, path):
    """按原图尺寸绘制搜索轨迹；模型坐标来自 512×320 输入。"""
    overlay = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    x_scale = overlay.width / result["width"]
    y_scale = overlay.height / result["height"]
    points = [
        (round(fix["x"] * x_scale), round(fix["y"] * y_scale))
        for fix in result["fixations"]
    ]
    radius = max(5, round(min(overlay.size) * 0.012))
    line_width = max(2, radius // 3)
    if len(points) > 1:
        draw.line(points, fill="#ffffff", width=line_width + 3)
        draw.line(points, fill="#087e73", width=line_width)
    for index, (x, y) in enumerate(points):
        color = "#e66b3d" if index == 0 else "#087e73"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=color, outline="white", width=2)
        draw.text((x + radius + 3, y - radius), str(index),
                  fill="white", stroke_width=2, stroke_fill="#253534")
    overlay.save(path)


def main():
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit("输入图片不存在：{}".format(args.image))
    steps = (args.max_steps if args.max_steps is not None
             else MAX_STEPS[args.condition])
    engine = InferenceEngine(PROJECT_ROOT, device=args.device)
    try:
        with Image.open(args.image) as image:
            image.load()
            result = engine.predict(
                image, args.task, condition=args.condition,
                decoding=args.decoding, max_steps=steps,
                stop_threshold=args.stop_threshold, seed=args.seed,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = args.output_dir / "scanpath.png"
            save_overlay(image, result, overlay_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("推理失败：{}".format(exc)) from exc

    # 热图为供网页显示的 base64 数据，结构化结果只保留数值。
    summary = {key: result[key] for key in (
        "condition", "task", "task_id", "decoding", "seed",
        "stop_threshold", "width", "height", "original_size",
        "fixations", "stop_reason", "model", "timing",
    )}
    summary["steps"] = [
        {key: value for key, value in step.items() if key != "heatmap"}
        for step in result["steps"]
    ]
    json_path = args.output_dir / "scanpath.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print("轨迹图：{}\n结构化结果：{}".format(overlay_path, json_path))


if __name__ == "__main__":
    main()
