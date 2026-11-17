"""提供同源网页与单图推理 API，默认仅监听本机。"""

import argparse
import base64
import binascii
import io
import json
import logging
import math
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlsplit
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .engine import InferenceEngine, MAX_STEPS, TASK_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 15 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
LABELS = TASK_LABELS
LOGGER = logging.getLogger(__name__)


class APIError(Exception):
    """可直接展示给页面的请求错误。"""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def read_image(content):
    """验证图片并统一 EXIF 方向，拒绝超大图片和不支持的格式。"""
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise APIError(413, "请选择不超过 10 MB 的图片。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise APIError(400, "支持 JPG、PNG、WebP 图片。")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise APIError(413, "请将图片缩小至 2400 万像素以内。")
                return ImageOps.exif_transpose(image).convert("RGB")
    except (
        UnidentifiedImageError, OSError, Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise APIError(400, "图片无法读取，请重新选择文件。") from exc


def image_data_url(image):
    """返回方向已校正的预览，避免返回原始 EXIF 和过大的图片。"""
    preview = image.copy()
    preview.thumbnail((2048, 2048))
    stream = io.BytesIO()
    preview.save(stream, format="JPEG", quality=90)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


class Application:
    """封装请求校验、数据集示例与推理串行访问。"""

    def __init__(self, project_root=PROJECT_ROOT, device="auto", engine=None,
                 dataset_root=None):
        self.root = Path(project_root).resolve()
        self.dataset_root = Path(dataset_root or self.root / "dataset").resolve()
        self.engine = engine or InferenceEngine(self.root, device=device)
        self.lock = threading.Lock()

    def status(self):
        state = self.engine.status()
        state["tasks"] = [{"id": key, "label": label}
                          for key, label in LABELS.items()]
        state["max_steps"] = MAX_STEPS
        state["busy"] = self.lock.locked()
        return state

    def example_path(self, example_id):
        if not isinstance(example_id, str):
            raise APIError(400, "示例编号无效。")
        parts = example_id.split("/")
        if len(parts) != 3:
            raise APIError(404, "示例不存在。")
        condition, category, filename = parts
        if (condition not in MAX_STEPS
                or category.replace("_", " ") not in LABELS
                or not filename or filename in {".", ".."}
                or "\\" in filename):
            raise APIError(404, "示例不存在。")
        directory = "images" if condition == "TP" else "imagesTA"
        image_root = (self.dataset_root / directory).resolve()
        path = (image_root / category / filename).resolve()
        if (image_root not in path.parents
                or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
                or not path.is_file()):
            raise APIError(404, "示例不存在。")
        return path

    def examples(self, condition="TP", task="bottle"):
        if condition not in MAX_STEPS or task not in LABELS:
            raise APIError(400, "请选择有效的模型条件和目标类别。")
        category = task.replace(" ", "_")
        directory = "images" if condition == "TP" else "imagesTA"
        folder = self.dataset_root / directory / category
        rows = []
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                identifier = "/".join((condition, category, path.name))
                # 与下载接口使用同一校验，防止越界的符号链接成为示例。
                try:
                    self.example_path(identifier)
                except APIError:
                    continue
                rows.append({
                    "id": identifier, "label": LABELS[task] + " · " + path.stem,
                    "url": "/api/example?id=" + quote(identifier, safe=""),
                    "task": task, "condition": condition,
                })
                if len(rows) == 6:
                    break
        return {"examples": rows}

    def predict(self, payload):
        if not isinstance(payload, dict):
            raise APIError(400, "请求必须为 JSON 对象。")
        condition = payload.get("condition", "TP")
        task = payload.get("task")
        decoding = payload.get("decoding", "greedy")
        if not isinstance(condition, str) or condition not in MAX_STEPS:
            raise APIError(400, "模型条件必须为 TP 或 TA。")
        if not isinstance(task, str) or task not in LABELS:
            raise APIError(400, "请选择支持的 18 个目标类别之一。")
        if decoding not in ("greedy", "sample"):
            raise APIError(400, "解码方式必须为 greedy 或 sample。")
        max_steps = payload.get("max_steps", MAX_STEPS[condition])
        threshold = payload.get("stop_threshold", 0.5)
        seed = payload.get("seed", 42)
        if (type(max_steps) is not int
                or not 1 <= max_steps <= MAX_STEPS[condition]):
            raise APIError(400, "解码步数超出当前模型支持范围。")
        if (type(threshold) not in (int, float)
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise APIError(400, "停止阈值必须在 0 到 1 之间。")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise APIError(400, "随机种子必须为 0 至 4294967295 的整数。")
        data = payload.get("image")
        example_id = payload.get("example_id")
        if bool(data) == bool(example_id):
            raise APIError(400, "请提供一张上传图片或一个示例编号。")
        if example_id:
            if not isinstance(example_id, str):
                raise APIError(400, "示例编号无效。")
            # 示例必须与当前选择对应，防止切换控件后使用过期示例。
            if example_id.split("/")[:2] != [condition, task.replace(" ", "_")]:
                raise APIError(400, "示例与当前条件或目标类别不一致，请重新选择。")
            raw = self.example_path(example_id).read_bytes()
        else:
            if not isinstance(data, str) or not data.startswith((
                "data:image/jpeg;base64,", "data:image/png;base64,",
                "data:image/webp;base64,",
            )):
                raise APIError(400, "请上传 JPG、PNG 或 WebP 图片。")
            try:
                raw = base64.b64decode(data.split(",", 1)[1], validate=True)
            except (ValueError, binascii.Error) as exc:
                raise APIError(400, "图片数据不完整，请重新上传。") from exc
        image = read_image(raw)
        if not self.lock.acquire(blocking=False):
            raise APIError(429, "模型正在处理另一张图片，请稍候重试。")
        try:
            prediction = self.engine.predict(
                image, task, condition=condition, decoding=decoding,
                max_steps=max_steps, stop_threshold=threshold, seed=seed,
            )
        except (ImportError, FileNotFoundError) as exc:
            raise APIError(503, "推理依赖或权重未就绪：" + str(exc)) from exc
        except RuntimeError as exc:
            LOGGER.exception("模型推理失败")
            raise APIError(503, str(exc)) from exc
        finally:
            self.lock.release()
        timing = prediction.get("timing", {})
        return {
            **prediction,
            "image": {
                "width": image.width, "height": image.height,
                "model_width": prediction.get("width", 512),
                "model_height": prediction.get("height", 320),
                "data_url": image_data_url(image),
            },
            "heatmaps": [step["heatmap"] for step in prediction.get("steps", [])],
            "meta": {
                "condition": condition, "task": task, "decoding": decoding,
                "device": prediction.get("model", {}).get("device", "unknown"),
                "elapsed_ms": timing.get("total_ms", 0),
                "inference_ms": timing.get("inference_ms", 0),
                "stop_reason": prediction.get("stop_reason"),
                "max_steps": max_steps, "stop_threshold": threshold,
                "seed": seed, "center_initial": True,
            },
        }


def make_handler(application):
    """生成只开放明确资源的 HTTP 处理器。"""

    class Handler(BaseHTTPRequestHandler):
        server_version = "GLDAS-Gaze"

        def _send(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(content)

        def _json(self, status, data):
            content = json.dumps(data, ensure_ascii=False, allow_nan=False)
            self._send(status, content.encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            route = urlsplit(self.path)
            try:
                if route.path == "/api/status":
                    return self._json(200, application.status())
                query = parse_qs(route.query)
                if route.path == "/api/examples":
                    return self._json(200, application.examples(
                        query.get("condition", ["TP"])[0],
                        query.get("task", ["bottle"])[0],
                    ))
                if route.path == "/api/example":
                    path = application.example_path(query.get("id", [""])[0])
                    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".png": "image/png", ".webp": "image/webp"}
                    return self._send(200, path.read_bytes(), mime[path.suffix.lower()])
                files = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
                    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
                }
                if route.path not in files:
                    raise APIError(404, "资源不存在。")
                filename, mime = files[route.path]
                self._send(200, (STATIC_ROOT / filename).read_bytes(), mime)
            except APIError as exc:
                self._json(exc.status, {"error": str(exc)})
            except Exception:
                LOGGER.exception("读取资源失败")
                self._json(500, {"error": "资源暂时不可用，请查看服务日志。"})

        def do_POST(self):
            try:
                if urlsplit(self.path).path != "/api/predict":
                    raise APIError(404, "接口不存在。")
                origin = self.headers.get("Origin")
                if origin and origin not in (
                    "http://" + self.headers.get("Host", ""),
                    "https://" + self.headers.get("Host", ""),
                ):
                    raise APIError(403, "请从当前演示页面发起推理。")
                if self.headers.get_content_type() != "application/json":
                    raise APIError(415, "请求格式必须为 application/json。")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise APIError(400, "请求长度无效。") from exc
                if not 0 < length <= MAX_BODY_BYTES:
                    raise APIError(413, "请求为空或图片超过大小限制。")
                try:
                    payload = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeError) as exc:
                    raise APIError(400, "请求 JSON 无效。") from exc
                self._json(200, application.predict(payload))
            except APIError as exc:
                self._json(exc.status, {"error": str(exc)})
            except Exception:
                LOGGER.exception("处理推理请求失败")
                self._json(500, {"error": "推理未完成，请查看服务日志后重试。"})

        def log_message(self, message, *args):
            LOGGER.info("%s %s", self.client_address[0], message % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="GLDAS-Gaze 网页单图推理")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    application = Application(device=args.device, dataset_root=args.dataset_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(application))
    LOGGER.info("网页推理已启动：http://%s:%d", args.host, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("网页推理服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
