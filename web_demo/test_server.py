"""验证网页上传边界和 HTTP 协议，不加载模型权重。"""

import base64
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from .server import APIError, Application, make_handler
from http.server import ThreadingHTTPServer


def png_bytes(size=(32, 20)):
    """构造真实可解码的小图片，避免读取完整数据集。"""
    stream = io.BytesIO()
    Image.new("RGB", size, (30, 90, 150)).save(stream, "PNG")
    return stream.getvalue()


def upload_payload(**updates):
    payload = {
        "image": "data:image/png;base64," + base64.b64encode(
            png_bytes()).decode("ascii"),
        "task": "bottle", "condition": "TP", "max_steps": 3,
    }
    payload.update(updates)
    return payload


class FakeEngine:
    """仅替代昂贵计算，保留真实图片解析和 HTTP 处理。"""

    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.failure = None

    def status(self):
        return {"ready": True, "device": "cpu", "models": {
            "TP": {"available": True}, "TA": {"available": True},
        }}

    def predict(self, image, task, **options):
        self.calls.append((image.copy(), task, options))
        self.started.set()
        if self.block and not self.release.wait(timeout=5):
            raise RuntimeError("测试未及时释放推理任务。")
        if self.failure:
            raise self.failure
        heatmap = "data:image/png;base64," + base64.b64encode(
            png_bytes()).decode("ascii")
        return {
            "width": 512, "height": 320,
            "fixations": [
                {"x": 256, "y": 160, "stop_probability": 0.1},
                {"x": 128, "y": 80, "stop_probability": 0.8},
            ],
            "steps": [{"step": 1, "x": 128, "y": 80,
                       "stop_probability": 0.1, "heatmap": heatmap}],
            "stop_reason": "model_stop",
            "model": {"device": "cpu", "checkpoint": "TP.pt"},
            "timing": {"total_ms": 12.5, "inference_ms": 10.0},
        }


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        for directory in ("images", "imagesTA"):
            category = self.root / "dataset" / directory / "bottle"
            category.mkdir(parents=True)
            (category / "sample.png").write_bytes(png_bytes())
        self.engine = FakeEngine()
        self.app = Application(self.root, engine=self.engine)

    def assert_api_error(self, status, payload):
        with self.assertRaises(APIError) as caught:
            self.app.predict(payload)
        self.assertEqual(caught.exception.status, status)
        self.assertEqual(self.engine.calls, [])

    def test_upload_returns_preview_coordinates_and_run_metadata(self):
        result = self.app.predict(upload_payload(
            decoding="sample", seed=123, stop_threshold=0.7))
        self.assertEqual(result["image"]["width"], 32)
        self.assertEqual(result["image"]["height"], 20)
        self.assertEqual(result["image"]["model_width"], 512)
        self.assertEqual(result["image"]["model_height"], 320)
        preview = result["image"]["data_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(preview))) as image:
            self.assertEqual(image.size, (32, 20))
        self.assertEqual(result["fixations"][1]["x"], 128)
        self.assertEqual(len(result["heatmaps"]), 1)
        self.assertEqual(result["meta"]["seed"], 123)
        self.assertEqual(result["meta"]["decoding"], "sample")
        self.assertEqual(result["meta"]["stop_reason"], "model_stop")
        self.assertEqual(result["meta"]["elapsed_ms"], 12.5)
        image, task, options = self.engine.calls[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(task, "bottle")
        self.assertEqual(options["seed"], 123)
        json.dumps(result, allow_nan=False)

    def test_example_discovery_and_predict_use_matching_image(self):
        for condition in ("TP", "TA"):
            with self.subTest(condition=condition):
                example = self.app.examples(condition, "bottle")["examples"][0]
                result = self.app.predict({
                    "example_id": example["id"], "task": "bottle",
                    "condition": condition,
                })
                self.assertEqual(result["meta"]["condition"], condition)
                self.assertEqual(result["image"]["width"], 32)
                self.assertEqual(example["condition"], condition)
                self.assertTrue(example["url"].startswith("/api/example?id="))

    def test_exif_orientation_matches_inference_and_preview(self):
        stream = io.BytesIO()
        image = Image.new("RGB", (40, 20), (0, 100, 0))
        exif = image.getexif()
        exif[274] = 6
        image.save(stream, "JPEG", exif=exif)
        payload = upload_payload(image="data:image/jpeg;base64," +
                                 base64.b64encode(stream.getvalue()).decode())
        result = self.app.predict(payload)
        self.assertEqual(self.engine.calls[0][0].size, (20, 40))
        self.assertEqual((result["image"]["width"],
                          result["image"]["height"]), (20, 40))

    def test_invalid_parameters_never_reach_model(self):
        cases = [
            {"task": "person"}, {"task": []}, {"condition": "bad"},
            {"decoding": "beam"}, {"max_steps": 0}, {"max_steps": 7},
            {"condition": "TA", "max_steps": 11}, {"max_steps": True},
            {"max_steps": 2.5}, {"max_steps": "3"}, {"seed": -1},
            {"seed": 2**32}, {"seed": False}, {"seed": 1.1},
            {"stop_threshold": -0.1}, {"stop_threshold": 1.1},
            {"stop_threshold": float("nan")},
            {"stop_threshold": float("inf")}, {"stop_threshold": True},
            {"stop_threshold": "0.5"},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                self.assert_api_error(400, upload_payload(**changes))

    def test_numeric_boundaries_are_accepted(self):
        for threshold, seed in ((0, 0), (1, 2**32 - 1)):
            with self.subTest(threshold=threshold, seed=seed):
                result = self.app.predict(upload_payload(
                    stop_threshold=threshold, seed=seed, max_steps=1))
                self.assertEqual(result["meta"]["seed"], seed)

    def test_missing_ambiguous_or_mismatched_image_is_rejected(self):
        cases = [[], {}, {"task": "bottle"},
                 upload_payload(example_id="TP/bottle/sample.png"),
                 {"task": "bottle", "condition": "TA",
                  "example_id": "TP/bottle/sample.png"},
                 {"task": "cup", "example_id": "TP/bottle/sample.png"}]
        for payload in cases:
            with self.subTest(payload_type=type(payload).__name__):
                self.assert_api_error(400, payload)

    def test_corrupt_unsupported_or_large_images_are_rejected(self):
        self.assert_api_error(400, upload_payload(
            image="data:image/png;base64,%%%"))
        self.assert_api_error(400, upload_payload(
            image="data:image/png;base64," + base64.b64encode(
                b"not an image").decode()))
        self.assert_api_error(400, upload_payload(
            image="data:image/svg+xml;base64,AAAA"))
        with patch("web_demo.server.MAX_IMAGE_BYTES", 16):
            self.assert_api_error(413, upload_payload())
        with patch("web_demo.server.MAX_IMAGE_PIXELS", 10):
            self.assert_api_error(413, upload_payload())

    def test_paths_and_symlinks_cannot_escape_image_directory(self):
        bad_ids = ["../bottle/sample.png", "TP/../sample.png",
                   "TP/bottle/../../private.png", "TP/bottle/..",
                   "TP/bottle/..\\private.png", "/etc/passwd",
                   "TP/bottle/sample.txt"]
        for identifier in bad_ids:
            with self.subTest(identifier=identifier):
                with self.assertRaises(APIError) as caught:
                    self.app.example_path(identifier)
                self.assertEqual(caught.exception.status, 404)
        external = self.root / "private.png"
        external.write_bytes(png_bytes())
        symlink = self.root / "dataset/images/bottle/escape.png"
        symlink.symlink_to(external)
        with self.assertRaises(APIError) as caught:
            self.app.example_path("TP/bottle/escape.png")
        self.assertEqual(caught.exception.status, 404)
        rows = self.app.examples()["examples"]
        self.assertFalse(any("escape" in row["id"] for row in rows))

    def test_model_failure_releases_busy_lock(self):
        self.engine.failure = RuntimeError("模型暂时不可用")
        with self.assertLogs("web_demo.server", level="ERROR"):
            with self.assertRaises(APIError) as caught:
                self.app.predict(upload_payload())
        self.assertEqual(caught.exception.status, 503)
        self.assertFalse(self.app.lock.locked())
        self.engine.failure = None
        self.app.predict(upload_payload())
        self.assertEqual(len(self.engine.calls), 2)

    def test_status_exposes_tasks_limits_and_busy_state(self):
        state = self.app.status()
        self.assertTrue(state["ready"])
        self.assertEqual(len(state["tasks"]), 18)
        self.assertEqual(state["max_steps"], {"TP": 6, "TA": 10})
        self.assertFalse(state["busy"])
        with self.app.lock:
            self.assertTrue(self.app.status()["busy"])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        category = root / "dataset/images/bottle"
        category.mkdir(parents=True)
        (category / "sample.png").write_bytes(png_bytes())
        static = root / "static"
        static.mkdir()
        for name in ("index.html", "app.js", "style.css"):
            (static / name).write_text("test", encoding="utf-8")
        static_patch = patch("web_demo.server.STATIC_ROOT", static)
        static_patch.start()
        self.addCleanup(static_patch.stop)
        self.engine = FakeEngine()
        self.app = Application(root, engine=self.engine)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.engine.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            content = response.read()
            return response.status, dict(response.getheaders()), content
        finally:
            connection.close()

    def post(self, payload=None, **headers):
        return self.request("POST", "/api/predict", json.dumps(
            payload if payload is not None else upload_payload()),
            {"Content-Type": "application/json", **headers})

    def test_get_routes_and_resource_allowlist(self):
        for route, mime in (("/", "text/html"),
                            ("/static/app.js", "text/javascript"),
                            ("/static/style.css", "text/css"),
                            ("/api/status", "application/json"),
                            ("/api/examples", "application/json"),
                            ("/api/example?id=TP%2Fbottle%2Fsample.png",
                             "image/png")):
            with self.subTest(route=route):
                status, headers, body = self.request("GET", route)
                self.assertEqual(status, 200)
                self.assertIn(mime, headers["Content-Type"])
                self.assertEqual(int(headers["Content-Length"]), len(body))
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        for route in ("/../README.md", "/static/../server.py",
                      "/api/predict", "/unknown"):
            with self.subTest(route=route):
                self.assertEqual(self.request("GET", route)[0], 404)

    def test_post_predict_returns_json_schema(self):
        status, headers, body = self.post()
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertIn("fixations", result)
        self.assertIn("heatmaps", result)
        self.assertIn("image", result)
        self.assertEqual(result["meta"]["task"], "bottle")
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_cross_origin_post_is_rejected(self):
        status, _, body = self.post(Origin="https://another.example")
        self.assertEqual(status, 403)
        self.assertIn("error", json.loads(body))
        self.assertEqual(self.engine.calls, [])
        status, _, _ = self.post(Origin="null")
        self.assertEqual(status, 403)

    def test_same_origin_and_cli_post_are_accepted(self):
        origin = "http://127.0.0.1:" + str(self.server.server_port)
        self.assertEqual(self.post(Origin=origin)[0], 200)
        self.assertEqual(self.post()[0], 200)

    def test_invalid_http_envelope_returns_json_error(self):
        cases = [
            ("POST", "/unknown", "{}", {"Content-Type": "application/json"},
             404),
            ("POST", "/api/predict", "{}", {"Content-Type": "text/plain"},
             415),
            ("POST", "/api/predict", "{bad", {
                "Content-Type": "application/json"}, 400),
            ("POST", "/api/predict", "{}", {
                "Content-Type": "application/json", "Content-Length": "bad"},
             400),
            ("POST", "/api/predict", "", {
                "Content-Type": "application/json"}, 413),
        ]
        for method, route, body, headers, expected in cases:
            with self.subTest(route=route, expected=expected):
                status, _, content = self.request(method, route, body, headers)
                self.assertEqual(status, expected)
                self.assertIn("error", json.loads(content))
        with patch("web_demo.server.MAX_BODY_BYTES", 2):
            self.assertEqual(self.post()[0], 413)
        self.assertEqual(self.engine.calls, [])

    def test_concurrent_request_gets_busy_without_second_inference(self):
        self.engine.block = True
        results = []
        errors = []

        def first_request():
            try:
                results.append(self.post())
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=first_request)
        worker.start()
        try:
            self.assertTrue(self.engine.started.wait(timeout=3))
            busy_status = json.loads(self.request("GET", "/api/status")[2])
            self.assertTrue(busy_status["busy"])
            status, _, body = self.post()
            self.assertEqual(status, 429)
            self.assertIn("error", json.loads(body))
            self.assertEqual(len(self.engine.calls), 1)
        finally:
            self.engine.release.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results[0][0], 200)
        self.assertFalse(self.app.status()["busy"])


if __name__ == "__main__":
    unittest.main()
