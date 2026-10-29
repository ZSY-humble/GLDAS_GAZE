"""GLDAS-Gaze 单图推理；网页启动时不加载权重或训练数据。"""

import base64
import importlib.metadata
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import threading
import time


TASKS = (
    "bottle", "bowl", "car", "chair", "clock", "cup", "fork",
    "keyboard", "knife", "laptop", "microwave", "mouse", "oven",
    "potted plant", "sink", "stop sign", "toilet", "tv",
)
TASK_LABELS = dict(zip(TASKS, (
    "瓶子", "碗", "汽车", "椅子", "时钟", "杯子", "叉子", "键盘", "刀",
    "笔记本电脑", "微波炉", "鼠标", "烤箱", "盆栽", "水槽", "停止标志",
    "马桶", "电视",
)))
MAX_STEPS = {"TP": 6, "TA": 10}


class InferenceEngine:
    """缓存 TP/TA 模型，并将单图解码为网页可直接展示的结果。"""

    def __init__(self, project_root, device="auto"):
        self.root = Path(project_root).resolve()
        if device not in ("auto", "cpu", "cuda") and not (
            isinstance(device, str) and device.startswith("cuda:")
        ):
            raise ValueError("设备须为 auto、cpu、cuda 或 cuda:设备编号。")
        self.requested_device = device
        self._device = None
        self._models = {}
        self._lock = threading.RLock()
        self._last_error = None

    def status(self):
        """快速检查文件和依赖，不导入 PyTorch，也不读取大型权重。"""
        dependencies = {}
        errors = []
        for module, package in (
            ("torch", "torch"), ("torchvision", "torchvision"),
            ("detectron2", "detectron2"), ("numpy", "numpy"),
            ("PIL", "Pillow"), ("yaml", "PyYAML"),
            ("fvcore", "fvcore"), ("yacs", "yacs"),
        ):
            found = importlib.util.find_spec(module) is not None
            version = None
            if found:
                try:
                    version = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    version = "已安装"
            else:
                errors.append("缺少依赖：" + package)
            dependencies[package] = {"installed": found, "version": version}

        torch_version = dependencies["torch"]["version"] or ""
        numpy_version = dependencies["numpy"]["version"] or ""
        if torch_version.startswith("1.") and numpy_version.startswith("2."):
            errors.append("PyTorch 1.x 需要 NumPy 1.x，请安装 numpy<2。")

        common_files = (
            "pretrained_models/M2F_R50.pkl",
            "pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl",
            "configs/resnet50.yaml",
        )
        missing_common = [p for p in common_files if not (self.root / p).is_file()]
        errors.extend("缺少文件：" + p for p in missing_common)
        models = {}
        for condition in MAX_STEPS:
            checkpoint = "pretrained_models/{}.pt".format(condition)
            config = "configs/coco_search18_dense_SSL_{}.json".format(condition)
            available = all((self.root / p).is_file() for p in (checkpoint, config))
            models[condition] = {
                "available": available and not missing_common,
                "loaded": condition in self._models,
                "checkpoint": checkpoint,
                "max_steps": MAX_STEPS[condition],
            }
        if not any(model["available"] for model in models.values()):
            errors.append("未找到可用的 TP/TA 配置及模型权重。")
        return {
            "ready": not errors,
            "device": self._device or self.requested_device,
            "requested_device": self.requested_device,
            "models": models,
            "tasks": list(TASKS),
            "task_options": [
                {"id": i, "name": task, "label": TASK_LABELS[task]}
                for i, task in enumerate(TASKS)
            ],
            "dependencies": dependencies,
            "errors": errors,
            "error": "；".join(errors) if errors else None,
            "last_error": self._last_error,
        }

    def _runtime(self):
        try:
            import numpy as np
            import torch
            import yaml
            from PIL import Image, ImageOps
            from common.config import JsonConfig
            from gldas_gaze.models import GLDASGazeModel
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "模型运行环境不完整：{}。请使用已安装项目依赖的 Python 环境。"
                .format(exc)
            ) from exc

        if self._device is None:
            if self.requested_device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            elif self.requested_device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA 当前不可用，请使用 --device cpu。")
                self._device = self.requested_device
            else:
                self._device = "cpu"
            if self._device == "cpu":
                # 单图推理限制线程，避免在大量逻辑核机器上过度调度。
                torch.set_num_threads(min(4, torch.get_num_threads()))
        return np, torch, yaml, Image, ImageOps, JsonConfig, GLDASGazeModel

    def _load_model(self, condition, runtime):
        if condition in self._models:
            return self._models[condition], 0.0
        started = time.perf_counter()
        np, torch, yaml, _, _, JsonConfig, Model = runtime
        config_path = self.root / (
            "configs/coco_search18_dense_SSL_{}.json".format(condition)
        )
        checkpoint_path = self.root / "pretrained_models" / (condition + ".pt")
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError("缺少 {} 的配置或模型权重。".format(condition))
        with config_path.open(encoding="utf-8") as stream:
            hparams = JsonConfig(json.load(stream))

        # 训练配置可能记录旧机器绝对路径，推理只引用当前项目的权重。
        backbone_config = self.root / "configs/resnet50.yaml"
        with backbone_config.open(encoding="utf-8") as stream:
            backbone = yaml.safe_load(stream)
        backbone["MODEL"]["WEIGHTS"] = str(
            self.root / "pretrained_models/M2F_R50.pkl"
        )
        with tempfile.TemporaryDirectory(prefix="gldas-inference-") as folder:
            resolved_config = Path(folder) / "backbone.yaml"
            resolved_config.write_text(
                yaml.safe_dump(backbone, allow_unicode=True), encoding="utf-8"
            )
            hparams.Data.backbone_config = str(resolved_config)
            model = Model(
                hparams.Data,
                num_decoder_layers=hparams.Model.n_dec_layers,
                hidden_dim=hparams.Model.embedding_dim,
                nhead=hparams.Model.n_heads,
                ntask=len(TASKS),
                tgt_vocab_size=(
                    hparams.Data.patch_count + len(hparams.Data.special_symbols)
                ),
                num_output_layers=hparams.Model.num_output_layers,
                separate_fix_arch=hparams.Model.separate_fix_arch,
                train_encoder=False,
                train_pixel_decoder=False,
                train_ema=False,
                use_dino=hparams.Train.use_dino_pretrained_model,
                dropout=hparams.Train.dropout,
                dim_feedforward=hparams.Model.hidden_dim,
                parallel_arch=hparams.Model.parallel_arch,
                dorsal_source=hparams.Model.dorsal_source,
                num_encoder_layers=hparams.Model.n_enc_layers,
                output_centermap="centermap_pred" in hparams.Train.losses,
                output_saliency="saliency_pred" in hparams.Train.losses,
                output_target_map="target_map_pred" in hparams.Train.losses,
                transfer_learning_setting=hparams.Train.transfer_learn,
                project_queries=hparams.Train.project_queries,
                is_pretraining=False,
                output_feature_map_name=hparams.Model.output_feature_map_name,
                cross_stream_scale=hparams.Model.get("cross_stream_scale", 0.35),
            )

        # 只加载项目维护者提供的本地权重，不接受网页上传任意检查点。
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        state = checkpoint.get("model", checkpoint)
        if state and all(name.startswith("module.") for name in state):
            state = {name[7:]: value for name, value in state.items()}
        active_keys = set(model.state_dict())
        missing = active_keys - set(state)
        extras = set(state) - active_keys
        allowed_legacy = {"dorsal_ind_emb.weight"}
        if condition == "TA":
            allowed_legacy.update({
                "dorsal_stream_token", "ventral_ind_emb.weight",
                "scale_cond.to_gamma_beta.0.bias",
                "scale_cond.to_gamma_beta.0.weight",
                "scale_cond.to_gamma_beta.2.bias",
                "scale_cond.to_gamma_beta.2.weight",
            })
        if missing or not extras.issubset(allowed_legacy):
            raise RuntimeError(
                "模型参数结构不一致，缺失：{}；未知参数：{}。".format(
                    sorted(missing), sorted(extras - allowed_legacy)
                )
            )
        # 仅忽略已经确认不参与当前网络计算的旧版冗余参数。
        model.load_state_dict(
            {name: value for name, value in state.items() if name in active_keys},
            strict=True,
        )
        del checkpoint, state
        model.to(self._device)
        model.eval()
        model.requires_grad_(False)

        # TP 的 zero_shot 配置以该映射训练；TA 使用类别名称的字母序。
        task_ids = dict(zip(TASKS, range(len(TASKS))))
        mapping_path = self.root / "dataset/all_task_ids.npy"
        if hparams.Train.zero_shot and mapping_path.is_file():
            mapping = np.load(str(mapping_path), allow_pickle=True).item()
            task_ids = {str(name): int(index) for name, index in mapping.items()}
            if set(task_ids) != set(TASKS) or set(task_ids.values()) != set(range(18)):
                raise RuntimeError("all_task_ids.npy 与 18 类模型任务映射不一致。")
        cached = (model, hparams, task_ids)
        self._models[condition] = cached
        return cached, (time.perf_counter() - started) * 1000

    @staticmethod
    def _heatmap_data_url(values, np, Image):
        """将模型下一注视点分数渲染为透明热图，不伪造注视密度。"""
        values = values.detach().cpu().numpy().astype("float32")
        low, high = float(values.min()), float(values.max())
        normalized = (values - low) / max(high - low, 1e-12)
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255
        rgba[:, :, 1] = np.clip(normalized * 255, 0, 255).astype("uint8")
        rgba[:, :, 3] = np.clip(normalized * 210, 0, 210).astype("uint8")
        output = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(
            output.getvalue()
        ).decode("ascii")

    def predict(self, image, task, condition="TP", decoding="greedy",
                max_steps=6, stop_threshold=0.5, seed=42):
        """执行真实单图推理；max_steps 是解码次数，另含中心起点。"""
        condition = str(condition).upper()
        if condition not in MAX_STEPS:
            raise ValueError("模型条件须为 TP 或 TA。")
        if task not in TASKS:
            raise ValueError("请选择 COCO-Search18 的 18 个搜索类别之一。")
        if decoding not in ("greedy", "sample"):
            raise ValueError("解码模式须为 greedy 或 sample。")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise ValueError("最大步数须为整数。")
        if not 1 <= max_steps <= MAX_STEPS[condition]:
            raise ValueError("{} 最大步数应为 1 至 {}。".format(
                condition, MAX_STEPS[condition]
            ))
        if not isinstance(stop_threshold, (int, float)) or not (
            math.isfinite(stop_threshold) and 0 <= stop_threshold <= 1
        ):
            raise ValueError("停止阈值须在 0 至 1 之间。")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
            raise ValueError("随机种子须为 0 至 4294967295 的整数。")

        with self._lock:
            started = time.perf_counter()
            try:
                result = self._predict_locked(
                    image, task, condition, decoding,
                    max_steps, float(stop_threshold), seed, started
                )
                self._last_error = None
                return result
            except Exception as exc:
                self._last_error = str(exc)
                raise

    def _predict_locked(self, image, task, condition, decoding,
                        max_steps, stop_threshold, seed, started):
        runtime = self._runtime()
        np, torch, _, Image, ImageOps, _, _ = runtime
        (model, hparams, task_ids), load_ms = self._load_model(condition, runtime)
        pa = hparams.Data
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = {"width": image.width, "height": image.height}
        resized = image.resize((pa.im_w, pa.im_h), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self._device)
        generator = torch.Generator(device=self._device)
        generator.manual_seed(seed)
        task_tensor = torch.tensor([task_ids[task]], device=self._device)
        fixations = [{"x": pa.im_w / 2, "y": pa.im_h / 2,
                      "stop_probability": None}]
        points = torch.full((1, 1, 2), 0.5, device=self._device)
        yy = torch.arange(pa.im_h, device=self._device).view(-1, 1)
        xx = torch.arange(pa.im_w, device=self._device).view(1, -1)

        def inhibition(x, y):
            return ((xx - x) ** 2 + (yy - y) ** 2 <= pa.IOR_radius ** 2)

        mask = inhibition(pa.im_w / 2, pa.im_h / 2)
        steps = []
        stop_reason = "max_steps"
        inference_started = time.perf_counter()
        with torch.inference_mode():
            encoded = model.encode(tensor)
            for index in range(max_steps):
                # 与 common.utils.transform_fixations 的推理分支一致。
                low_grid = torch.tensor(
                    [pa.im_w // 32, pa.im_h // 32], device=self._device
                )
                high_grid = torch.tensor(
                    [pa.im_w // 4, pa.im_h // 4], device=self._device
                )
                low = (points * low_grid).long()
                high = (points * high_grid).long()
                low_tokens = low_grid[0] * low[:, :, 1] + low[:, :, 0] + 1
                high_tokens = high_grid[0] * high[:, :, 1] + high[:, :, 0] + 1
                dorsal, position, padding, feature_maps = encoded
                output = model.decode_and_predict(
                    dorsal.clone(), position, padding, feature_maps,
                    low_tokens, None, high_tokens, task_tensor
                )
                scores = output["pred_fixation_map"].reshape(pa.im_h, pa.im_w)
                stop = float(output["pred_termination"].reshape(-1)[0].item())
                if not math.isfinite(stop) or not bool(torch.isfinite(scores).all()):
                    raise RuntimeError("模型输出包含无效数值，请检查权重和运行环境。")
                fixations[-1]["stop_probability"] = stop
                heatmap = self._heatmap_data_url(scores, np, Image)
                step = {"step": index + 1, "stop_probability": stop,
                        "heatmap": heatmap, "x": None, "y": None}
                steps.append(step)
                # 先判断是否停止；与评估中截断后的轨迹语义一致。
                if stop > stop_threshold:
                    stop_reason = "model_stop"
                    break

                scores = scores.clone()
                if pa.enforce_IOR:
                    scores[mask] = 0
                flat = scores.reshape(-1)
                if float(flat.sum()) <= 1e-12:
                    flat = (~mask).to(flat.dtype).reshape(-1)
                    if not bool(flat.any()):
                        stop_reason = "all_locations_inhibited"
                        break
                if decoding == "sample":
                    location = int(torch.multinomial(
                        flat, 1, generator=generator
                    ).item())
                else:
                    location = int(flat.argmax().item())
                x, y = location % pa.im_w, location // pa.im_w
                fixations.append({"x": x, "y": y, "stop_probability": None})
                step.update({"x": x, "y": y})
                next_point = torch.tensor(
                    [[[x / pa.im_w, y / pa.im_h]]], device=self._device
                )
                points = torch.cat([points, next_point], dim=1)
                mask |= inhibition(x, y)
        if self._device.startswith("cuda"):
            torch.cuda.synchronize(self._device)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        return {
            "condition": condition,
            "task": task,
            "task_id": task_ids[task],
            "decoding": decoding,
            "seed": seed,
            "stop_threshold": stop_threshold,
            "width": pa.im_w,
            "height": pa.im_h,
            "original_size": original_size,
            "fixations": fixations,
            "steps": steps,
            "heatmap": steps[-1]["heatmap"] if steps else None,
            "stop_reason": stop_reason,
            "model": {
                "checkpoint": "pretrained_models/{}.pt".format(condition),
                "device": self._device,
                "max_steps": max_steps,
                "trained_max_steps": MAX_STEPS[condition],
            },
            "timing": {
                "load_ms": round(load_ms, 2),
                "inference_ms": round(inference_ms, 2),
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        }
