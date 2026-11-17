import argparse
import os
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import JsonConfig
from gldas_gaze.builder import build
from gldas_gaze.evaluation import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        "Cross-condition evaluation for GLDAS-Gaze"
    )
    parser.add_argument(
        "--model-hparams", type=str, required=True,
        help="Config used to build the trained model architecture."
    )
    parser.add_argument(
        "--eval-hparams", type=str, required=True,
        help="Config used to build the evaluation data protocol."
    )
    parser.add_argument(
        "--dataset-root", type=str, required=True,
        help="Dataset root directory."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to the trained checkpoint."
    )
    parser.add_argument(
        "--tap", choices=["TP", "TA"], required=True,
        help="Evaluation condition: TP or TA."
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument(
        "--eval-mode", choices=["greedy", "sample"], default="greedy",
        help="Scanpath decoding strategy."
    )
    parser.add_argument(
        "--disable-saliency", action="store_true",
        help="Disable cIG/cNSS/cAUC computation."
    )
    parser.add_argument(
        "--max-eval-steps", type=int, default=None,
        help="Override eval max_traj_length. Default: use eval config."
    )
    parser.add_argument(
        "--extend-time-embedding", choices=["repeat_last", "zero"],
        default="repeat_last",
        help="How to extend fix_ind_emb when eval steps exceed training steps."
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory for metrics and prediction JSON files."
    )
    parser.add_argument(
        "--save-overlays", action="store_true",
        help="Save step-wise scanpath overlay visualizations."
    )
    parser.add_argument(
        "--overlay-dir", type=str, default="scanpath_step_overlays",
        help="Directory for overlay visualizations."
    )
    return parser.parse_args()


def _inner_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def extend_fixation_time_embedding(model, target_len, mode):
    """跨条件评估时，补齐未训练过的时间步嵌入，避免 TA 长序列越界。"""
    inner = _inner_model(model)
    old_emb = inner.fix_ind_emb
    old_len, dim = old_emb.weight.shape
    if target_len <= old_len:
        return False

    new_emb = nn.Embedding(target_len, dim).to(old_emb.weight.device)
    with torch.no_grad():
        new_emb.weight[:old_len].copy_(old_emb.weight)
        if mode == "repeat_last":
            fill_value = old_emb.weight[-1:].repeat(target_len - old_len, 1)
        elif mode == "zero":
            fill_value = torch.zeros(
                target_len - old_len, dim, device=old_emb.weight.device
            )
        else:
            raise ValueError(f"Unsupported extension mode: {mode}")
        new_emb.weight[old_len:].copy_(fill_value)

    inner.fix_ind_emb = new_emb
    inner.pa.max_traj_length = max(inner.pa.max_traj_length, target_len)
    return True


def load_checkpoint(model, ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {missing}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {unexpected}")


def unpack_eval_bundle(bundle):
    return {
        "valid_img_loader_tp": bundle[5],
        "valid_img_loader_ta": bundle[6],
        "bbox_annos": bundle[8],
        "human_cdf": bundle[9],
        "fix_clusters": bundle[10],
        "prior_maps_tp": bundle[11],
        "prior_maps_ta": bundle[12],
        "sss_strings": bundle[13],
        "valid_gaze_loader_tp": bundle[14],
        "valid_gaze_loader_ta": bundle[15],
        "sps_test_tp": bundle[16],
        "sps_test_ta": bundle[17],
    }


def main():
    args = parse_args()
    dataset_root = args.dataset_root.rstrip("/")
    sample_action = args.eval_mode == "sample"
    output_saliency_metrics = not args.disable_saliency

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model_hparams = JsonConfig(args.model_hparams)
    eval_hparams = JsonConfig(args.eval_hparams)
    if args.max_eval_steps is not None:
        eval_hparams.Data.max_traj_length = args.max_eval_steps

    print(f"Building model from: {args.model_hparams}")
    model_bundle = build(
        model_hparams, dataset_root, device,
        is_pretraining=False, is_eval=True, split=args.split
    )
    model = model_bundle[0]
    load_checkpoint(model, args.ckpt, device)
    model.to(device)
    model.eval()

    eval_steps = eval_hparams.Data.max_traj_length
    if extend_fixation_time_embedding(
        model, eval_steps, args.extend_time_embedding
    ):
        print(
            "Extended fixation time embedding to "
            f"{eval_steps} steps with mode={args.extend_time_embedding}."
        )

    # 评估数据按 eval-hparams 构建，确保 TP/TA 的长度和数据过滤协议正确。
    print(f"Building evaluation data from: {args.eval_hparams}")
    eval_bundle = build(
        eval_hparams, dataset_root, torch.device("cpu"),
        is_pretraining=False, is_eval=True, split=args.split
    )
    eval_data = unpack_eval_bundle(eval_bundle)
    del eval_bundle

    if args.tap == "TP":
        valid_img_loader = eval_data["valid_img_loader_tp"]
        valid_gaze_loader = eval_data["valid_gaze_loader_tp"]
        prior_maps = eval_data["prior_maps_tp"]
        human_scanpath_test = eval_data["sps_test_tp"]
    else:
        valid_img_loader = eval_data["valid_img_loader_ta"]
        valid_gaze_loader = eval_data["valid_gaze_loader_ta"]
        prior_maps = eval_data["prior_maps_ta"]
        human_scanpath_test = eval_data["sps_test_ta"]

    log_dir = args.log_dir
    if log_dir is None:
        model_name = os.path.splitext(os.path.basename(args.model_hparams))[0]
        eval_name = os.path.splitext(os.path.basename(args.eval_hparams))[0]
        log_dir = os.path.join(
            "assets", "cross_condition_eval",
            f"{model_name}__to__{eval_name}__{args.tap}"
        )
    os.makedirs(log_dir, exist_ok=True)

    center_initial = eval_hparams.Data.name in ["COCO-Search18"]
    metrics_dict, _ = evaluate(
        model, device,
        valid_img_loader, valid_gaze_loader,
        eval_hparams.Data,
        eval_data["bbox_annos"], eval_data["human_cdf"],
        eval_data["fix_clusters"], prior_maps,
        eval_data["sss_strings"], dataset_root,
        human_scanpath_test,
        sample_action=sample_action,
        output_saliency_metrics=output_saliency_metrics,
        center_initial=center_initial,
        log_dir=log_dir,
        save_overlays=args.save_overlays,
        overlay_dir=args.overlay_dir,
    )

    print(f"{args.tap}:", metrics_dict)
    print(f"Saved outputs to: {log_dir}")


if __name__ == "__main__":
    main()
