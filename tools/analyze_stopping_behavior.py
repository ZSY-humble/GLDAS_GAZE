import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Analyze GLDAS-Gaze stopping behavior")
    parser.add_argument(
        "--dataset-root", type=str, required=True,
        help="COCO-Search18 dataset root."
    )
    parser.add_argument(
        "--pred-root", type=str, default="assets/cross_condition_eval",
        help="Root directory that contains predictions_*.json files."
    )
    parser.add_argument(
        "--output-dir", type=str, default="assets/stopping_analysis",
        help="Directory for summary files and length-distribution figures."
    )
    parser.add_argument(
        "--patterns", nargs="+", default=["**/predictions_*.json"],
        help="Glob patterns, relative to --pred-root, for prediction files."
    )
    parser.add_argument(
        "--max-plot-len", type=int, default=None,
        help="Maximum length bin shown in plots. Default: inferred from data."
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def target_hit_step(xs, ys, bbox):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    hit_x = np.logical_and(x > bbox[0], x < bbox[0] + bbox[2])
    hit_y = np.logical_and(y > bbox[1], y < bbox[1] + bbox[3])
    hit = np.logical_and(hit_x, hit_y)
    if np.any(hit):
        return int(np.argmax(hit)) + 1
    return None


def effective_human_length(sp):
    if sp["condition"] == "present" and sp.get("fixOnTarget", False):
        hit_step = target_hit_step(sp["X"], sp["Y"], sp["bbox"])
        if hit_step is not None:
            return hit_step
    return len(sp["X"])


def load_human_scanpaths(dataset_root):
    path = Path(dataset_root) / "coco_search_fixations_512x320_on_target_allvalid.json"
    scanpaths = load_json(path)
    groups = defaultdict(list)
    for sp in scanpaths:
        if sp.get("split") != "test":
            continue
        if sp["condition"] == "present" and not sp.get("fixOnTarget", False):
            continue
        key = (sp["task"], sp["name"], sp["condition"])
        groups[key].append({
            "length": effective_human_length(sp),
            "raw_length": len(sp["X"]),
        })
    return groups


def load_bbox_annos(dataset_root):
    bbox_path = Path(dataset_root) / "bbox_annos.npy"
    if not bbox_path.exists():
        return {}
    return np.load(bbox_path, allow_pickle=True).item()


def prediction_key(sp):
    return sp["task"], sp["name"], sp["condition"]


def prediction_length(sp):
    return len(sp["X"])


def summarize_prediction_file(pred_path, human_groups, bbox_annos):
    preds = load_json(pred_path)
    if not preds:
        raise ValueError(f"No predictions found in {pred_path}")

    pair_abs_errors = []
    pair_signed_errors = []
    human_lengths = []
    pred_lengths = []
    missing_gt = 0
    tp_hit_flags = []
    tp_hit_steps = []

    for pred in preds:
        key = prediction_key(pred)
        gt_items = human_groups.get(key, [])
        pred_len = prediction_length(pred)
        pred_lengths.append(pred_len)

        if pred["condition"] == "present":
            bbox = bbox_annos.get(f"{pred['task']}_{pred['name']}")
            if bbox is not None:
                hit_step = target_hit_step(pred["X"], pred["Y"], bbox)
                tp_hit_flags.append(hit_step is not None)
                if hit_step is not None:
                    tp_hit_steps.append(hit_step)

        if not gt_items:
            missing_gt += 1
            continue

        for gt in gt_items:
            gt_len = gt["length"]
            human_lengths.append(gt_len)
            signed = pred_len - gt_len
            pair_signed_errors.append(signed)
            pair_abs_errors.append(abs(signed))

    condition = preds[0]["condition"]
    label = pred_path.parent.name
    pair_abs_errors = np.asarray(pair_abs_errors, dtype=float)
    pair_signed_errors = np.asarray(pair_signed_errors, dtype=float)
    human_lengths = np.asarray(human_lengths, dtype=float)
    pred_lengths = np.asarray(pred_lengths, dtype=float)

    summary = {
        "label": label,
        "prediction_file": str(pred_path),
        "condition": condition,
        "num_predictions": int(len(preds)),
        "missing_gt_predictions": int(missing_gt),
        "human_avg_len": safe_mean(human_lengths),
        "human_std_len": safe_std(human_lengths),
        "pred_avg_len": safe_mean(pred_lengths),
        "pred_std_len": safe_std(pred_lengths),
        "len_mae": safe_mean(pair_abs_errors),
        "len_err_std": safe_std(pair_abs_errors),
        "signed_len_error": safe_mean(pair_signed_errors),
        "over_search_rate": safe_mean(pair_signed_errors > 0),
        "under_search_rate": safe_mean(pair_signed_errors < 0),
        "exact_len_rate": safe_mean(pair_signed_errors == 0),
        "tp_target_hit_rate": safe_mean(tp_hit_flags),
        "tp_target_hit_step_mean": safe_mean(tp_hit_steps),
    }
    return summary, {
        "human_lengths": human_lengths.astype(int).tolist(),
        "pred_lengths": pred_lengths.astype(int).tolist(),
    }


def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def safe_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return float(np.std(arr))


def find_prediction_files(pred_root, patterns):
    root = Path(pred_root)
    files = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(p for p in files if p.is_file()))


def write_summary_csv(summaries, output_path):
    fieldnames = [
        "label", "condition", "num_predictions", "missing_gt_predictions",
        "human_avg_len", "human_std_len", "pred_avg_len", "pred_std_len",
        "len_mae", "len_err_std", "signed_len_error",
        "over_search_rate", "under_search_rate", "exact_len_rate",
        "tp_target_hit_rate", "tp_target_hit_step_mean",
        "prediction_file",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def plot_length_distribution(label, condition, lengths, output_dir, max_plot_len):
    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    human_lengths = np.asarray(lengths["human_lengths"], dtype=int)
    pred_lengths = np.asarray(lengths["pred_lengths"], dtype=int)
    if human_lengths.size == 0 or pred_lengths.size == 0:
        return None

    max_len = max(int(human_lengths.max()), int(pred_lengths.max()))
    if max_plot_len is not None:
        max_len = min(max_len, max_plot_len)
    bins = np.arange(1, max_len + 2)

    human_hist = np.array([(human_lengths == i).mean() for i in bins[:-1]])
    pred_hist = np.array([(pred_lengths == i).mean() for i in bins[:-1]])
    x = np.arange(len(bins) - 1)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    width = 0.38
    ax.bar(x - width / 2, human_hist, width=width, label="Human", color="#4C78A8")
    ax.bar(x + width / 2, pred_hist, width=width, label="GLDAS-Gaze", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in bins[:-1]])
    ax.set_xlabel("Scanpath length")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{label} ({condition})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    safe_label = label.replace("/", "_").replace("\\", "_").replace(":", "_")
    output_path = output_dir / f"length_distribution_{safe_label}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    human_groups = load_human_scanpaths(args.dataset_root)
    bbox_annos = load_bbox_annos(args.dataset_root)
    pred_files = find_prediction_files(args.pred_root, args.patterns)
    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files found under {args.pred_root} with {args.patterns}"
        )

    summaries = []
    distributions = {}
    used_labels = set()
    figures = []
    for pred_path in pred_files:
        summary, lengths = summarize_prediction_file(
            pred_path, human_groups, bbox_annos
        )
        if summary["label"] in used_labels:
            summary["label"] = f"{summary['label']}__{pred_path.stem}"
        used_labels.add(summary["label"])
        summaries.append(summary)
        distributions[summary["label"]] = lengths
        figure = plot_length_distribution(
            summary["label"], summary["condition"], lengths,
            output_dir, args.max_plot_len
        )
        if figure is not None:
            figures.append(figure)

    write_summary_csv(summaries, output_dir / "stopping_summary.csv")
    with open(output_dir / "stopping_metrics.json", "w") as f:
        json.dump({
            "summaries": summaries,
            "length_distributions": distributions,
            "figures": figures,
        }, f, indent=2)

    print(f"Analyzed {len(summaries)} prediction files.")
    print(f"Saved CSV to: {output_dir / 'stopping_summary.csv'}")
    print(f"Saved JSON to: {output_dir / 'stopping_metrics.json'}")
    for row in summaries:
        print(
            f"{row['label']}: condition={row['condition']} "
            f"human_avg={fmt(row['human_avg_len'])} "
            f"pred_avg={fmt(row['pred_avg_len'])} "
            f"len_mae={fmt(row['len_mae'])} "
            f"signed={fmt(row['signed_len_error'])} "
            f"over={fmt(row['over_search_rate'])} "
            f"under={fmt(row['under_search_rate'])} "
            f"hit={fmt(row['tp_target_hit_rate'])}"
        )


def fmt(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
