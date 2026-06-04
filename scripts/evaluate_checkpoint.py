import argparse
import csv
import json
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Classifier
from operation import AudioSpectrogramDataset


def decode_batch(raw, labels):
    pred = torch.sigmoid(raw).detach().cpu()
    target = labels.detach().cpu()

    def decode_angle(sin_value, cos_value):
        sin_value = sin_value * 2.0 - 1.0
        cos_value = cos_value * 2.0 - 1.0
        angle = torch.rad2deg(torch.atan2(sin_value, cos_value))
        return torch.remainder(angle + 360.0, 360.0)

    decoded = {
        "distance_pred": pred[:, 0] * 50.0,
        "height_pred": pred[:, 1] * 50.0,
        "azimuth_pred": decode_angle(pred[:, 2], pred[:, 3]),
        "orientation_pred": decode_angle(pred[:, 4], pred[:, 5]),
        "distance_true": target[:, 0] * 50.0,
        "height_true": target[:, 1] * 50.0,
        "azimuth_true": decode_angle(target[:, 2], target[:, 3]),
        "orientation_true": decode_angle(target[:, 4], target[:, 5]),
    }
    return {k: v.numpy() for k, v in decoded.items()}


def circular_error(pred_deg, true_deg):
    diff = np.abs(pred_deg - true_deg)
    return np.minimum(diff, 360.0 - diff)


def parse_training_log(path):
    rows = []
    current = None
    patterns = {
        "loss": re.compile(r"Loss: ([0-9.]+)"),
        "a": re.compile(r"MAE a: ([0-9.]+).*?([0-9.]+)$"),
        "b": re.compile(r"MAE b: ([0-9.]+).*?([0-9.]+)$"),
        "c": re.compile(r"MAE c: ([0-9.]+).*?([0-9.]+)$"),
        "d": re.compile(r"MAE d: ([0-9.]+).*?([0-9.]+)$"),
    }
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            epoch_match = re.search(r"Epoch ([0-9]+) - Test results", line)
            if epoch_match:
                current = {"epoch": int(epoch_match.group(1))}
                rows.append(current)
                continue
            if current is None:
                continue
            for key, pattern in patterns.items():
                match = pattern.search(line.strip())
                if not match:
                    continue
                if key == "loss":
                    current["test_loss"] = float(match.group(1))
                else:
                    current[f"mae_{key}"] = float(match.group(1))
                    current[f"rmse_{key}"] = float(match.group(2))
    return rows


def write_training_curves(rows, out_csv, out_png):
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [row["test_loss"] for row in rows], marker="o")
    axes[0, 0].set_title("Test loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE sum")

    axes[0, 1].plot(epochs, [row["mae_a"] for row in rows], marker="o", label="distance")
    axes[0, 1].plot(epochs, [row["mae_b"] for row in rows], marker="o", label="height")
    axes[0, 1].set_title("Distance and height MAE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("meters")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["mae_c"] for row in rows], marker="o")
    axes[1, 0].set_title("Azimuth MAE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("degrees")

    axes[1, 1].plot(epochs, [row["mae_d"] for row in rows], marker="o")
    axes[1, 1].set_title("Orientation MAE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("degrees")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_model_diagram(out_png):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        ("8-channel WAV", 0.03),
        ("Per-channel\nMel spectrogram", 0.18),
        ("Resize\n8 x 256 x 256", 0.34),
        ("Residual CNN\nencoder", 0.50),
        ("Global avg pool", 0.66),
        ("FC + GELU + FC", 0.80),
        ("6 outputs\n[a,b,sin/cos az,\nsin/cos orient]", 0.93),
    ]
    for text, x in boxes:
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f2f2f2", "edgecolor": "#333333"},
            fontsize=10,
        )
    for (_, x0), (_, x1) in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(x1 - 0.07, 0.5), xytext=(x0 + 0.07, 0.5), arrowprops={"arrowstyle": "->"})
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--training-log", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--examples", type=int, default=24)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    figures_dir = os.path.join(args.out_dir, "figures")
    tables_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    if args.training_log:
        rows = parse_training_log(args.training_log)
        write_training_curves(
            rows,
            os.path.join(tables_dir, "training_metrics.csv"),
            os.path.join(figures_dir, "training_curves.png"),
        )

    write_model_diagram(os.path.join(figures_dir, "model_architecture.png"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config")
    if cfg is None:
        from config import config as cfg

    dataset = AudioSpectrogramDataset(cfg, mode="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = Classifier(channel_in=8).to(device)
    model.load_state_dict(checkpoint["classifier_state_dict"])
    model.eval()

    all_rows = []
    sample_index = 0
    with torch.no_grad():
        for specs, labels in loader:
            raw = model(specs.to(device).float())
            decoded = decode_batch(raw, labels)
            batch_size = labels.size(0)
            for i in range(batch_size):
                row = {"sample_index": sample_index, "file": dataset.file_paths[sample_index]}
                for key, values in decoded.items():
                    row[key] = float(values[i])
                row["distance_abs_error"] = abs(row["distance_pred"] - row["distance_true"])
                row["height_abs_error"] = abs(row["height_pred"] - row["height_true"])
                row["azimuth_abs_error"] = float(circular_error(row["azimuth_pred"], row["azimuth_true"]))
                row["orientation_abs_error"] = float(circular_error(row["orientation_pred"], row["orientation_true"]))
                all_rows.append(row)
                sample_index += 1

    metrics = {
        "samples": len(all_rows),
        "distance_mae_m": float(np.mean([r["distance_abs_error"] for r in all_rows])),
        "distance_rmse_m": float(math.sqrt(np.mean([r["distance_abs_error"] ** 2 for r in all_rows]))),
        "height_mae_m": float(np.mean([r["height_abs_error"] for r in all_rows])),
        "height_rmse_m": float(math.sqrt(np.mean([r["height_abs_error"] ** 2 for r in all_rows]))),
        "azimuth_mae_deg": float(np.mean([r["azimuth_abs_error"] for r in all_rows])),
        "azimuth_rmse_deg": float(math.sqrt(np.mean([r["azimuth_abs_error"] ** 2 for r in all_rows]))),
        "orientation_mae_deg": float(np.mean([r["orientation_abs_error"] for r in all_rows])),
        "orientation_rmse_deg": float(math.sqrt(np.mean([r["orientation_abs_error"] ** 2 for r in all_rows]))),
    }
    with open(os.path.join(tables_dir, "final_checkpoint_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fieldnames = list(all_rows[0].keys())
    with open(os.path.join(tables_dir, "prediction_examples.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows[: args.examples])

    plot_rows = all_rows[: args.examples]
    x = np.arange(len(plot_rows))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].plot(x, [r["azimuth_true"] for r in plot_rows], marker="o", label="true")
    axes[0].plot(x, [r["azimuth_pred"] for r in plot_rows], marker="x", label="pred")
    axes[0].set_title("Azimuth examples")
    axes[0].set_ylabel("degrees")
    axes[0].legend()
    axes[1].plot(x, [r["distance_true"] for r in plot_rows], marker="o", label="true")
    axes[1].plot(x, [r["distance_pred"] for r in plot_rows], marker="x", label="pred")
    axes[1].set_title("Distance examples")
    axes[1].set_ylabel("meters")
    axes[1].legend()
    axes[2].plot(x, [r["orientation_true"] for r in plot_rows], marker="o", label="true")
    axes[2].plot(x, [r["orientation_pred"] for r in plot_rows], marker="x", label="pred")
    axes[2].set_title("Orientation examples")
    axes[2].set_ylabel("degrees")
    axes[2].set_xlabel("example index")
    axes[2].legend()
    fig.savefig(os.path.join(figures_dir, "prediction_examples.png"), dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
