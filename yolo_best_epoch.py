"""
YOLOv8 — Best Epoch Finder + Metric Plots
Run: python yolo_best_epoch.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH  = "runs/detect/runs/yolov8m/results.csv"
PLOT_PATH = "runs/detect/runs/yolov8m/best_epoch_plot.png"

# ── Load ────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
print(f"Loaded {len(df)} epochs from {CSV_PATH}\n")

# ── Best epoch per metric ────────────────────────────────────
metrics = {
    "mAP@0.50":    "metrics/mAP50(B)",
    "mAP@0.50:95": "metrics/mAP50-95(B)",
    "Precision":   "metrics/precision(B)",
    "Recall":      "metrics/recall(B)",
}

print("Best epoch per metric:")
print("-" * 45)
for name, col in metrics.items():
    best_idx = df[col].idxmax()
    epoch    = int(df.loc[best_idx, "epoch"])
    value    = df.loc[best_idx, col]
    print(f"  {name:<14}: epoch {epoch:3d}  →  {value:.4f}")

# Overall best = highest mAP@0.50
best_map_idx = df["metrics/mAP50(B)"].idxmax()
best_row     = df.loc[best_map_idx]
print()
print("=" * 45)
print(f"  Best overall epoch : {int(best_row['epoch'])}")
print(f"  mAP@0.50           : {best_row['metrics/mAP50(B)']:.4f}")
print(f"  mAP@0.50:95        : {best_row['metrics/mAP50-95(B)']:.4f}")
print(f"  Precision          : {best_row['metrics/precision(B)']:.4f}")
print(f"  Recall             : {best_row['metrics/recall(B)']:.4f}")
print(f"  Train box loss     : {best_row['train/box_loss']:.4f}")
print(f"  Val box loss       : {best_row['val/box_loss']:.4f}")
print("=" * 45)

# ── Plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (name, col) in zip(axes.flatten(), metrics.items()):
    best_idx  = df[col].idxmax()
    best_ep   = df.loc[best_idx, "epoch"]
    best_val  = df.loc[best_idx, col]

    ax.plot(df["epoch"], df[col], "b-", linewidth=1.5, label=name)
    ax.axvline(best_ep, color="red", linestyle="--", alpha=0.6)
    ax.scatter(best_ep, best_val, color="red", zorder=5,
               label=f"Best @ ep{int(best_ep)}: {best_val:.4f}")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("YOLOv8 — Best Epoch Per Metric", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.close()
print(f"\nPlot saved → {PLOT_PATH}")