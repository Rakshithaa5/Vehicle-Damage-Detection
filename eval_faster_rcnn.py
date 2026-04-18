"""
Faster R-CNN — Complete Evaluation Script
==========================================
Generates metrics + plots on TEST set:
  - mAP, Precision, Recall, F1
  - Confusion matrix
  - PR curve
  - F1 curve
  - Per-class AP

Run: python eval_faster_rcnn.py
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json, csv
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── Config ─────────────────────────────────────────────────
CLASS_NAMES  = ["dent","scratch","crack","shattered_glass","bumper_damage","deformation"]
NUM_CLASSES  = 7       # 6 + background
IMGSZ        = 512
CONF_THRESH  = 0.25
IOU_THRESH   = 0.50
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR      = Path("runs/faster_rcnn_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────
class CarDDDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.imgs    = sorted(
            list(self.img_dir.glob("*.jpg")) +
            list(self.img_dir.glob("*.png"))
        )

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        img      = Image.open(img_path).convert("RGB")
        img      = img.resize((IMGSZ, IMGSZ))
        img_t    = torchvision.transforms.functional.to_tensor(img)

        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if not line.strip(): continue
                parts = line.strip().split()
                if len(parts) != 5: continue
                cls, cx, cy, w, h = map(float, parts)
                x1 = max(0, (cx-w/2)*IMGSZ)
                y1 = max(0, (cy-h/2)*IMGSZ)
                x2 = min(IMGSZ, (cx+w/2)*IMGSZ)
                y2 = min(IMGSZ, (cy+h/2)*IMGSZ)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1,y1,x2,y2])
                    labels.append(int(cls)+1)

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),  dtype=torch.int64),
        }
        return img_t, target

def collate_fn(batch): return tuple(zip(*batch))

# ── IoU ────────────────────────────────────────────────────
def box_iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)

# ── Run inference ──────────────────────────────────────────
def run_inference(model, dataloader):
    model.eval()
    all_preds   = []   # list of {boxes, labels, scores}
    all_targets = []   # list of {boxes, labels}

    print("  Running inference on test set...")
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(dataloader):
            imgs  = [img.to(DEVICE) for img in imgs]
            preds = model(imgs)
            for pred, target in zip(preds, targets):
                all_preds.append({
                    "boxes":  pred["boxes"].cpu().numpy(),
                    "labels": pred["labels"].cpu().numpy(),
                    "scores": pred["scores"].cpu().numpy(),
                })
                all_targets.append({
                    "boxes":  target["boxes"].numpy(),
                    "labels": target["labels"].numpy(),
                })
            if (i+1) % 10 == 0:
                print(f"    Batch {i+1}/{len(dataloader)}")

    return all_preds, all_targets

# ── Compute AP per class ───────────────────────────────────
def compute_ap(preds, targets, cls_id, conf_thresh, iou_thresh):
    # Collect all detections for this class sorted by confidence
    det_scores, det_tp, det_fp = [], [], []
    n_gt = 0

    for pred, target in zip(preds, targets):
        gt_mask  = target["labels"] == cls_id
        gt_boxes = target["boxes"][gt_mask]
        n_gt    += len(gt_boxes)

        pred_mask   = (pred["labels"] == cls_id) & (pred["scores"] >= conf_thresh)
        pred_boxes  = pred["boxes"][pred_mask]
        pred_scores = pred["scores"][pred_mask]

        order       = np.argsort(-pred_scores)
        pred_boxes  = pred_boxes[order]
        pred_scores = pred_scores[order]

        matched = set()
        for pb, ps in zip(pred_boxes, pred_scores):
            det_scores.append(ps)
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi not in matched:
                    iou = box_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
            if best_iou >= iou_thresh and best_gi >= 0:
                det_tp.append(1); det_fp.append(0)
                matched.add(best_gi)
            else:
                det_tp.append(0); det_fp.append(1)

    if n_gt == 0: return 0.0, [], [], []

    order    = np.argsort(-np.array(det_scores))
    tp_arr   = np.array(det_tp)[order]
    fp_arr   = np.array(det_fp)[order]
    tp_cum   = np.cumsum(tp_arr)
    fp_cum   = np.cumsum(fp_arr)
    rec      = tp_cum / (n_gt + 1e-6)
    prec     = tp_cum / (tp_cum + fp_cum + 1e-6)

    # Add sentinel
    rec  = np.concatenate([[0], rec,  [1]])
    prec = np.concatenate([[1], prec, [0]])

    # Smooth precision
    for i in range(len(prec)-2, -1, -1):
        prec[i] = max(prec[i], prec[i+1])

    ap = np.trapz(prec, rec)
    return ap, rec, prec, det_scores

# ── Confusion matrix ───────────────────────────────────────
def compute_confusion(preds, targets, iou_thresh):
    n = len(CLASS_NAMES) + 1   # +1 for background
    cm = np.zeros((n, n), dtype=int)

    for pred, target in zip(preds, targets):
        pred_mask   = pred["scores"] >= CONF_THRESH
        pred_boxes  = pred["boxes"][pred_mask]
        pred_labels = pred["labels"][pred_mask]
        pred_scores = pred["scores"][pred_mask]
        gt_boxes    = target["boxes"]
        gt_labels   = target["labels"]

        matched_gt  = set()
        matched_pred= set()

        for pi, (pb, pl, ps) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            best_iou, best_gi = 0, -1
            for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi

            pred_cls = int(pl) - 1   # convert to 0-indexed
            if best_iou >= iou_thresh and best_gi not in matched_gt:
                gt_cls = int(gt_labels[best_gi]) - 1
                cm[gt_cls][pred_cls] += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                if 0 <= pred_cls < len(CLASS_NAMES):
                    cm[len(CLASS_NAMES)][pred_cls] += 1  # FP

        for gi, gl in enumerate(gt_labels):
            if gi not in matched_gt:
                gt_cls = int(gl) - 1
                if 0 <= gt_cls < len(CLASS_NAMES):
                    cm[gt_cls][len(CLASS_NAMES)] += 1    # FN

    return cm

# ── Plot confusion matrix ──────────────────────────────────
def plot_confusion(cm, normalized=False):
    labels = CLASS_NAMES + ["background"]
    if normalized:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = np.divide(cm.astype(float), row_sums,
                             out=np.zeros_like(cm, dtype=float),
                             where=row_sums!=0)
        title = "Faster R-CNN — Confusion matrix (normalized)"
        fmt   = ".2f"
        vmax  = 1.0
    else:
        cm_plot = cm.astype(float)
        title   = "Faster R-CNN — Confusion matrix"
        fmt     = "d"
        vmax    = cm.max()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_plot, interpolation="nearest",
                   cmap=plt.cm.Blues, vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)

    thresh = cm_plot.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = f"{int(cm_plot[i,j])}" if fmt=="d" else f"{cm_plot[i,j]:.2f}"
            ax.text(j, i, val, ha="center", va="center", fontsize=8,
                    color="white" if cm_plot[i,j] > thresh else "black")

    plt.tight_layout()
    suffix = "_normalized" if normalized else ""
    path   = OUT_DIR / f"confusion_matrix{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

# ── Plot PR curve ──────────────────────────────────────────
def plot_pr_curve(all_rec, all_prec, all_ap):
    colors = ["#185FA5","#534AB7","#0F6E56","#993C1D","#854F0B","#185FA5"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (cls, rec, prec, ap) in enumerate(zip(CLASS_NAMES, all_rec, all_prec, all_ap)):
        if len(rec) > 0:
            ax.plot(rec, prec, color=colors[i%len(colors)], linewidth=1.5,
                    label=f"{cls} (AP={ap:.3f})")
    mAP = np.mean(all_ap)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Faster R-CNN — Precision-Recall Curve (mAP@0.5={mAP:.4f})")
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "BoxPR_curve.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path.name}")

# ── Plot F1 curve ──────────────────────────────────────────
def plot_f1_curve(preds, targets):
    confs  = np.linspace(0.01, 0.99, 50)
    f1s    = []
    for conf in confs:
        tp=fp=fn=0
        for pred, target in zip(preds, targets):
            mask        = pred["scores"] >= conf
            pred_labels = pred["labels"][mask]
            pred_boxes  = pred["boxes"][mask]
            gt_labels   = target["labels"]
            gt_boxes    = target["boxes"]
            matched     = set()
            for pb, pl in zip(pred_boxes, pred_labels):
                found = False
                for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if gi not in matched and gl==pl and box_iou(pb,gb)>=IOU_THRESH:
                        tp+=1; matched.add(gi); found=True; break
                if not found: fp+=1
            fn += len(gt_labels) - len(matched)
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        f1s.append(2*prec*rec/(prec+rec+1e-6))

    best_conf = confs[np.argmax(f1s)]
    best_f1   = max(f1s)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(confs, f1s, color="#185FA5", linewidth=2)
    ax.axvline(best_conf, color="red", linestyle="--", linewidth=1,
               label=f"Best F1={best_f1:.3f} @ conf={best_conf:.2f}")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("Faster R-CNN — F1 Curve")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "BoxF1_curve.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path.name}")

# ── Per-class bar chart ────────────────────────────────────
def plot_per_class_ap(all_ap):
    colors = ["#185FA5","#534AB7","#0F6E56","#993C1D","#854F0B","#D85A30"]
    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(CLASS_NAMES, all_ap, color=colors, alpha=0.85)
    ax.set_xlabel("Damage class"); ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Faster R-CNN — Per-class AP on test set")
    ax.set_ylim(0,1); ax.grid(axis="y", alpha=0.3)
    for bar, ap in zip(bars, all_ap):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{ap:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = OUT_DIR / "per_class_ap.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path.name}")

# ── Main ───────────────────────────────────────────────────
def main():
    print("="*55)
    print("  Faster R-CNN — Full Evaluation on Test Set")
    print("="*55)
    print(f"  Device : {DEVICE}")
    print(f"  Output : {OUT_DIR.resolve()}")

    # Load model
    model_path = Path("runs/faster_rcnn/best.pt")
    if not model_path.exists():
        print(f"\n  ERROR: {model_path} not found!")
        return

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"  Model loaded: {model_path}")

    # Load test dataset
    test_ds = CarDDDataset("dataset/images/test", "dataset/labels/test")
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)
    print(f"  Test images: {len(test_ds)}\n")

    # Run inference
    preds, targets = run_inference(model, test_dl)

    # Compute per-class AP
    print("\n  Computing metrics...")
    all_ap, all_rec, all_prec = [], [], []
    tp_total=fp_total=fn_total=gt_total=0

    print(f"\n  {'Class':<20} {'AP':>8} {'TP':>6} {'FP':>6} {'GT':>6}")
    print("  " + "-"*50)

    for cls_id in range(1, NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_id-1]
        ap, rec, prec, scores = compute_ap(preds, targets, cls_id, CONF_THRESH, IOU_THRESH)
        all_ap.append(ap); all_rec.append(rec); all_prec.append(prec)

        # Count TP/FP/GT for this class
        tp=fp=gt=0
        for pred, target in zip(preds, targets):
            gt_mask = target["labels"] == cls_id
            gt      += gt_mask.sum()
            pred_mask   = (pred["labels"]==cls_id) & (pred["scores"]>=CONF_THRESH)
            pred_boxes  = pred["boxes"][pred_mask]
            gt_boxes    = target["boxes"][gt_mask]
            matched     = set()
            for pb in pred_boxes:
                found = False
                for gi, gb in enumerate(gt_boxes):
                    if gi not in matched and box_iou(pb,gb)>=IOU_THRESH:
                        tp+=1; matched.add(gi); found=True; break
                if not found: fp+=1
        fn = gt - tp
        tp_total+=tp; fp_total+=fp; fn_total+=fn; gt_total+=gt
        print(f"  {cls_name:<20} {ap:>8.4f} {tp:>6} {fp:>6} {gt:>6}")

    mAP       = np.mean(all_ap)
    precision = tp_total/(tp_total+fp_total+1e-6)
    recall    = tp_total/(gt_total+1e-6)
    f1        = 2*precision*recall/(precision+recall+1e-6)

    print("\n" + "="*55)
    print("  FASTER R-CNN — TEST SET RESULTS")
    print("="*55)
    print(f"  mAP@0.5     : {mAP:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print("="*55)

    # Compare with YOLOv8
    print(f"\n  Compare with YOLOv8:")
    print(f"  YOLOv8  mAP50 : 0.7448")
    print(f"  FasterRCNN    : {mAP:.4f}  {'Better!' if mAP > 0.7448 else 'YOLOv8 leads'}")

    # Generate all plots
    print("\n  Generating plots...")
    cm = compute_confusion(preds, targets, IOU_THRESH)
    plot_confusion(cm, normalized=False)
    plot_confusion(cm, normalized=True)
    plot_pr_curve(all_rec, all_prec, all_ap)
    plot_f1_curve(preds, targets)
    plot_per_class_ap(all_ap)

    # Save CSV
    row = {
        "model":      "Faster R-CNN",
        "mAP50":      round(mAP,4),
        "precision":  round(precision,4),
        "recall":     round(recall,4),
        "f1":         round(f1,4),
    }
    with open(OUT_DIR/"metrics.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader(); w.writerow(row)
    print(f"  Saved: metrics.csv")

    print(f"\n  All outputs in: {OUT_DIR.resolve()}")
    print("="*55)


if __name__ == "__main__":
    main()
