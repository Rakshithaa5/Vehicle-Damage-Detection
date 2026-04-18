"""
Test Faster R-CNN current best.pt
Evaluates on validation set and shows mAP, precision, recall
Run: python test_faster_rcnn_map.py
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np

CLASS_NAMES = [
    "background","dent","scratch","crack",
    "shattered_glass","bumper_damage","deformation"
]
IMGSZ      = 512
NUM_CLASSES = 7
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESH = 0.25
IOU_THRESH  = 0.45

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
                x1 = max(0, (cx - w/2) * IMGSZ)
                y1 = max(0, (cy - h/2) * IMGSZ)
                x2 = min(IMGSZ, (cx + w/2) * IMGSZ)
                y2 = min(IMGSZ, (cy + h/2) * IMGSZ)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),  dtype=torch.int64),
        }
        return img_t, target, img_path.name

def collate_fn(batch):
    return tuple(zip(*batch))

# ── IoU ────────────────────────────────────────────────────
def box_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1    = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2    = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)

# ── Evaluate ───────────────────────────────────────────────
def evaluate(model, dataloader):
    model.eval()
    all_tp = {i: [] for i in range(1, NUM_CLASSES)}
    all_fp = {i: [] for i in range(1, NUM_CLASSES)}
    all_gt = {i: 0  for i in range(1, NUM_CLASSES)}

    with torch.no_grad():
        for imgs, targets, names in dataloader:
            imgs = [img.to(DEVICE) for img in imgs]
            preds = model(imgs)

            for pred, target in zip(preds, targets):
                gt_boxes  = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()

                for cls in range(1, NUM_CLASSES):
                    gt_mask  = gt_labels == cls
                    all_gt[cls] += gt_mask.sum()

                pred_boxes  = pred["boxes"].cpu().numpy()
                pred_labels = pred["labels"].cpu().numpy()
                pred_scores = pred["scores"].cpu().numpy()

                # Filter by confidence
                mask        = pred_scores >= CONF_THRESH
                pred_boxes  = pred_boxes[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]

                # Sort by confidence
                order       = np.argsort(-pred_scores)
                pred_boxes  = pred_boxes[order]
                pred_labels = pred_labels[order]

                gt_matched = set()
                for pb, pl in zip(pred_boxes, pred_labels):
                    if pl not in all_tp: continue
                    best_iou = 0
                    best_idx = -1
                    for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if gl == pl and gi not in gt_matched:
                            iou = box_iou(pb, gb)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = gi
                    if best_iou >= IOU_THRESH and best_idx >= 0:
                        all_tp[pl].append(1)
                        all_fp[pl].append(0)
                        gt_matched.add(best_idx)
                    else:
                        all_tp[pl].append(0)
                        all_fp[pl].append(1)

    # Compute per-class AP
    aps = []
    print("\n  Per-class results:")
    print(f"  {'Class':<20} {'AP':>8} {'TP':>6} {'FP':>6} {'GT':>6}")
    print("  " + "-"*50)

    for cls in range(1, NUM_CLASSES):
        tp  = np.array(all_tp[cls])
        fp  = np.array(all_fp[cls])
        n   = all_gt[cls]
        if n == 0:
            continue
        tp_cum  = np.cumsum(tp)
        fp_cum  = np.cumsum(fp)
        rec     = tp_cum / (n + 1e-6)
        prec    = tp_cum / (tp_cum + fp_cum + 1e-6)

        # AP via trapezoid
        ap = np.trapz(prec, rec) if len(prec) > 1 else 0.0
        aps.append(ap)

        name = CLASS_NAMES[cls]
        print(f"  {name:<20} {ap:>8.4f} {int(tp.sum()):>6} {int(fp.sum()):>6} {int(n):>6}")

    mAP = np.mean(aps) if aps else 0.0
    total_tp = sum(sum(v) for v in all_tp.values())
    total_fp = sum(sum(v) for v in all_fp.values())
    total_gt = sum(all_gt.values())
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall    = total_tp / (total_gt + 1e-6)

    return mAP, precision, recall


def main():
    print("=" * 50)
    print("  Faster R-CNN — Evaluation")
    print("=" * 50)

    # Find model
    model_path = Path("runs/faster_rcnn/best.pt")
    if not model_path.exists():
        print("ERROR: runs/faster_rcnn/best.pt not found!")
        return

    print(f"  Model : {model_path}")
    print(f"  Device: {DEVICE}")

    # Build model
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Use val set
    val_ds = CarDDDataset("dataset/images/val", "dataset/labels/val")
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    print(f"  Val images: {len(val_ds)}")

    print("\n  Evaluating... (this takes 2-3 mins)")
    mAP, precision, recall = evaluate(model, val_dl)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n" + "=" * 50)
    print("  FASTER R-CNN CURRENT RESULTS")
    print("=" * 50)
    print(f"  mAP@0.5     : {mAP:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print("=" * 50)
    print("\n  Compare with YOLOv8:")
    print(f"  YOLOv8  mAP50 : 0.7448")
    print(f"  FasterRCNN mAP: {mAP:.4f}  {'✅ Better!' if mAP > 0.7448 else '⏳ Still training...'}")
    print("=" * 50)


if __name__ == "__main__":
    main()