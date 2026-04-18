"""
EfficientDet-D2 — Periodic mAP Checker
=======================================
Run this in a SEPARATE terminal while training, or pause training every 5 epochs and run it.

Usage:
  python check_map.py                      # checks latest checkpoint
  python check_map.py --epoch 10           # checks epoch_010.pt specifically
  python check_map.py --watch              # auto-runs every time a new checkpoint appears

Install: pip install effdet timm
"""

import torch, argparse, time
import torchvision.transforms.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ── Config (must match training) ────────────────────────────
CONFIG = {
    "num_classes": 6,
    "imgsz":       512,
    "batch_size":  2,        # lower than training to save VRAM during check
    "num_workers": 0,
    "conf_thresh": 0.25,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
}

CLASS_NAMES = [
    "dent", "scratch", "crack",
    "shattered_glass", "bumper_damage", "deformation"
]

VAL_IMG_DIR = "dataset/images/val"
VAL_LBL_DIR = "dataset/labels/val"
CKPT_DIR    = Path("runs/efficientdet")
LOG_PATH    = CKPT_DIR / "map_log.txt"

try:
    from effdet import create_model, DetBenchPredict
    EFFDET_OK = True
except ImportError:
    EFFDET_OK = False


# ── Dataset ─────────────────────────────────────────────────
class ValDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, imgsz=512):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.imgsz   = imgsz
        self.imgs    = sorted(
            list(self.img_dir.glob("*.jpg")) +
            list(self.img_dir.glob("*.png"))
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.imgsz, self.imgsz))
        img_t = F.to_tensor(img)

        gt_boxes, gt_labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = map(float, parts)
                x1 = max(0.0, (cx - w/2) * self.imgsz)
                y1 = max(0.0, (cy - h/2) * self.imgsz)
                x2 = min(float(self.imgsz), (cx + w/2) * self.imgsz)
                y2 = min(float(self.imgsz), (cy + h/2) * self.imgsz)
                if x2 > x1 and y2 > y1:
                    gt_boxes.append([x1, y1, x2, y2])   # xyxy
                    gt_labels.append(int(cls) + 1)       # 1-indexed

        return img_t, gt_boxes, gt_labels


def collate_fn(batch):
    imgs      = torch.stack([b[0] for b in batch])
    gt_boxes  = [b[1] for b in batch]
    gt_labels = [b[2] for b in batch]
    return imgs, gt_boxes, gt_labels


# ── IoU ──────────────────────────────────────────────────────
def box_iou(box, boxes):
    """box: [4], boxes: [N,4] xyxy → IoU [N]"""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    a1    = (box[2]-box[0]) * (box[3]-box[1])
    a2    = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / (a1 + a2 - inter + 1e-6)


def compute_ap(recalls, precisions):
    r = np.concatenate([[0.], recalls, [1.]])
    p = np.concatenate([[1.], precisions, [0.]])
    for i in range(len(p)-2, -1, -1):
        p[i] = max(p[i], p[i+1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx+1] - r[idx]) * p[idx+1]))


def compute_map(all_preds, all_gts, iou_thresh, num_classes):
    per_class = {}
    for cls in range(1, num_classes+1):
        tp_list, fp_list, sc_list = [], [], []
        n_gt = 0
        for preds, gts in zip(all_preds, all_gts):
            gt_idx   = [i for i,l in enumerate(gts["labels"])  if l == cls]
            pred_idx = [i for i,l in enumerate(preds["labels"]) if l == cls]
            gt_b  = np.array([gts["boxes"][i]   for i in gt_idx],   dtype=np.float32).reshape(-1,4)
            pb    = np.array([preds["boxes"][i]  for i in pred_idx], dtype=np.float32).reshape(-1,4)
            sc    = np.array([preds["scores"][i] for i in pred_idx], dtype=np.float32)
            n_gt += len(gt_b)
            matched = np.zeros(len(gt_b), dtype=bool)
            order = np.argsort(-sc)
            pb, sc = pb[order], sc[order]
            for b, s in zip(pb, sc):
                sc_list.append(s)
                if len(gt_b) == 0:
                    tp_list.append(0); fp_list.append(1); continue
                ious = box_iou(b, gt_b)
                best = int(np.argmax(ious))
                if ious[best] >= iou_thresh and not matched[best]:
                    tp_list.append(1); fp_list.append(0)
                    matched[best] = True
                else:
                    tp_list.append(0); fp_list.append(1)
        if n_gt == 0:
            per_class[cls] = 0.0; continue
        order  = np.argsort(-np.array(sc_list))
        tp_cum = np.cumsum(np.array(tp_list)[order])
        fp_cum = np.cumsum(np.array(fp_list)[order])
        rec    = tp_cum / (n_gt + 1e-6)
        prec   = tp_cum / (tp_cum + fp_cum + 1e-6)
        per_class[cls] = compute_ap(rec, prec)
    return float(np.mean(list(per_class.values()))), per_class


# ── Load model ───────────────────────────────────────────────
def load_model(ckpt_path, num_classes, imgsz, device):
    base = create_model(
        "tf_efficientdet_d2",
        bench_task      = "",
        num_classes     = num_classes,
        image_size      = (imgsz, imgsz),
        pretrained      = False,
        checkpoint_path = "",
    )
    bench = DetBenchPredict(base)
    state = torch.load(ckpt_path, map_location="cpu")
    # strip 'model.' prefix if saved from DetBenchTrain
    state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
             for k, v in state.items()}
    bench.load_state_dict(state, strict=False)
    return bench.to(device).eval()


# ── Run inference + collect preds/gts ───────────────────────
def run_inference(model, dataloader, device, conf_thresh, imgsz):
    all_preds, all_gts = [], []

    with torch.no_grad():
        for imgs, gt_boxes_batch, gt_labels_batch in dataloader:
            imgs = imgs.to(device)

            try:
                dets_batch = model(imgs)   # [B, max_det, 6]
            except Exception as e:
                print(f"  [WARN] Inference error: {e}")
                continue

            for i in range(imgs.shape[0]):
                dets = dets_batch[i].cpu().numpy()
                mask = dets[:, 4] >= conf_thresh
                dets = dets[mask]

                all_preds.append({
                    "boxes":  dets[:, :4].tolist()            if len(dets) else [],
                    "scores": dets[:,  4].tolist()            if len(dets) else [],
                    "labels": dets[:,  5].astype(int).tolist() if len(dets) else [],
                })
                all_gts.append({
                    "boxes":  gt_boxes_batch[i],
                    "labels": gt_labels_batch[i],
                })

    return all_preds, all_gts


# ── Main check ───────────────────────────────────────────────
def run_check(ckpt_path, epoch_label, dataset, dataloader, device):
    print(f"\n{'='*58}")
    print(f"  mAP Check — {epoch_label}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*58}")

    model = load_model(ckpt_path, CONFIG["num_classes"], CONFIG["imgsz"], device)

    all_preds, all_gts = run_inference(
        model, dataloader, device, CONFIG["conf_thresh"], CONFIG["imgsz"]
    )

    map50,   per50   = compute_map(all_preds, all_gts, 0.50, CONFIG["num_classes"])
    map75,   per75   = compute_map(all_preds, all_gts, 0.75, CONFIG["num_classes"])

    print(f"\n  mAP@0.50   : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP@0.75   : {map75:.4f}  ({map75*100:.1f}%)")
    print()
    print(f"  {'Class':<20} {'AP@0.50':>9} {'AP@0.75':>9}")
    print("  " + "-"*40)
    for cls in range(1, CONFIG["num_classes"]+1):
        name = CLASS_NAMES[cls-1]
        print(f"  {name:<20} {per50.get(cls,0):>9.4f} {per75.get(cls,0):>9.4f}")

    # Overfitting hint
    print()
    if map50 < 0.20:
        hint = "⚠️  Very low — model still learning or conf_thresh too high"
    elif map50 < 0.45:
        hint = "🔄 Developing — keep training"
    elif map50 < 0.65:
        hint = "👍 Decent — monitor for plateau"
    else:
        hint = "✅ Strong result"
    print(f"  Verdict    : {hint}")

    # Log to file
    with open(LOG_PATH, "a") as f:
        f.write(f"{epoch_label} | mAP@0.50={map50:.4f} | mAP@0.75={map75:.4f}\n")
        for cls in range(1, CONFIG["num_classes"]+1):
            f.write(f"  {CLASS_NAMES[cls-1]}: AP50={per50.get(cls,0):.4f}  AP75={per75.get(cls,0):.4f}\n")
        f.write("\n")

    print(f"\n  Log appended → {LOG_PATH}")
    del model
    torch.cuda.empty_cache()
    return map50


# ── Entry point ──────────────────────────────────────────────
def main():
    if not EFFDET_OK:
        print("[ERROR] effdet not installed. Run: pip install effdet timm")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=None,
                        help="Check a specific epoch checkpoint (e.g. --epoch 10)")
    parser.add_argument("--watch", action="store_true",
                        help="Auto-watch for new checkpoints every 5 min")
    parser.add_argument("--all", action="store_true",
                        help="Run mAP on ALL epoch_*.pt checkpoints found")
    args = parser.parse_args()

    device  = torch.device(CONFIG["device"])
    dataset = ValDataset(VAL_IMG_DIR, VAL_LBL_DIR, CONFIG["imgsz"])
    dl      = DataLoader(dataset, batch_size=CONFIG["batch_size"],
                         shuffle=False, collate_fn=collate_fn,
                         num_workers=CONFIG["num_workers"])
    print(f"  Val images : {len(dataset)}")

    # ── Mode 1: specific epoch ───────────────────────────────
    if args.epoch is not None:
        ckpt = CKPT_DIR / f"epoch_{args.epoch:03d}.pt"
        if not ckpt.exists():
            print(f"[ERROR] Not found: {ckpt}")
            return
        run_check(str(ckpt), f"epoch_{args.epoch:03d}", dataset, dl, device)

    # ── Mode 2: all checkpoints ──────────────────────────────
    elif args.all:
        checkpoints = sorted(CKPT_DIR.glob("epoch_*.pt"))
        if not checkpoints:
            print("[ERROR] No epoch_*.pt checkpoints found in", CKPT_DIR)
            return
        print(f"  Found {len(checkpoints)} checkpoints\n")
        results = []
        for ckpt in checkpoints:
            ep    = int(ckpt.stem.split("_")[1])
            map50 = run_check(str(ckpt), ckpt.stem, dataset, dl, device)
            results.append((ep, map50))
        print("\n" + "="*58)
        print("  Summary — all checkpoints")
        print("  " + "-"*30)
        for ep, m in results:
            bar = "█" * int(m * 40)
            print(f"  epoch {ep:3d} | {m:.4f} | {bar}")
        best_ep, best_m = max(results, key=lambda x: x[1])
        print(f"\n  Best: epoch {best_ep} → mAP@0.50 = {best_m:.4f}")

    # ── Mode 3: watch mode ───────────────────────────────────
    elif args.watch:
        print("  Watch mode — checking every 5 min for new checkpoints")
        print("  Press Ctrl+C to stop\n")
        checked = set()
        while True:
            checkpoints = sorted(CKPT_DIR.glob("epoch_*.pt"))
            new = [c for c in checkpoints if c.name not in checked]
            for ckpt in new:
                run_check(str(ckpt), ckpt.stem, dataset, dl, device)
                checked.add(ckpt.name)
            if new:
                print(f"\n  Waiting for next checkpoint... (Ctrl+C to stop)")
            time.sleep(300)   # check every 5 min

    # ── Mode 4: default — check best.pt ─────────────────────
    else:
        ckpt = CKPT_DIR / "best.pt"
        if not ckpt.exists():
            # fallback: latest epoch checkpoint
            checkpoints = sorted(CKPT_DIR.glob("epoch_*.pt"))
            if not checkpoints:
                print("[ERROR] No checkpoints found in", CKPT_DIR)
                print("        Run with --epoch N or --all")
                return
            ckpt = checkpoints[-1]
        run_check(str(ckpt), ckpt.stem, dataset, dl, device)


if __name__ == "__main__":
    main()