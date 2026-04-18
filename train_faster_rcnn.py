"""
Model 3 — EfficientDet-D2
Vehicle Damage Detection | CarDD Dataset
===============================================
RTX 5050 8GB — laptop safe settings

Install deps first (once):
  pip install effdet timm

Run: python train_efficientdet.py
"""

import torch, time, csv, json, random, subprocess
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

# ── Config ─────────────────────────────────────────────────
CONFIG = {
    "num_classes":  6,
    "epochs":       100,
    "batch_size":   4,
    "lr":           1e-4,
    "weight_decay": 1e-4,
    "imgsz":        512,
    "num_workers":  0,
    "save_period":  5,
    "patience":     15,
    "temp_limit":   85,
    "temp_pause":   60,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

CLASS_NAMES = [
    "dent", "scratch", "crack",
    "shattered_glass", "bumper_damage", "deformation"
]

# ── Check effdet ────────────────────────────────────────────
try:
    from effdet import create_model, DetBenchTrain
    from effdet.anchors import Anchors, AnchorLabeler
    EFFDET_OK = True
except ImportError:
    EFFDET_OK = False


# ── Dataset ────────────────────────────────────────────────
class CarDDDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, imgsz=512, train=False):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.imgsz   = imgsz
        self.train   = train
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

        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = map(float, parts)
                x1 = max(0.0, (cx - w / 2) * self.imgsz)
                y1 = max(0.0, (cy - h / 2) * self.imgsz)
                x2 = min(float(self.imgsz), (cx + w / 2) * self.imgsz)
                y2 = min(float(self.imgsz), (cy + h / 2) * self.imgsz)
                if x2 > x1 and y2 > y1:
                    # Store as [y1, x1, y2, x2] — effdet format
                    boxes.append([y1, x1, y2, x2])
                    labels.append(int(cls) + 1)   # 1-indexed, 0 = background

        # ── Augmentation (train only) ───────────────────────
        if self.train and boxes:
            if random.random() > 0.5:
                img = F.hflip(img)
                boxes = [
                    [y1, self.imgsz - x2, y2, self.imgsz - x1]
                    for y1, x1, y2, x2 in boxes
                ]
            img = F.adjust_brightness(img, 1.0 + random.uniform(-0.3, 0.3))
            img = F.adjust_contrast(img,   1.0 + random.uniform(-0.3, 0.3))
            img = F.adjust_saturation(img, 1.0 + random.uniform(-0.2, 0.2))

        img_tensor = F.to_tensor(img)

        # Return raw boxes/labels as tensors (variable length — collated below)
        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)   # [N, 4]
            labels_t = torch.tensor(labels, dtype=torch.long)       # [N]
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.long)

        return img_tensor, boxes_t, labels_t


def collate_fn(batch):
    """Pad boxes/labels to the max count in the batch."""
    imgs, all_boxes, all_labels = zip(*batch)
    imgs = torch.stack(imgs)                          # [B, 3, H, W]

    max_n = max(b.shape[0] for b in all_boxes)
    max_n = max(max_n, 1)                             # avoid 0-dim edge case

    B = len(all_boxes)
    boxes_padded  = torch.zeros((B, max_n, 4), dtype=torch.float32)
    labels_padded = torch.zeros((B, max_n),    dtype=torch.long)

    for i, (b, l) in enumerate(zip(all_boxes, all_labels)):
        n = b.shape[0]
        if n > 0:
            boxes_padded[i, :n]  = b
            labels_padded[i, :n] = l

    return imgs, boxes_padded, labels_padded


def build_model(num_classes, imgsz):
    """
    Create the bare EfficientDet backbone (bench_task=''),
    then wrap it in DetBenchTrain which owns the anchor labeler
    and computes label_cls_* / label_box_* targets internally.
    """
    # bench_task='' → returns the raw model without any bench wrapper
    base_model = create_model(
        "tf_efficientdet_d2",
        bench_task      = "",
        num_classes     = num_classes,
        image_size      = (imgsz, imgsz),
        pretrained      = True,
        checkpoint_path = "",
    )
    # DetBenchTrain wraps the model AND handles anchor assignment
    model = DetBenchTrain(base_model)
    return model


def get_gpu_temp():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return int(r.stdout.strip())
    except:
        return None


def find_latest_checkpoint():
    ckpt_dir = Path("runs/efficientdet")
    if not ckpt_dir.exists():
        return None, 0
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        epoch  = int(latest.stem.split("_")[1])
        print(f"  Found checkpoint: {latest.name} (epoch {epoch})")
        return str(latest), epoch
    best = ckpt_dir / "best.pt"
    if best.exists():
        return str(best), 0
    return None, 0


def main():
    print("=" * 60)
    print("  EfficientDet-D2 — Vehicle Damage Detection")
    print("  [Augmentation + Early Stopping + Crash Protection]")
    print("=" * 60)

    if not EFFDET_OK:
        print("\n  [ERROR] effdet not installed!")
        print("  Run: pip install effdet timm")
        return

    device = torch.device(CONFIG["device"])
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Batch size : {CONFIG['batch_size']}")
    print(f"  Image size : {CONFIG['imgsz']}")
    print(f"  Epochs     : {CONFIG['epochs']}")
    print(f"  LR         : {CONFIG['lr']}")
    print(f"  Temp limit : {CONFIG['temp_limit']}C")
    print("=" * 60)

    train_ds = CarDDDataset(
        "dataset/images/train", "dataset/labels/train",
        CONFIG["imgsz"], train=True
    )
    val_ds = CarDDDataset(
        "dataset/images/val", "dataset/labels/val",
        CONFIG["imgsz"], train=False
    )
    train_dl = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"],
        shuffle=True, collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"]
    )

    print(f"\n  Train: {len(train_ds)} images | Val: {len(val_ds)} images\n")

    model = build_model(CONFIG["num_classes"], CONFIG["imgsz"]).to(device)

    ckpt_path, start_epoch = find_latest_checkpoint()
    if ckpt_path:
        print(f"  Resuming from : {ckpt_path}")
        # DetBenchTrain wraps the inner model; load into the bench wrapper
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("  Starting fresh training...")
        start_epoch = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )
    scaler = torch.amp.GradScaler("cuda")

    Path("runs/efficientdet").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    history_path = Path("runs/efficientdet/loss_history.json")
    if history_path.exists() and start_epoch > 0:
        with open(history_path) as f:
            history = json.load(f)
        best_loss = min(h["loss"] for h in history)
        print(f"  Loaded history : {len(history)} epochs | best loss: {best_loss:.4f}")
    else:
        history   = []
        best_loss = float("inf")

    patience_ct = 0
    start       = time.time()

    print(f"\n  Training epoch {start_epoch + 1} → {CONFIG['epochs']}...\n")

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        n_batches  = 0

        for batch_idx, (imgs, boxes, labels) in enumerate(train_dl):
            imgs   = imgs.to(device)
            boxes  = boxes.to(device)     # [B, max_n, 4]  (y1,x1,y2,x2)
            labels = labels.to(device)    # [B, max_n]      (1-indexed)

            # Build the target dict that DetBenchTrain.forward() expects:
            #   bbox  : [B, N, 4]  float32  (y1, x1, y2, x2) pixel coords
            #   cls   : [B, N]     float32  (1-indexed class ids)
            # DetBenchTrain handles anchor labeling → label_cls_*, label_box_*
            target = {
                "bbox": boxes.float(),
                "cls":  labels.float(),
            }

            try:
                with torch.amp.autocast("cuda"):
                    output = model(imgs, target)
                    loss   = output["loss"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                n_batches  += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  [OOM] Epoch {epoch+1} batch {batch_idx} — clearing cache...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    time.sleep(5)
                else:
                    raise e

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch + 1, "loss": round(avg_loss, 4)})
        scheduler.step()

        # ── GPU temperature check ───────────────────────────
        temp = get_gpu_temp()
        if temp is not None:
            tstr = f" | GPU: {temp}C"
            if temp > CONFIG["temp_limit"]:
                tstr += " ⚠️  OVERHEATING"
                print(f"  Epoch [{epoch+1:3d}/{CONFIG['epochs']}] | Loss: {avg_loss:.4f}{tstr}")
                print(f"  Pausing {CONFIG['temp_pause']}s to cool down...")
                time.sleep(CONFIG["temp_pause"])
            else:
                tstr += " ✅"
        else:
            tstr = ""

        # Live ETA
        elapsed_hr  = (time.time() - start) / 3600
        epochs_done = (epoch + 1) - start_epoch
        epochs_left = CONFIG["epochs"] - (epoch + 1)
        eta_hr      = (elapsed_hr / epochs_done) * epochs_left if epochs_done > 0 else 0

        print(f"  Epoch [{epoch+1:3d}/{CONFIG['epochs']}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}{tstr} | ETA: {eta_hr:.1f}h")

        # ── Save best ───────────────────────────────────────
        if avg_loss < best_loss:
            best_loss   = avg_loss
            patience_ct = 0
            torch.save(model.state_dict(), "runs/efficientdet/best.pt")
            print(f"  --> ✅ Best saved (loss={best_loss:.4f})")
        else:
            patience_ct += 1
            if patience_ct >= CONFIG["patience"]:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1} — no improvement for {CONFIG['patience']} epochs")
                break

        # ── Periodic checkpoint ─────────────────────────────
        if (epoch + 1) % CONFIG["save_period"] == 0:
            torch.save(model.state_dict(), f"runs/efficientdet/epoch_{epoch+1:03d}.pt")
            print(f"  --> 💾 Checkpoint: epoch_{epoch+1:03d}.pt")

        # Save history every epoch — crash protection
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    elapsed = (time.time() - start) / 3600

    best_pt = Path("runs/efficientdet/best.pt")
    model_mb = round(best_pt.stat().st_size / 1e6, 1) if best_pt.exists() else 0.0

    row = {
        "model":         "EfficientDet-D2",
        "epochs":        CONFIG["epochs"],
        "best_loss":     round(best_loss, 4),
        "train_time_hr": round(elapsed, 2),
        "batch_size":    CONFIG["batch_size"],
        "imgsz":         CONFIG["imgsz"],
        "model_size_mb": model_mb,
    }
    with open("results/efficientdet_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader()
        w.writerow(row)

    print("\n" + "=" * 60)
    print("  ✅ EFFICIENTDET TRAINING COMPLETE")
    print("=" * 60)
    for k, v in row.items():
        print(f"  {k:<20}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()