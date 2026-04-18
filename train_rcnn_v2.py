import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────
NUM_CLASSES    = 7
BATCH_SIZE     = 8
EPOCHS         = 80
START_EPOCH    = 21
LR             = 0.0005
WEIGHT_DECAY   = 0.001
PATIENCE       = 15
SAVE_EVERY     = 5
CHECKPOINT_DIR = r"C:\Users\raksh\dl\runs\faster_rcnn_v2"
RESUME_FROM    = r"C:\Users\raksh\dl\runs\faster_rcnn\epoch_020.pt"

TRAIN_IMG_DIR  = r"C:\Users\raksh\dl\CarDD_raw\CarDD_release\CarDD_COCO\train2017"
TRAIN_ANN_FILE = r"C:\Users\raksh\dl\CarDD_raw\CarDD_release\CarDD_COCO\annotations\instances_train2017.json"
VAL_IMG_DIR    = r"C:\Users\raksh\dl\CarDD_raw\CarDD_release\CarDD_COCO\val2017"
VAL_ANN_FILE   = r"C:\Users\raksh\dl\CarDD_raw\CarDD_release\CarDD_COCO\annotations\instances_val2017.json"

# ── Augmentation ──────────────────────────────────────────
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3,
                                contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10,
                          sat_shift_limit=30, p=0.4),
    A.GaussNoise(p=0.2),
    A.Sharpen(alpha=(0.2, 0.5), p=0.3),
    A.Normalize(),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc',
                             label_fields=['class_labels'],
                             min_visibility=0.3))

val_aug = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# ── Dataset ───────────────────────────────────────────────
class CarDDDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        from pycocotools.coco import COCO
        self.img_dir    = img_dir
        self.coco       = COCO(ann_file)
        self.ids        = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img      = np.array(Image.open(img_path).convert('RGB'))

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        boxes  = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes  = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,),   dtype=np.int64)
        else:
            boxes  = np.array(boxes,  dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        if self.transforms:
            if len(boxes) > 0:
                transformed = self.transforms(
                    image=img,
                    bboxes=boxes.tolist(),
                    class_labels=labels.tolist()
                )
                img    = transformed['image']
                boxes  = torch.tensor(transformed['bboxes'],       dtype=torch.float32)
                labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)
                if len(boxes) == 0:
                    boxes  = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,),   dtype=torch.int64)
            else:
                transformed = self.transforms(image=img, bboxes=[], class_labels=[])
                img    = transformed['image']
                boxes  = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            img    = T.ToTensor()(Image.fromarray(img))
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes':    boxes,
            'labels':   labels,
            'image_id': torch.tensor([img_id])
        }
        return img, target

    def get_classes_in_image(self, idx):
        img_id  = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)
        return [a['category_id'] for a in anns]

# ── Collate ───────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))

# ── Weighted Sampler ──────────────────────────────────────
def build_weighted_sampler(dataset):
    class_counts  = {1: 280, 2: 434, 3: 90, 4: 129, 5: 118, 6: 55}
    class_weights = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
    sample_weights = []
    for i in range(len(dataset)):
        classes = dataset.get_classes_in_image(i)
        w = max(class_weights.get(c, 1.0) for c in classes) if classes else 1.0
        sample_weights.append(w)
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# ── Model ─────────────────────────────────────────────────
def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ── Evaluate ──────────────────────────────────────────────
CLASS_NAMES = ['dent', 'scratch', 'crack',
               'shattered_glass', 'bumper_damage', 'deformation']

def evaluate(model, val_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    with torch.no_grad():
        for images, targets in val_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds   = model(images)
            metric.update(preds, targets)
    result = metric.compute()
    return result['map'].item(), result

# ── Train One Epoch ───────────────────────────────────────
def train_one_epoch(model, optimizer, loader, epoch, device):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += losses.item()
        if i % 50 == 0:
            print(f"  Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {losses.item():.4f}")
    return total_loss / len(loader)

# ── Main ──────────────────────────────────────────────────
if __name__ == '__main__':

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = build_model(NUM_CLASSES).to(device)
    print(f"Resuming from: {RESUME_FROM}")
    ckpt = torch.load(RESUME_FROM, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    print(f"✅ Loaded epoch 20 weights. Continuing from epoch {START_EPOCH}...\n")

    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Datasets
    print("Loading datasets...")
    train_dataset = CarDDDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=train_aug)
    val_dataset   = CarDDDataset(VAL_IMG_DIR,   VAL_ANN_FILE,   transforms=val_aug)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    sampler      = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler,
                              num_workers=2,
                              pin_memory=True,
                              persistent_workers=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=4,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=True,
                              persistent_workers=True,
                              collate_fn=collate_fn)

    # Training Loop
    best_map         = 0.0
    patience_counter = 0
    END_EPOCH        = START_EPOCH + EPOCHS
    loss_history     = []
    map_history      = []

    print(f"Training from epoch {START_EPOCH} to max epoch {END_EPOCH}")
    print("=" * 55)

    for epoch in range(START_EPOCH, END_EPOCH + 1):

        avg_loss = train_one_epoch(model, optimizer, train_loader, epoch, device)
        val_map, full_result = evaluate(model, val_loader, device)
        scheduler.step()

        loss_history.append(avg_loss)
        map_history.append(val_map)

        print(f"\nEpoch {epoch:03d} | Loss: {avg_loss:.4f} | "
              f"mAP: {val_map:.4f} | Best: {best_map:.4f}")

        # Per-class AP
        if 'map_per_class' in full_result:
            aps = full_result['map_per_class'].tolist()
            for name, ap in zip(CLASS_NAMES, aps):
                status = '🟢' if ap > 0.5 else '🟡' if ap > 0.3 else '🔴'
                print(f"  {status} {name:<20} AP: {ap:.4f}")

        # Overfitting Monitor
        if len(loss_history) >= 3:
            loss_trend = loss_history[-1] - loss_history[-3]
            map_trend  = map_history[-1]  - map_history[-3]

            print(f"\n  📊 Trend (last 3 epochs):")
            print(f"     Loss: {loss_trend:+.4f} {'⬇️ good' if loss_trend < 0 else '⬆️ rising'}")
            print(f"     mAP : {map_trend:+.4f}  {'⬆️ good' if map_trend  > 0 else '⬇️ dropping'}")

            if loss_trend < -0.01 and map_trend < -0.005:
                print(f"  ⚠️  OVERFIT WARNING: loss dropping but mAP dropping too!")
            elif loss_trend > 0.01 and map_trend < 0:
                print(f"  ⚠️  INSTABILITY: both loss and mAP degrading")
            elif loss_trend < 0 and map_trend > 0:
                print(f"  ✅ Healthy: loss down, mAP up")
            else:
                print(f"  🟡 Watching: no clear trend yet")

        # Save every N epochs
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': val_map
            }, ckpt_path)
            print(f"  💾 Saved: {ckpt_path}")

        # Best checkpoint
        if val_map > best_map:
            best_map         = val_map
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'map': best_map
            }, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  ✅ New best saved! mAP: {best_map:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{PATIENCE})")

        # Early Stopping
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch}. Best mAP: {best_map:.4f}")
            break

    print(f"\n🏁 Training complete.")
    print(f"   Best mAP  : {best_map:.4f}")
    print(f"   Best model: {os.path.join(CHECKPOINT_DIR, 'best.pt')}")