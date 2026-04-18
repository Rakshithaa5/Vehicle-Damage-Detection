import glob
import torch
import os
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ── Config ────────────────────────────────────────────────
NUM_CLASSES  = 7  # 6 damage classes + 1 background
CHECKPOINT_DIR = r"C:\Users\raksh\dl\runs\faster_rcnn"
VAL_IMG_DIR  = r"C:\Users\raksh\dl\CarDD_raw\CarDD_release\CarDD_COCO\val2017"
VAL_ANN_FILE = r"C:\Users\raksh\dl\CarDD_raw\CarDD_release\CarDD_COCO\annotations\instances_val2017.json"

# ── Device ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Model ─────────────────────────────────────────────────
def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = build_model(NUM_CLASSES).to(device)

# ── Val Loader ────────────────────────────────────────────
def get_transform():
    return T.Compose([T.ToTensor()])

val_dataset = CocoDetection(VAL_IMG_DIR, VAL_ANN_FILE, transform=get_transform())
val_loader  = DataLoader(val_dataset, batch_size=2, shuffle=False,
                         collate_fn=lambda x: tuple(zip(*x)))

print(f"Validation images: {len(val_dataset)}")

# ── Evaluate Function ─────────────────────────────────────
def evaluate(model, val_loader):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]

            # Convert COCO targets to torchmetrics format
            tgt_list = []
            for t in targets:
                if len(t) == 0:
                    tgt_list.append({
                        'boxes':  torch.zeros((0, 4), dtype=torch.float32).to(device),
                        'labels': torch.zeros((0,),   dtype=torch.int64).to(device)
                    })
                    continue

                boxes = torch.tensor([
                    [a['bbox'][0], a['bbox'][1],
                     a['bbox'][0] + a['bbox'][2],
                     a['bbox'][1] + a['bbox'][3]]
                    for a in t
                ], dtype=torch.float32).to(device)

                labels = torch.tensor(
                    [a['category_id'] for a in t], dtype=torch.int64
                ).to(device)

                tgt_list.append({'boxes': boxes, 'labels': labels})

            preds = model(images)
            metric.update(preds, tgt_list)

    result = metric.compute()
    return result['map'].item()

# ── Loop Over All Checkpoints ─────────────────────────────
checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
print(f"\nFound {len(checkpoints)} checkpoints\n")

results = []

for ckpt_path in checkpoints:
    epoch_num = int(os.path.basename(ckpt_path).replace("epoch_", "").replace(".pt", ""))

    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    val_map = evaluate(model, val_loader)
    results.append({'epoch': epoch_num, 'mAP': val_map})
    print(f"Epoch {epoch_num:03d} | mAP: {val_map:.4f}")

# ── Summary ───────────────────────────────────────────────
print("\n=== All Results (best → worst) ===")
for r in sorted(results, key=lambda x: x['mAP'], reverse=True):
    bar = '█' * int(r['mAP'] * 40)
    print(f"Epoch {r['epoch']:03d} | mAP: {r['mAP']:.4f} | {bar}")

best = max(results, key=lambda x: x['mAP'])
print(f"\n✅ Best checkpoint: epoch_{best['epoch']:03d}.pt  |  mAP: {best['mAP']:.4f}")