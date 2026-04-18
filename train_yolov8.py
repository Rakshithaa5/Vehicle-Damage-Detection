"""
Model 1 — YOLOv8m Training (Windows Compatible)
Vehicle Damage Detection | CarDD Dataset
RTX 5050 8GB optimized
"""

from ultralytics import YOLO
import torch, time, csv
from pathlib import Path

# Windows MUST have all code inside this block
if __name__ == "__main__":

    print("="*50)
    print("  YOLOv8m — Vehicle Damage Detection")
    print("="*50)
    print(f"  GPU available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*50)

    # Load pretrained COCO weights
    model = YOLO("yolov8m.pt")

    start = time.time()

    # Train
    results = model.train(
        data          = "dataset.yaml",
        epochs        = 100,
        imgsz         = 640,
        batch         = 16,       # reduce to 8 if OOM
        lr0           = 0.01,
        lrf           = 0.001,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 3,
        half          = True,     # fp16
        device        = 0,        # GPU
        workers       = 0,        # CRITICAL for Windows
        cache         = False,    # CRITICAL for Windows
        augment       = True,
        mosaic        = 1.0,
        mixup         = 0.15,
        copy_paste    = 0.3,
        fliplr        = 0.5,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 10.0,
        project       = "runs",
        name          = "yolov8m",
        plots         = True,
        save          = True,
        exist_ok      = True,
    )

    elapsed = (time.time() - start) / 3600

    # Evaluate on test set
    print("\nEvaluating on test set...")
    best_model = YOLO("runs/yolov8m/weights/best.pt")
    metrics = best_model.val(
        data    = "dataset.yaml",
        split   = "test",
        conf    = 0.25,
        iou     = 0.45,
        workers = 0,
    )

    # Save results
    Path("results").mkdir(exist_ok=True)
    row = {
        "model":         "YOLOv8m",
        "mAP50":         round(metrics.box.map50, 4),
        "mAP50_95":      round(metrics.box.map,   4),
        "precision":     round(metrics.box.mp,    4),
        "recall":        round(metrics.box.mr,    4),
        "train_time_hr": round(elapsed,           2),
        "model_size_mb": round(
            Path("runs/yolov8m/weights/best.pt").stat().st_size / 1e6, 1),
    }

    with open("results/yolov8_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader()
        w.writerow(row)

    print("\n" + "="*50)
    print("  YOLOV8m RESULTS")
    print("="*50)
    for k, v in row.items():
        print(f"  {k:<20}: {v}")
    print(f"\n  Best model : runs/yolov8m/weights/best.pt")
    print(f"  Metrics    : results/yolov8_metrics.csv")
    print("="*50)
