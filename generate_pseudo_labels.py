"""
generate_pseudo_labels.py
Stage 2A: CLIP Teacher for Auto Severity Pseudo-Labeling
Vehicle Damage Detection | CarDD Dataset
Zero manual labeling required ✅

Usage:
  python generate_pseudo_labels.py \
    --det_model runs/yolov8m/weights/best.pt \
    --data_dir dataset/images/val \
    --output pseudo_labels.json \
    --conf_thresh 0.25

Output:
  - pseudo_labels.json: Structured labels for training
  - crops/: Cropped damage patches (224x224) for student training
"""

import argparse, json, time, sys
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# ── Dependencies ──────────────────────────────────────────
try:
    from ultralytics import YOLO
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("\n[ERROR] Missing dependencies. Install with:")
    print("  pip install ultralytics transformers torch torchvision\n")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────
CONFIG = {
    "imgsz_detect": 640,      # YOLO detection size
    "imgsz_crop":   224,      # CLIP input size
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "conf_thresh":  0.25,     # YOLO confidence threshold
    "severity_levels": ["minor", "moderate", "severe"],
}

# Severity prompts per damage class (CLIP zero-shot)
SEVERITY_PROMPTS = {
    "dent": [
        "a minor car dent, small shallow indentation",
        "a moderate car dent, visible depression in panel",
        "a severe car dent, deep deformation with paint damage"
    ],
    "scratch": [
        "a light surface scratch on car paint, barely visible",
        "a deep scratch on car paint, clearly visible groove",
        "a scratch with paint removed, exposing primer or metal"
    ],
    "crack": [
        "a hairline crack on car surface, very thin line",
        "a visible crack on car panel, noticeable propagation",
        "a severe structural crack, wide and spreading"
    ],
    "shattered_glass": [
        "a small glass chip on windshield, minor damage",
        "a cracked windshield with spiderweb pattern",
        "fully shattered car glass, multiple fragments"
    ],
    "bumper_damage": [
        "a scuffed car bumper, minor surface abrasion",
        "a cracked car bumper, visible fracture line",
        "a detached car bumper, hanging or misaligned"
    ],
    "deformation": [
        "slight panel deformation on car, minor bending",
        "moderate denting on car body, noticeable shape change",
        "major structural deformation, severe panel collapse"
    ],
}

CLASS_NAMES = [
    "dent", "scratch", "crack", 
    "shattered_glass", "bumper_damage", "deformation"
]


# ── Pseudo Label Generator ─────────────────────────────────
class PseudoLabelGenerator:
    def __init__(self, det_model_path, device="cuda"):
        print(f"[\u2713] Loading YOLOv8 detector: {det_model_path}")
        self.detector = YOLO(det_model_path)
        
        print(f"[\u2713] Loading CLIP ViT-B/32 for zero-shot labeling")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        self.device = device
        self.stats = {"total_images": 0, "total_detections": 0, "skipped": 0}
    
    def _clip_score_severity(self, crop_pil, damage_class):
        """
        Use CLIP zero-shot to score severity prompts.
        Returns: (severity_label, confidence_0_to_1)
        """
        prompts = SEVERITY_PROMPTS.get(damage_class, SEVERITY_PROMPTS["dent"])
        
        inputs = self.clip_processor(
            text=prompts,
            images=crop_pil,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # logits_per_image: [1, num_prompts]
            probs = torch.softmax(outputs.logits_per_image, dim=1)[0]
            severity_idx = torch.argmax(probs).item()
            confidence = probs[severity_idx].item()
        
        severity_label = CONFIG["severity_levels"][severity_idx]
        return severity_label, round(confidence, 4)
    
    def _crop_and_preprocess(self, img_pil, bbox, img_w, img_h):
        """Crop detection and resize to CLIP input size."""
        x1, y1, x2, y2 = bbox
        # Convert normalized YOLO bbox to pixel coords
        x1_px = max(0, int(x1 * img_w))
        y1_px = max(0, int(y1 * img_h))
        x2_px = min(img_w, int(x2 * img_w))
        y2_px = min(img_h, int(y2 * img_h))
        
        crop = img_pil.crop((x1_px, y1_px, x2_px, y2_px))
        # Resize to CLIP input size with padding to avoid distortion
        crop = crop.resize(
            (CONFIG["imgsz_crop"], CONFIG["imgsz_crop"]), 
            Image.Resampling.BICUBIC
        )
        return crop
    
    def process_image(self, img_path, output_crop_dir):
        """Process single image: detect → crop → CLIP label → save."""
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_w, img_h = img_pil.size
        except Exception as e:
            print(f"  [\u2717] Failed to open {img_path}: {e}")
            return []
        
        # Stage 1: YOLO detection
        results = self.detector(
            img_pil, 
            conf=CONFIG["conf_thresh"],
            verbose=False
        )[0]
        
        if len(results.boxes) == 0:
            return []
        
        detections = []
        
        for box in results.boxes:
            # Extract bbox and class
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2] normalized
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            damage_class = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            
            # Crop and preprocess
            crop = self._crop_and_preprocess(img_pil, xyxy, img_w, img_h)
            
            # Stage 2A: CLIP zero-shot severity scoring
            severity, sev_conf = self._clip_score_severity(crop, damage_class)
            
            # Save crop for student training
            crop_filename = f"{img_path.stem}_{damage_class}_{len(detections)}.jpg"
            crop_path = output_crop_dir / crop_filename
            crop.save(crop_path)
            
            # Build pseudo-label entry
            entry = {
                "image_path": str(img_path),
                "crop_path": str(crop_path),
                "damage_class": damage_class,
                "bbox_normalized": xyxy.tolist(),  # [x1,y1,x2,y2] 0-1
                "detection_confidence": round(conf, 4),
                "pseudo_severity": severity,
                "pseudo_confidence": sev_conf,
                "severity_scores": {  # Full distribution for weighted training
                    sev: round(score.item(), 4) 
                    for sev, score in zip(
                        CONFIG["severity_levels"],
                        torch.softmax(
                            self.clip_model(
                                **self.clip_processor(
                                    text=SEVERITY_PROMPTS[damage_class],
                                    images=crop.resize((224,224)),
                                    return_tensors="pt",
                                    padding=True
                                ).to(self.device)
                            ).logits_per_image,
                            dim=1
                        )[0].cpu()
                    )
                } if False else {}  # Skip full scores to save time; enable if needed
            }
            detections.append(entry)
            self.stats["total_detections"] += 1
        
        self.stats["total_images"] += 1
        return detections
    
    def run(self, data_dir, output_json, output_crops_dir="crops"):
        """Process all images in data_dir and save pseudo-labels."""
        data_dir = Path(data_dir)
        output_json = Path(output_json)
        output_crops_dir = Path(output_crops_dir)
        output_crops_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_paths = sorted(
            list(data_dir.glob("*.jpg")) + 
            list(data_dir.glob("*.png")) +
            list(data_dir.glob("*.jpeg"))
        )
        
        if not image_paths:
            print(f"[\u2717] No images found in {data_dir}")
            return
        
        print(f"[\u2713] Processing {len(image_paths)} images...")
        all_pseudo_labels = []
        start_time = time.time()
        
        for idx, img_path in enumerate(image_paths):
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {idx+1}/{len(image_paths)} | Rate: {rate:.1f} img/s")
            
            detections = self.process_image(img_path, output_crops_dir)
            all_pseudo_labels.extend(detections)
        
        # Save results
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "det_model": self.detector.ckpt_path,
                    "clip_model": "openai/clip-vit-base-patch32",
                    "config": CONFIG,
                    "stats": self.stats
                },
                "pseudo_labels": all_pseudo_labels
            }, f, indent=2)
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\n[\u2713] Pseudo-label generation complete!")
        print(f"  Images processed : {self.stats['total_images']}")
        print(f"  Detections found : {self.stats['total_detections']}")
        print(f"  Output JSON      : {output_json}")
        print(f"  Crops saved to   : {output_crops_dir}/")
        print(f"  Time elapsed     : {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Avg per image    : {elapsed/max(1,self.stats['total_images']):.2f}s")
        
        return all_pseudo_labels


# ── Main ──────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Generate severity pseudo-labels using CLIP")
    parser.add_argument("--det_model", type=str, required=True,
                       help="Path to trained YOLOv8 model (e.g., runs/yolov8m/weights/best.pt)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory with images to process (e.g., dataset/images/val)")
    parser.add_argument("--output", type=str, default="pseudo_labels.json",
                       help="Output JSON file for pseudo-labels")
    parser.add_argument("--crops_dir", type=str, default="crops",
                       help="Directory to save cropped damage patches")
    parser.add_argument("--conf_thresh", type=float, default=0.25,
                       help="YOLO confidence threshold for detections")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use: 'cuda', 'cpu', or 'cuda:0'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Override device if specified
    device = args.device or CONFIG["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[\u26a0\uFE0F] CUDA not available, falling back to CPU")
        device = "cpu"
    
    print("=" * 60)
    print("  Stage 2A: CLIP Teacher — Pseudo-Label Generation")
    print("  Vehicle Damage Severity Assessment | Zero Manual Labels")
    print("=" * 60)
    print(f"  Detector model : {args.det_model}")
    print(f"  Input images   : {args.data_dir}")
    print(f"  Output JSON    : {args.output}")
    print(f"  Crops directory: {args.crops_dir}")
    print(f"  Device         : {device}")
    print(f"  Confidence thresh: {args.conf_thresh}")
    print("=" * 60 + "\n")
    
    generator = PseudoLabelGenerator(
        det_model_path=args.det_model,
        device=device
    )
    
    pseudo_labels = generator.run(
        data_dir=args.data_dir,
        output_json=args.output,
        output_crops_dir=args.crops_dir
    )
    
    if pseudo_labels:
        print(f"\n[\u2713] Ready for Stage 2B: Train student model with:")
        print(f"  python train_severity_student.py --pseudo_labels {args.output}")
    else:
        print(f"\n[\u26a0\uFE0F] No detections found. Check:")
        print(f"  - YOLO model path and training")
        print(f"  - Confidence threshold (--conf_thresh)")
        print(f"  - Input images contain detectable damage")