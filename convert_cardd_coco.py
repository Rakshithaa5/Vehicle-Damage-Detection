"""
CarDD COCO → YOLO Converter
============================
Converts CarDD's COCO JSON annotations to YOLO txt format
and sets up the correct folder structure.

Run: python convert_cardd_coco.py
"""

import json, shutil, os
from pathlib import Path

# ── Paths (auto-detected from CarDD_raw) ──────────────────
RAW_ROOT   = Path("CarDD_raw/CarDD_release/CarDD_COCO")
ANNO_DIR   = RAW_ROOT / "annotations"
OUT_DIR    = Path("dataset")

SPLITS = {
    "train": {
        "json":    ANNO_DIR / "instances_train2017.json",
        "img_dir": RAW_ROOT / "train2017",
        "out_img": OUT_DIR  / "images/train",
        "out_lbl": OUT_DIR  / "labels/train",
    },
    "val": {
        "json":    ANNO_DIR / "instances_val2017.json",
        "img_dir": RAW_ROOT / "val2017",
        "out_img": OUT_DIR  / "images/val",
        "out_lbl": OUT_DIR  / "labels/val",
    },
    "test": {
        "json":    ANNO_DIR / "instances_test2017.json",
        "img_dir": RAW_ROOT / "test2017",
        "out_img": OUT_DIR  / "images/test",
        "out_lbl": OUT_DIR  / "labels/test",
    },
}

def convert_split(split_name, cfg):
    print(f"\n[{split_name.upper()}] Processing...")

    # Create output folders
    cfg["out_img"].mkdir(parents=True, exist_ok=True)
    cfg["out_lbl"].mkdir(parents=True, exist_ok=True)

    # Load COCO JSON
    if not cfg["json"].exists():
        print(f"  [SKIP] JSON not found: {cfg['json']}")
        return 0, 0

    with open(cfg["json"]) as f:
        coco = json.load(f)

    # Build category map: coco_id → yolo_id (0-indexed)
    categories = coco.get("categories", [])
    print(f"  Categories found: {[c['name'] for c in categories]}")
    cat_map = {c["id"]: i for i, c in enumerate(categories)}

    # Build image id → filename map
    images = {img["id"]: img for img in coco.get("images", [])}

    # Group annotations by image id
    ann_by_img = {}
    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        if iid not in ann_by_img:
            ann_by_img[iid] = []
        ann_by_img[iid].append(ann)

    n_imgs = 0
    n_anns = 0

    for img_id, img_info in images.items():
        fname    = img_info["file_name"]
        W        = img_info["width"]
        H        = img_info["height"]
        src_img  = cfg["img_dir"] / fname

        # Try alternate path (some COCO datasets use just filename without subdir)
        if not src_img.exists():
            src_img = cfg["img_dir"] / Path(fname).name

        if not src_img.exists():
            continue

        # Copy image
        dst_img = cfg["out_img"] / Path(fname).name
        shutil.copy2(src_img, dst_img)

        # Write YOLO label file
        anns  = ann_by_img.get(img_id, [])
        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_map:
                continue
            yolo_cls = cat_map[cat_id]

            # COCO bbox: [x, y, width, height] (top-left corner)
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            nw = bw / W
            nh = bh / H

            # Clamp to [0,1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            if nw > 0 and nh > 0:
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_anns += 1

        # Write label file (even if empty — keeps image-label pairing)
        lbl_file = cfg["out_lbl"] / (Path(fname).stem + ".txt")
        lbl_file.write_text("\n".join(lines))
        n_imgs += 1

    print(f"  Images converted : {n_imgs}")
    print(f"  Annotations      : {n_anns}")
    return n_imgs, n_anns


def write_yaml(categories):
    """Write dataset.yaml using actual CarDD category names"""
    names = {i: c["name"] for i, c in enumerate(categories)}
    nc    = len(names)

    yaml_content = f"""# CarDD — Vehicle Damage Detection
path: {OUT_DIR.resolve()}

train: images/train
val:   images/val
test:  images/test

nc: {nc}
names:
"""
    for i, name in names.items():
        yaml_content += f"  {i}: {name}\n"

    # Write to dataset folder AND project root
    (OUT_DIR / "dataset.yaml").write_text(yaml_content)
    Path("dataset.yaml").write_text(yaml_content)
    print(f"\n  dataset.yaml written with {nc} classes: {list(names.values())}")


def print_distribution():
    print("\n" + "="*50)
    print("  Class distribution — training set")
    print("="*50)

    # Load yaml to get class names
    yaml_path = Path("dataset.yaml")
    if not yaml_path.exists():
        return

    counts = {}
    for txt in (OUT_DIR / "labels/train").glob("*.txt"):
        for line in txt.read_text().strip().splitlines():
            if line:
                cls = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1

    total = sum(counts.values())
    for cls_id in sorted(counts):
        count = counts[cls_id]
        bar   = "█" * int((count / max(total, 1)) * 30)
        print(f"  Class {cls_id}: {count:>6} annotations  {bar}")

    print(f"\n  Total annotations : {total}")
    print("="*50)


def main():
    print("="*50)
    print("  CarDD COCO → YOLO Converter")
    print("="*50)

    # Verify raw folder exists
    if not RAW_ROOT.exists():
        print(f"\n[ERROR] Raw folder not found: {RAW_ROOT}")
        print("Make sure setup_cardd.py ran and extracted to CarDD_raw/")
        return

    # Get categories from train JSON
    train_json = SPLITS["train"]["json"]
    if not train_json.exists():
        print(f"[ERROR] Train JSON not found: {train_json}")
        return

    with open(train_json) as f:
        coco = json.load(f)
    categories = coco.get("categories", [])

    total_imgs = 0
    total_anns = 0

    for split_name, cfg in SPLITS.items():
        imgs, anns = convert_split(split_name, cfg)
        total_imgs += imgs
        total_anns += anns

    write_yaml(categories)
    print_distribution()

    print(f"""
{'='*50}
  CONVERSION COMPLETE
{'='*50}
  Total images     : {total_imgs}
  Total annotations: {total_anns}
  Output folder    : {OUT_DIR.resolve()}

  Next step:
      python train_yolov8.py
{'='*50}
""")


if __name__ == "__main__":
    main()
