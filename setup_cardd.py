"""
CarDD Dataset Setup Script
===========================
Extracts the CarDD zip file and reorganizes it into YOLO-ready format.
Run: python setup_cardd.py --zip CarDD.zip --output ./dataset
"""

import os, sys, shutil, zipfile, random, argparse
from pathlib import Path

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
CLASS_NAMES = ["dent","scratch","crack","shattered_glass","bumper_damage","deformation"]

def extract_zip(zip_path, extract_to="./CarDD_raw"):
    zip_path   = Path(zip_path)
    extract_to = Path(extract_to)
    if not zip_path.exists():
        print(f"[ERROR] Zip not found: {zip_path}"); sys.exit(1)
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/5] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"      Done → {extract_to.resolve()}")
    return extract_to

def inspect(root):
    print(f"\n[2/5] Inspecting structure ...")
    all_files  = list(Path(root).rglob("*"))
    img_exts   = {".jpg",".jpeg",".png",".bmp",".webp"}
    lbl_exts   = {".txt",".xml",".json"}
    images = [f for f in all_files if f.suffix.lower() in img_exts]
    labels = [f for f in all_files if f.suffix.lower() in lbl_exts]
    print(f"      Images found : {len(images)}")
    print(f"      Labels found : {len(labels)}")
    print("\n      Top-level folders:")
    for item in sorted(Path(root).iterdir()):
        print(f"        {item.name}")
    return images, labels

def match_pairs(images, labels):
    print(f"\n[3/5] Matching images with labels ...")
    label_map = {l.stem: l for l in labels}
    pairs = []
    for img in images:
        if img.stem in label_map:
            pairs.append((img, label_map[img.stem]))
        else:
            print(f"      [WARN] No label for {img.name} — skipping")
    print(f"      Matched: {len(pairs)} pairs")
    return pairs

def split(pairs):
    print(f"\n[4/5] Splitting dataset ...")
    random.seed(42); random.shuffle(pairs)
    n       = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits  = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train+n_val],
        "test":  pairs[n_train+n_val:],
    }
    for s, d in splits.items():
        print(f"      {s:6s}: {len(d)} pairs")
    return splits

def build(splits, output_dir):
    print(f"\n[5/5] Building YOLO structure ...")
    out = Path(output_dir)
    for s in ["train","val","test"]:
        (out/"images"/s).mkdir(parents=True, exist_ok=True)
        (out/"labels"/s).mkdir(parents=True, exist_ok=True)
    for split_name, pairs in splits.items():
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, out/"images"/split_name/img_path.name)
            dest = out/"labels"/split_name/(lbl_path.stem+".txt")
            if lbl_path.suffix == ".txt":
                shutil.copy2(lbl_path, dest)
            elif lbl_path.suffix == ".xml":
                convert_xml(lbl_path, dest)
    print(f"      Done → {out.resolve()}")
    return out

def convert_xml(xml_path, out_txt):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path); root = tree.getroot()
    size = root.find("size")
    W = float(size.find("width").text)
    H = float(size.find("height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower().replace(" ","_")
        if name not in CLASS_NAMES: continue
        cls  = CLASS_NAMES.index(name)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        cx=(xmin+xmax)/2/W; cy=(ymin+ymax)/2/H
        w=(xmax-xmin)/W;    h=(ymax-ymin)/H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    Path(out_txt).write_text("\n".join(lines))

def write_yaml(out_dir):
    yaml = f"""path: {Path(out_dir).resolve()}
train: images/train
val:   images/val
test:  images/test
nc: {len(CLASS_NAMES)}
names:\n"""
    for i,n in enumerate(CLASS_NAMES):
        yaml += f"  {i}: {n}\n"
    p = Path(out_dir)/"dataset.yaml"
    p.write_text(yaml)
    # Also write to project root
    Path("dataset.yaml").write_text(yaml)
    print(f"\n      dataset.yaml written!")

def check_dist(out_dir):
    print("\n" + "="*45)
    print("  Class distribution — training set")
    print("="*45)
    counts = {n:0 for n in CLASS_NAMES}
    for txt in (Path(out_dir)/"labels"/"train").glob("*.txt"):
        for line in txt.read_text().strip().splitlines():
            if line:
                idx = int(line.split()[0])
                if idx < len(CLASS_NAMES):
                    counts[CLASS_NAMES[idx]] += 1
    total = sum(counts.values())
    for name, count in counts.items():
        bar = "█" * int((count/max(total,1))*30)
        print(f"  {name:<20} {count:>5}  {bar}")
    print(f"\n  Total annotations: {total}")
    print("="*45)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip",    required=True)
    parser.add_argument("--output", default="./dataset")
    args = parser.parse_args()

    print("\n" + "="*45)
    print("  CarDD → YOLO Setup")
    print("="*45)

    raw     = extract_zip(args.zip)
    imgs, lbls = inspect(raw)
    pairs   = match_pairs(imgs, lbls)
    splits  = split(pairs)
    out_dir = build(splits, args.output)
    write_yaml(out_dir)
    check_dist(out_dir)

    print(f"""
✅ Dataset ready!

Next step:
    python train_yolov8.py
""")

if __name__ == "__main__":
    main()
