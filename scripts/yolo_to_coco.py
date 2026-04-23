"""Convert merged YOLO-format dataset to COCO JSON for DEIM training.

Reads `data/merged/{images,labels}/{train,val}/` and writes
`data/merged/annotations/instances_{train,val}.json`. Images are not
copied — DEIM's `img_folder` points at the same directory as Ultralytics.

Category IDs are 0-indexed to match the YOLO class IDs in traffic_light.yaml.
Set DEIM's `remap_mscoco_category: False` and `num_classes: 7`.
"""

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MERGED_DIR = ROOT / "data" / "merged"
DATA_YAML = ROOT / "data" / "traffic_light.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_class_names() -> list[str]:
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    return [names[i] for i in sorted(names)]


def yolo_to_coco_bbox(cx: float, cy: float, w: float, h: float,
                     img_w: int, img_h: int) -> list[float]:
    """YOLO (cx, cy, w, h) normalized → COCO (x_min, y_min, w, h) absolute."""
    bw = w * img_w
    bh = h * img_h
    return [(cx - w / 2) * img_w, (cy - h / 2) * img_h, bw, bh]


def convert_split(split: str, names: list[str]) -> dict:
    img_dir = MERGED_DIR / "images" / split
    lbl_dir = MERGED_DIR / "labels" / split

    images = []
    annotations = []
    img_id = 0
    ann_id = 0

    img_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS and not p.name.startswith("._")
    )

    for img_path in tqdm(img_paths, desc=f"convert {split}"):
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    bbox = yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h)
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1
        img_id += 1

    categories = [{"id": i, "name": n} for i, n in enumerate(names)]

    return {
        "info": {"description": "Traffic light detection (YOLO→COCO)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    names = load_class_names()
    print(f"Classes ({len(names)}): {names}")

    out_dir = MERGED_DIR / "annotations"
    out_dir.mkdir(exist_ok=True)

    for split in args.splits:
        coco = convert_split(split, names)
        out_path = out_dir / f"instances_{split}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f)
        print(f"  {split}: {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations → {out_path}")


if __name__ == "__main__":
    main()
