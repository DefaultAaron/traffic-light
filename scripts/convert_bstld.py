"""Convert BSTLD dataset from YAML to YOLO format.

Source: data/raw/BSTLD/{train,test}/
Output: data/raw/BSTLD/yolo_labels/*.txt
"""

from pathlib import Path

import yaml

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "BSTLD"
OUTPUT_DIR = RAW_DIR / "yolo_labels"

IMG_W, IMG_H = 1280, 720

CLASS_MAP = {
    "Red": 0,
    "RedLeft": 0,
    "RedRight": 0,
    "RedStraight": 0,
    "RedStraightLeft": 0,
    "Yellow": 1,
    "Green": 2,
    "GreenLeft": 2,
    "GreenRight": 2,
    "GreenStraight": 2,
    "GreenStraightLeft": 2,
    "GreenStraightRight": 2,
}
SKIP_CLASSES = {"off"}


def resolve_image_path(yaml_path: Path, entry_path: str) -> Path | None:
    """Resolve the actual image path from the YAML entry."""
    p = Path(entry_path)
    filename = p.name

    # Train YAML: relative paths like ./rgb/train/bag_name/123456.png
    if entry_path.startswith("./"):
        candidate = yaml_path.parent / entry_path
        if candidate.exists():
            return candidate

    # Test YAML: absolute Bosch paths — extract filename, look in test/rgb/test/
    # Path pattern: /net/.../traffic_lights/<run_name>/<frame>.png
    parts = p.parts
    if len(parts) >= 2:
        run_name = parts[-2]
        candidate = yaml_path.parent / "rgb" / "test" / run_name / filename
        if candidate.exists():
            return candidate

    # Fallback: search in rgb subdirectory
    for candidate in yaml_path.parent.rglob(filename):
        return candidate

    return None


def convert_split(yaml_path: Path, split_name: str) -> tuple[int, int, dict[int, int]]:
    """Convert one YAML split file. Returns (images, boxes, class_counts)."""
    print(f"  Loading {yaml_path} ...")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    total_images = 0
    total_boxes = 0
    total_counts: dict[int, int] = {}
    missing_images = 0

    for entry in data:
        entry_path = entry["path"]
        boxes = entry.get("boxes", [])

        img_path = resolve_image_path(yaml_path, entry_path)
        if img_path is None:
            missing_images += 1
            continue

        # Use split prefix + image filename as label stem to avoid collisions
        label_stem = f"{split_name}_{img_path.stem}"

        lines = []
        for box in boxes:
            label = box["label"]
            if label in SKIP_CLASSES:
                continue
            cls_id = CLASS_MAP.get(label)
            if cls_id is None:
                print(f"    WARNING: unknown class '{label}', skipping")
                continue

            xmin, ymin = box["x_min"], box["y_min"]
            xmax, ymax = box["x_max"], box["y_max"]

            cx = (xmin + xmax) / 2.0 / IMG_W
            cy = (ymin + ymax) / 2.0 / IMG_H
            w = (xmax - xmin) / IMG_W
            h = (ymax - ymin) / IMG_H

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_counts[cls_id] = total_counts.get(cls_id, 0) + 1

        out_path = OUTPUT_DIR / f"{label_stem}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        total_images += 1
        total_boxes += len(lines)

    if missing_images:
        print(f"    WARNING: {missing_images} images not found")

    return total_images, total_boxes, total_counts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grand_images = 0
    grand_boxes = 0
    grand_counts: dict[int, int] = {}

    for split_name, yaml_name in [("train", "train/train.yaml"), ("test", "test/test.yaml")]:
        yaml_path = RAW_DIR / yaml_name
        if not yaml_path.exists():
            print(f"  WARNING: {yaml_path} not found, skipping {split_name}")
            continue

        images, boxes, counts = convert_split(yaml_path, split_name)
        grand_images += images
        grand_boxes += boxes
        for cls_id, cnt in counts.items():
            grand_counts[cls_id] = grand_counts.get(cls_id, 0) + cnt

    cls_names = {0: "red", 1: "yellow", 2: "green"}
    print(f"\nBSTLD conversion complete:")
    print(f"  Images: {grand_images}")
    print(f"  Total boxes: {grand_boxes}")
    for cls_id in sorted(grand_counts):
        print(f"  {cls_names[cls_id]}: {grand_counts[cls_id]}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
