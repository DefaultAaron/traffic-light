"""Convert BSTLD dataset to YOLO format (7-class).

Train set: from YAML (has directional labels)
Test set:  from re-annotated Pascal VOC XML in annotations_fix/

Source: data/raw/BSTLD/{train,test}/
Output: data/raw/BSTLD/yolo_labels/*.txt
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "BSTLD"
OUTPUT_DIR = RAW_DIR / "yolo_labels"
ANNOT_FIX_DIR = RAW_DIR / "annotations_fix"

IMG_W, IMG_H = 1280, 720

# Train set YAML uses capitalized labels
TRAIN_CLASS_MAP = {
    "Red": 0,
    "RedLeft": 3,
    "RedRight": 5,
    "RedStraight": 0,      # RedStraight → red (forward collapsed to base color)
    "Yellow": 1,
    "Green": 2,
    "GreenLeft": 4,
    "GreenRight": 6,
    "GreenStraight": 2,    # GreenStraight → green (forward collapsed to base color)
}

# Test set re-annotated XML uses lowercase labels (same as S2TLD Annotations-fix)
TEST_CLASS_MAP = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "redLeft": 3,
    "greenLeft": 4,
    "redRight": 5,
    "greenRight": 6,
    # Forward arrows → base color (functionally equivalent to round lights)
    "redForward": 0,
    "greenForward": 2,
    "yellowLeft": 1,
    "yellowForward": 1,
    "yellowRight": 1,
}

# Combined multi-direction labels (e.g. StraightLeft) skipped — only 5 total,
# visually arrow-shaped but not round; rare in real-world Chinese traffic.
TRAIN_SKIP_CLASSES = {"off", "RedStraightLeft", "GreenStraightLeft", "GreenStraightRight"}
TEST_SKIP_CLASSES = {"off", "wait_on"}


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


def convert_train_yaml(yaml_path: Path) -> tuple[int, int, dict[int, int]]:
    """Convert train set from YAML. Returns (images, boxes, class_counts)."""
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

        label_stem = f"train_{img_path.stem}"

        lines = []
        for box in boxes:
            label = box["label"]
            if label in TRAIN_SKIP_CLASSES:
                continue
            cls_id = TRAIN_CLASS_MAP.get(label)
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


def convert_test_xml() -> tuple[int, int, dict[int, int]]:
    """Convert test set from re-annotated Pascal VOC XML. Returns (images, boxes, class_counts)."""
    if not ANNOT_FIX_DIR.exists():
        print(f"  WARNING: {ANNOT_FIX_DIR} not found, skipping test set")
        return 0, 0, {}

    xml_files = sorted(ANNOT_FIX_DIR.glob("*.xml"))
    print(f"  Test set: found {len(xml_files)} re-annotated XML files")

    images_dir = RAW_DIR / "test" / "rgb" / "test"

    # Build image index upfront (stem → path) to avoid rglob per XML
    img_index: dict[str, Path] = {}
    for img in images_dir.rglob("*.png"):
        img_index[img.stem] = img
    print(f"  Test image index: {len(img_index)} images")

    total_images = 0
    total_boxes = 0
    total_counts: dict[int, int] = {}
    missing_images = 0

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find the corresponding image via index
        img_path = img_index.get(xml_path.stem)
        if img_path is None:
            missing_images += 1
            continue

        size_el = root.find("size")
        w_img = int(size_el.find("width").text)
        h_img = int(size_el.find("height").text)

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name in TEST_SKIP_CLASSES:
                continue
            cls_id = TEST_CLASS_MAP.get(name)
            if cls_id is None:
                print(f"    WARNING: unknown class '{name}' in {xml_path.name}, skipping")
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            cx = (xmin + xmax) / 2.0 / w_img
            cy = (ymin + ymax) / 2.0 / h_img
            w = (xmax - xmin) / w_img
            h = (ymax - ymin) / h_img

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_counts[cls_id] = total_counts.get(cls_id, 0) + 1

        label_stem = f"test_{xml_path.stem}"
        out_path = OUTPUT_DIR / f"{label_stem}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        total_images += 1
        total_boxes += len(lines)

    if missing_images:
        print(f"    WARNING: {missing_images} test images not found")

    return total_images, total_boxes, total_counts


def main():
    # Clean previous output to avoid stale labels from old conversions
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grand_images = 0
    grand_boxes = 0
    grand_counts: dict[int, int] = {}

    # Train set: from YAML (has directional labels)
    train_yaml = RAW_DIR / "train" / "train.yaml"
    if train_yaml.exists():
        images, boxes, counts = convert_train_yaml(train_yaml)
        grand_images += images
        grand_boxes += boxes
        for cls_id, cnt in counts.items():
            grand_counts[cls_id] = grand_counts.get(cls_id, 0) + cnt
    else:
        print(f"  WARNING: {train_yaml} not found, skipping train set")

    # Test set: from re-annotated XML (7-class directional labels)
    images, boxes, counts = convert_test_xml()
    grand_images += images
    grand_boxes += boxes
    for cls_id, cnt in counts.items():
        grand_counts[cls_id] = grand_counts.get(cls_id, 0) + cnt

    cls_names = {0: "red", 1: "yellow", 2: "green", 3: "redLeft", 4: "greenLeft", 5: "redRight", 6: "greenRight"}
    print(f"\nBSTLD conversion complete:")
    print(f"  Images: {grand_images}")
    print(f"  Total boxes: {grand_boxes}")
    for cls_id in sorted(grand_counts):
        print(f"  {cls_names[cls_id]}: {grand_counts[cls_id]}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
