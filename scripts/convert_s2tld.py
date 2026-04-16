"""Convert S2TLD dataset from Pascal VOC XML to YOLO format (7-class).

Handles three subsets:
  - Original: data/raw/S2TLD/{JPEGImages,Annotations-fix}/ (1920x1080)
  - normal_1: data/raw/S2TLD/normal_1/{JPEGImages,Annotations-fix}/ (1280x720)
  - normal_2: data/raw/S2TLD/normal_2/{JPEGImages,Annotations-fix}/ (1280x720)
  All subsets use Annotations-fix (re-annotated with directional labels).

Output: data/raw/S2TLD/yolo_labels/*.txt
  - Original files keep original stems (timestamps with spaces)
  - normal_1/normal_2 files prefixed: normal1_000000.txt, normal2_000779.txt
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "S2TLD"
OUTPUT_DIR = RAW_DIR / "yolo_labels"

# (subset_prefix, annotations_dir, images_dir)
SUBSETS = [
    ("", RAW_DIR / "Annotations-fix", RAW_DIR / "JPEGImages"),
    (
        "normal1",
        RAW_DIR / "normal_1" / "Annotations-fix",
        RAW_DIR / "normal_1" / "JPEGImages",
    ),
    (
        "normal2",
        RAW_DIR / "normal_2" / "Annotations-fix",
        RAW_DIR / "normal_2" / "JPEGImages",
    ),
]

CLASS_MAP = {
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
    # Yellow directional → yellow round (data too sparse across all datasets)
    "yellowLeft": 1,
    "yellowForward": 1,
    "yellowRight": 1,
}
SKIP_CLASSES = {"off", "wait_on", "Wait_on"}


def convert_box(
    size: tuple[int, int], box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert (xmin, ymin, xmax, ymax) in pixels to (cx, cy, w, h) normalized."""
    w_img, h_img = size
    cx = (box[0] + box[2]) / 2.0 / w_img
    cy = (box[1] + box[3]) / 2.0 / h_img
    w = (box[2] - box[0]) / w_img
    h = (box[3] - box[1]) / h_img
    return cx, cy, w, h


def convert_one(xml_path: Path, out_stem: str) -> tuple[int, dict[int, int]]:
    """Convert one XML annotation file. Returns (num_boxes, class_counts)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    w_img = int(size_el.find("width").text)
    h_img = int(size_el.find("height").text)

    lines = []
    counts: dict[int, int] = {}
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name in SKIP_CLASSES:
            continue
        cls_id = CLASS_MAP.get(name)
        if cls_id is None:
            print(f"  WARNING: unknown class '{name}' in {xml_path.name}, skipping")
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        cx, cy, w, h = convert_box((w_img, h_img), (xmin, ymin, xmax, ymax))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[cls_id] = counts.get(cls_id, 0) + 1

    out_path = OUTPUT_DIR / f"{out_stem}.txt"
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return len(lines), counts


def main():
    # Clean previous output to avoid stale labels from old conversions
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grand_total_boxes = 0
    grand_total_counts: dict[int, int] = {}
    grand_total_images = 0

    for prefix, annot_dir, image_dir in SUBSETS:
        subset_name = prefix if prefix else "original"
        if not annot_dir.exists():
            print(f"  WARNING: {annot_dir} not found, skipping {subset_name}")
            continue

        xml_files = sorted(annot_dir.glob("*.xml"))
        print(f"\n{subset_name}: found {len(xml_files)} XML annotations")

        total_boxes = 0
        total_counts: dict[int, int] = {}
        skipped_no_image = 0

        for xml_path in xml_files:
            img_path = image_dir / f"{xml_path.stem}.jpg"
            if not img_path.exists():
                print(f"  WARNING: no image for {xml_path.name}, skipping")
                skipped_no_image += 1
                continue

            # Prefix output stem for normal_1/normal_2 to avoid collisions
            out_stem = f"{prefix}_{xml_path.stem}" if prefix else xml_path.stem

            n, counts = convert_one(xml_path, out_stem)
            total_boxes += n
            for cls_id, cnt in counts.items():
                total_counts[cls_id] = total_counts.get(cls_id, 0) + cnt

        n_images = len(xml_files) - skipped_no_image
        cls_names = {0: "red", 1: "yellow", 2: "green", 3: "redLeft", 4: "greenLeft", 5: "redRight", 6: "greenRight"}
        print(f"  {subset_name} conversion complete:")
        print(f"    Images: {n_images}")
        print(f"    Total boxes: {total_boxes}")
        for cls_id in sorted(total_counts):
            print(f"    {cls_names[cls_id]}: {total_counts[cls_id]}")
        if skipped_no_image:
            print(f"    Skipped (no image): {skipped_no_image}")

        grand_total_boxes += total_boxes
        grand_total_images += n_images
        for cls_id, cnt in total_counts.items():
            grand_total_counts[cls_id] = grand_total_counts.get(cls_id, 0) + cnt

    cls_names = {0: "red", 1: "yellow", 2: "green", 3: "redLeft", 4: "greenLeft", 5: "redRight", 6: "greenRight"}
    print(f"\n{'='*40}")
    print(f"S2TLD total (all subsets):")
    print(f"  Images: {grand_total_images}")
    print(f"  Total boxes: {grand_total_boxes}")
    for cls_id in sorted(grand_total_counts):
        print(f"  {cls_names[cls_id]}: {grand_total_counts[cls_id]}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
