"""Convert LISA dataset from CSV to YOLO format.

Source: data/raw/LISA/Annotations/Annotations/**/frameAnnotationsBOX.csv
Output: data/raw/LISA/yolo_labels/*.txt
"""

import csv
from pathlib import Path

from PIL import Image

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "LISA"
ANNOTATIONS_DIR = RAW_DIR / "Annotations" / "Annotations"
OUTPUT_DIR = RAW_DIR / "yolo_labels"

CLASS_MAP = {
    "stop": 0,
    "stopLeft": 0,
    "warning": 1,
    "warningLeft": 1,
    "go": 2,
    "goLeft": 2,
    "goForward": 2,
}

# CSV path prefix → actual directory mapping
# CSV: "dayTraining/dayClip1--00000.jpg" → actual: "dayTrain/dayTrain/dayClip1/frames/dayClip1--00000.jpg"
PATH_PREFIX_MAP = {
    "dayTraining": "dayTrain/dayTrain",
    "nightTraining": "nightTrain/nightTrain",
    "dayTest": None,       # sequence folders are top-level
    "nightTest": None,     # sequence folders are top-level
}


def resolve_image_path(csv_filename: str) -> Path | None:
    """Map CSV filename field to actual image path on disk.

    CSV patterns:
      dayTraining/dayClip1--00000.jpg  → dayTrain/dayTrain/dayClip1/frames/dayClip1--00000.jpg
      nightTraining/nightClip2--00000.jpg → nightTrain/nightTrain/nightClip2/frames/nightClip2--00000.jpg
      dayTest/daySequence1--00000.jpg  → daySequence1/daySequence1/frames/daySequence1--00000.jpg
      nightTest/nightSequence2--00000.jpg → nightSequence2/nightSequence2/frames/nightSequence2--00000.jpg
    """
    parts = csv_filename.split("/")
    if len(parts) != 2:
        return None

    prefix, img_name = parts
    # Extract clip/sequence name from image filename: "dayClip1--00000.jpg" → "dayClip1"
    clip_name = img_name.split("--")[0]

    if prefix in ("dayTraining", "nightTraining"):
        mapped_prefix = PATH_PREFIX_MAP[prefix]
        candidate = RAW_DIR / mapped_prefix / clip_name / "frames" / img_name
    else:
        # dayTest/nightTest → sequence is a top-level folder
        candidate = RAW_DIR / clip_name / clip_name / "frames" / img_name

    if candidate.exists():
        return candidate
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(ANNOTATIONS_DIR.rglob("frameAnnotationsBOX.csv"))
    print(f"LISA: found {len(csv_files)} annotation CSVs")

    # Collect all annotations grouped by image
    # key: resolved image path, value: list of (cls_id, xmin, ymin, xmax, ymax)
    image_annotations: dict[Path, list[tuple[int, float, float, float, float]]] = {}
    unknown_classes: dict[str, int] = {}
    missing_images = 0
    total_rows = 0

    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                total_rows += 1
                tag = row["Annotation tag"].strip()
                cls_id = CLASS_MAP.get(tag)
                if cls_id is None:
                    unknown_classes[tag] = unknown_classes.get(tag, 0) + 1
                    continue

                csv_filename = row["Filename"].strip()
                img_path = resolve_image_path(csv_filename)
                if img_path is None:
                    missing_images += 1
                    continue

                xmin = float(row["Upper left corner X"])
                ymin = float(row["Upper left corner Y"])
                xmax = float(row["Lower right corner X"])
                ymax = float(row["Lower right corner Y"])

                image_annotations.setdefault(img_path, []).append(
                    (cls_id, xmin, ymin, xmax, ymax)
                )

    # Write YOLO labels — need image dimensions for normalization
    print(f"  Processing {len(image_annotations)} images ...")
    total_boxes = 0
    total_counts: dict[int, int] = {}
    size_cache: dict[Path, tuple[int, int]] = {}

    for img_path, annotations in image_annotations.items():
        # Get image dimensions (cached)
        if img_path not in size_cache:
            with Image.open(img_path) as img:
                size_cache[img_path] = img.size  # (width, height)
        w_img, h_img = size_cache[img_path]

        lines = []
        for cls_id, xmin, ymin, xmax, ymax in annotations:
            cx = (xmin + xmax) / 2.0 / w_img
            cy = (ymin + ymax) / 2.0 / h_img
            w = (xmax - xmin) / w_img
            h = (ymax - ymin) / h_img
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_counts[cls_id] = total_counts.get(cls_id, 0) + 1

        # Use image stem as label filename (already unique across clips)
        out_path = OUTPUT_DIR / f"{img_path.stem}.txt"
        out_path.write_text("\n".join(lines) + "\n")
        total_boxes += len(lines)

    cls_names = {0: "red", 1: "yellow", 2: "green"}
    print(f"\nLISA conversion complete:")
    print(f"  CSV rows processed: {total_rows}")
    print(f"  Images: {len(image_annotations)}")
    print(f"  Total boxes: {total_boxes}")
    for cls_id in sorted(total_counts):
        print(f"  {cls_names[cls_id]}: {total_counts[cls_id]}")
    if missing_images:
        print(f"  Rows with missing images: {missing_images}")
    if unknown_classes:
        print(f"  Unknown classes skipped: {unknown_classes}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
