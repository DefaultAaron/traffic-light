"""Merge converted datasets and create stratified train/val split (7-class).

Reads YOLO labels from each dataset's yolo_labels/ directory,
copies images + labels into data/merged/{images,labels}/{train,val}/
with dataset prefix to avoid filename collisions.

Usage: python scripts/merge_datasets.py [--val-ratio 0.2] [--seed 42]
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MERGED_DIR = ROOT / "data" / "merged"

# (dataset_prefix, label_dir, image_dir, image_extensions)
DATASETS = [
    (
        "s2tld",
        RAW_DIR / "S2TLD" / "yolo_labels",
        None,  # images resolved via find_s2tld_image (original + normal_1 + normal_2)
        [".jpg"],
    ),
    (
        "bstld",
        RAW_DIR / "BSTLD" / "yolo_labels",
        None,  # images resolved separately due to train/test split
        [".png"],
    ),
    (
        "lisa",
        RAW_DIR / "LISA" / "yolo_labels",
        None,  # images resolved separately due to nested structure
        [".jpg"],
    ),
]


def build_image_index() -> dict[str, dict[str, Path]]:
    """Build a stem→path index for all datasets (once, upfront).

    Returns dict keyed by dataset prefix, each value is {stem: Path}.
    """
    index: dict[str, dict[str, Path]] = {}

    # S2TLD: original + normal_1 + normal_2
    s2tld_dir = RAW_DIR / "S2TLD"
    s2tld_idx: dict[str, Path] = {}
    # Original (stems are timestamps with spaces)
    for img in (s2tld_dir / "JPEGImages").glob("*.jpg"):
        s2tld_idx[img.stem] = img
    # normal_1 → prefix with "normal1_"
    normal1_dir = s2tld_dir / "normal_1" / "JPEGImages"
    if normal1_dir.exists():
        for img in normal1_dir.glob("*.jpg"):
            s2tld_idx[f"normal1_{img.stem}"] = img
    # normal_2 → prefix with "normal2_"
    normal2_dir = s2tld_dir / "normal_2" / "JPEGImages"
    if normal2_dir.exists():
        for img in normal2_dir.glob("*.jpg"):
            s2tld_idx[f"normal2_{img.stem}"] = img
    index["s2tld"] = s2tld_idx
    print(f"  s2tld image index: {len(s2tld_idx)} images")

    # BSTLD: train/test splits, label stems are {split}_{original_stem}
    bstld_dir = RAW_DIR / "BSTLD"
    bstld_idx: dict[str, Path] = {}
    for split in ("train", "test"):
        split_dir = bstld_dir / split / "rgb" / split
        if split_dir.exists():
            for img in split_dir.rglob("*.png"):
                bstld_idx[f"{split}_{img.stem}"] = img
    index["bstld"] = bstld_idx
    print(f"  bstld image index: {len(bstld_idx)} images")

    # LISA: all .jpg in frames dirs, excluding sample-*
    lisa_dir = RAW_DIR / "LISA"
    lisa_idx: dict[str, Path] = {}
    if lisa_dir.exists():
        for img in lisa_dir.rglob("*.jpg"):
            if "sample-" not in str(img):
                lisa_idx[img.stem] = img
    index["lisa"] = lisa_idx
    print(f"  lisa image index: {len(lisa_idx)} images")

    return index


def get_dominant_class(label_path: Path) -> int | None:
    """Get the most common class in a label file for stratification."""
    text = label_path.read_text().strip()
    if not text:
        return None
    classes = [int(line.split()[0]) for line in text.split("\n") if line.strip()]
    if not classes:
        return None
    return Counter(classes).most_common(1)[0][0]


def collect_pairs(image_index: dict[str, dict[str, Path]]) -> list[tuple[str, Path, Path, int | None]]:
    """Collect all (prefixed_stem, image_path, label_path, dominant_class) pairs."""
    pairs = []

    for prefix, label_dir, image_dir, _ in DATASETS:
        if not label_dir.exists():
            print(f"  WARNING: {label_dir} not found, skipping {prefix}")
            continue

        idx = image_index.get(prefix, {})
        label_files = sorted(label_dir.glob("*.txt"))
        found = 0
        missing = 0

        for label_path in label_files:
            stem = label_path.stem
            img_path = idx.get(stem)
            if img_path is None:
                missing += 1
                continue

            dominant_class = get_dominant_class(label_path)
            prefixed_stem = f"{prefix}_{stem}"
            pairs.append((prefixed_stem, img_path, label_path, dominant_class))
            found += 1

        print(f"  {prefix}: {found} pairs collected, {missing} missing images")

    return pairs


def stratified_split(
    pairs: list[tuple[str, Path, Path, int | None]],
    val_ratio: float,
    seed: int,
) -> tuple[list, list]:
    """Split pairs into train/val with stratification by dominant class."""
    rng = random.Random(seed)

    # Group by dominant class
    by_class: dict[int | None, list] = {}
    for pair in pairs:
        cls = pair[3]
        by_class.setdefault(cls, []).append(pair)

    train_pairs = []
    val_pairs = []

    for cls, cls_pairs in by_class.items():
        rng.shuffle(cls_pairs)
        n_val = max(1, int(len(cls_pairs) * val_ratio))
        val_pairs.extend(cls_pairs[:n_val])
        train_pairs.extend(cls_pairs[n_val:])

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs


def copy_pairs(pairs: list[tuple[str, Path, Path, int | None]], split: str):
    """Copy image and label files to merged directory."""
    img_dir = MERGED_DIR / "images" / split
    lbl_dir = MERGED_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for prefixed_stem, img_path, label_path, _ in tqdm(pairs, desc=f"Copying {split}"):
        suffix = img_path.suffix
        shutil.copy2(img_path, img_dir / f"{prefixed_stem}{suffix}")
        shutil.copy2(label_path, lbl_dir / f"{prefixed_stem}.txt")


def print_stats(train_pairs: list, val_pairs: list):
    """Print dataset statistics."""
    def count_classes(pairs):
        counts = Counter()
        for _, _, label_path, _ in pairs:
            text = label_path.read_text().strip()
            for line in text.split("\n"):
                if line.strip():
                    counts[int(line.split()[0])] += 1
        return counts

    cls_names = {
        0: "red", 1: "yellow", 2: "green",
        3: "redLeft", 4: "greenLeft",
        5: "redRight", 6: "greenRight",
    }

    train_counts = count_classes(train_pairs)
    val_counts = count_classes(val_pairs)
    total_counts = train_counts + val_counts

    print(f"\n{'='*50}")
    print(f"Merged dataset statistics:")
    print(f"{'='*50}")
    print(f"  Total images: {len(train_pairs) + len(val_pairs)}")
    print(f"  Train images: {len(train_pairs)}")
    print(f"  Val images:   {len(val_pairs)}")
    print()
    print(f"  {'Class':<14} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'-'*40}")
    for cls_id in sorted(total_counts):
        name = cls_names.get(cls_id, f"cls_{cls_id}")
        print(f"  {name:<14} {train_counts[cls_id]:>8} {val_counts[cls_id]:>8} {total_counts[cls_id]:>8}")
    print(f"  {'-'*40}")
    print(f"  {'total':<14} {sum(train_counts.values()):>8} {sum(val_counts.values()):>8} {sum(total_counts.values()):>8}")
    print(f"\n  Output: {MERGED_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Merge datasets with stratified split")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Clean previous merge
    if MERGED_DIR.exists():
        print(f"Removing previous merged data at {MERGED_DIR}")
        shutil.rmtree(MERGED_DIR)

    print("Building image index ...")
    image_index = build_image_index()

    print("Collecting image-label pairs ...")
    pairs = collect_pairs(image_index)
    print(f"\nTotal pairs: {len(pairs)}")

    if not pairs:
        print("ERROR: No pairs found. Run conversion scripts first.")
        return

    print(f"Splitting {args.val_ratio:.0%} validation (seed={args.seed}) ...")
    train_pairs, val_pairs = stratified_split(pairs, args.val_ratio, args.seed)

    print(f"Copying to {MERGED_DIR} ...")
    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    print_stats(train_pairs, val_pairs)


if __name__ == "__main__":
    main()
