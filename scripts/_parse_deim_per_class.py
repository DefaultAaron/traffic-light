"""Parse DEIM eval.pth (pickled COCOeval.eval dict) into a per-class
AP / P / R table aligned with the existing DEIM-S/M results-doc format.

P/R taken at best-F1 operating point on the IoU=0.5 PR curve (matches the
existing footnote in docs/reports/phase_2_round_1_results.md).
overall row = uniform mean across the 7 classes (same convention).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml


def load_class_names(yaml_path: Path) -> list[str]:
    cfg = yaml.safe_load(yaml_path.read_text())
    names = cfg["names"]
    return [names[i] for i in sorted(names)]


def per_class_from_eval(eval_dict: dict, k: int) -> dict:
    P = eval_dict["precision"]   # [T, R, K, A, M]
    T0, A0, M2 = 0, 0, 2          # IoU=0.50, area=all, maxDets=100

    pr_iou50 = P[T0, :, k, A0, M2]
    valid = pr_iou50 > -1
    ap50 = float(pr_iou50[valid].mean()) if valid.any() else 0.0

    pr_all = P[:, :, k, A0, M2]
    valid = pr_all > -1
    ap5095 = float(pr_all[valid].mean()) if valid.any() else 0.0

    p_curve = P[T0, :, k, A0, M2]
    r_curve = np.linspace(0.0, 1.0, 101)
    mask = p_curve > -1
    if not mask.any():
        return {"ap50": ap50, "ap5095": ap5095, "p_bestf1": 0.0, "r_bestf1": 0.0}
    pc = p_curve[mask]
    rc = r_curve[mask]
    f1 = np.where((pc + rc) > 0, 2 * pc * rc / (pc + rc), 0.0)
    j = int(f1.argmax())
    return {"ap50": ap50, "ap5095": ap5095,
            "p_bestf1": float(pc[j]), "r_bestf1": float(rc[j])}


def count_per_class(coco_ann: dict, num_classes: int) -> dict:
    inst_counter: Counter = Counter()
    img_per_class: dict[int, set] = {i: set() for i in range(num_classes)}
    for ann in coco_ann["annotations"]:
        cid = ann["category_id"]
        if cid in img_per_class:
            inst_counter[cid] += 1
            img_per_class[cid].add(ann["image_id"])
    return {i: {"imgs": len(img_per_class[i]), "insts": inst_counter[i]}
            for i in range(num_classes)}


def fmt_row(row: dict) -> str:
    return (f"| {row['class']:<10} | {row['imgs']:>6,} | {row['insts']:>6,} "
            f"| {row['p_bestf1']:.3f}  | {row['r_bestf1']:.3f}  "
            f"| {row['ap50']:.3f}  | {row['ap5095']:.3f}     |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-pth", required=True, type=Path)
    ap.add_argument("--ann-json", required=True, type=Path)
    ap.add_argument("--data-yaml", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-txt", required=True, type=Path)
    args = ap.parse_args()

    eval_dict = torch.load(args.eval_pth, map_location="cpu", weights_only=False)
    coco_ann = json.loads(args.ann_json.read_text())
    names = load_class_names(args.data_yaml)
    counts = count_per_class(coco_ann, len(names))

    rows: list[dict] = []
    ap50_sum = ap5095_sum = p_sum = r_sum = 0.0
    valid_classes = 0
    for k, name in enumerate(names):
        m = per_class_from_eval(eval_dict, k)
        c = counts[k]
        rows.append({"class": name, "imgs": c["imgs"], "insts": c["insts"],
                     "p_bestf1": m["p_bestf1"], "r_bestf1": m["r_bestf1"],
                     "ap50": m["ap50"], "ap5095": m["ap5095"]})
        if c["insts"] > 0:
            ap50_sum += m["ap50"]
            ap5095_sum += m["ap5095"]
            p_sum += m["p_bestf1"]
            r_sum += m["r_bestf1"]
            valid_classes += 1

    n = max(valid_classes, 1)
    overall = {"class": "all", "imgs": len(coco_ann["images"]),
               "insts": sum(c["insts"] for c in counts.values()),
               "p_bestf1": p_sum / n, "r_bestf1": r_sum / n,
               "ap50": ap50_sum / n, "ap5095": ap5095_sum / n}

    out = {"overall": overall, "per_class": rows,
           "source_eval_pth": str(args.eval_pth),
           "source_ann_json": str(args.ann_json)}
    args.out_json.write_text(json.dumps(out, indent=2))

    lines = [
        "| 类别       | 图片数 | 实例数 | 准确率 | 召回率 | mAP@50 | mAP@50:95 |",
        "| ---------- | ------ | ------ | ------ | ------ | ------ | --------- |",
        fmt_row(overall),
    ]
    lines.extend(fmt_row(r) for r in rows)
    args.out_txt.write_text("\n".join(lines) + "\n")
    print(f"[OK] {args.out_json}\n[OK] {args.out_txt}")


if __name__ == "__main__":
    main()
