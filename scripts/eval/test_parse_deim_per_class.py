"""Pytest-style fixtures + tests for scripts/eval/_parse_deim_per_class.py.

Synthetic eval pickle (no need to commit a 2.7 MB real eval.pth) +
synthetic COCO annotation JSON exercise the parse paths consumed by the
parser:
  - per-class AP@0.50 / AP@0.50:0.95 from the precision tensor
  - best-F1 P / R from the IoU=0.50 PR curve
  - image-count + instance-count from COCO annotations
  - overall row = uniform mean across classes with insts > 0

Note: the parser does NOT consume `eval_dict["recall"]`; the synthetic
recall tensor in `make_synthetic_eval` is filled with -1 only to match
the shape pycocotools writes. This is by design — parser semantics
match the existing R1 results-doc convention.

Environment: requires `numpy`, `torch`, `pyyaml` (provided by the project
uv venv at `.venv/`; bootstrap with `uv sync` or `.venv/bin/python -m pip
install numpy torch pyyaml` if running outside the project venv).

Run via project venv: `.venv/bin/python scripts/eval/test_parse_deim_per_class.py`
Or pytest: `.venv/bin/python -m pytest scripts/eval/test_parse_deim_per_class.py -v`
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

THIS = Path(__file__).resolve()
PARSER = THIS.parent / "_parse_deim_per_class.py"


def make_synthetic_eval(num_classes: int = 3) -> dict:
    """Build a minimal COCOeval-shaped eval dict.

    Shape rules pinned in pycocotools / faster_coco_eval:
      precision : [T=10, R=101, K, A=4, M=3]
      recall    : [T=10, K, A=4, M=3]
    Indexing in our parser: T0=0 (IoU=0.50), A0=0 (area=all), M2=2 (maxDets=100).

    For class 0: precision = 1.0 at all recall points → AP=1.0, best-F1 P=1.0 R=1.0.
    For class 1: precision drops from 1.0 to 0.0 linearly with recall → AP≈0.505.
    For class 2: all -1 (no annotations) → AP=0.0, best-F1 P=0 R=0.
    """
    T, R, A, M = 10, 101, 4, 3
    P = np.full((T, R, num_classes, A, M), -1.0, dtype=np.float64)
    R_arr = np.full((T, num_classes, A, M), -1.0, dtype=np.float64)
    P[0, :, 0, 0, 2] = 1.0
    P[:, :, 0, 0, 2] = 1.0
    recall_axis = np.linspace(0.0, 1.0, R)
    P[0, :, 1, 0, 2] = 1.0 - recall_axis
    P[:, :, 1, 0, 2] = 1.0 - recall_axis[None, :]
    return {"precision": P, "recall": R_arr,
            "params": None, "counts": None, "scores": None}


def make_synthetic_coco(class_counts: list[tuple[int, int]]) -> dict:
    """COCO JSON with `class_counts[k] = (n_images, n_instances)` for class k."""
    images: list[dict] = []
    annotations: list[dict] = []
    next_img_id = 1
    next_ann_id = 1
    for cid, (n_img, n_inst) in enumerate(class_counts):
        img_ids_this_class = list(range(next_img_id, next_img_id + n_img))
        for img_id in img_ids_this_class:
            images.append({"id": img_id, "file_name": f"{img_id}.jpg",
                           "width": 1280, "height": 720})
        for i in range(n_inst):
            ann_img = img_ids_this_class[i % max(n_img, 1)] if n_img else next_img_id
            annotations.append({"id": next_ann_id, "image_id": ann_img,
                                "category_id": cid, "bbox": [0, 0, 10, 10],
                                "area": 100, "iscrowd": 0})
            next_ann_id += 1
        next_img_id += n_img
    return {"images": images, "annotations": annotations,
            "categories": [{"id": k, "name": f"cls{k}"} for k in range(len(class_counts))]}


def run_parser(eval_dict: dict, coco_ann: dict, names: list[str], tmp: Path) -> dict:
    eval_pth = tmp / "eval.pth"
    ann_json = tmp / "ann.json"
    data_yaml = tmp / "data.yaml"
    out_json = tmp / "out.json"
    out_txt = tmp / "out.txt"
    torch.save(eval_dict, eval_pth)
    ann_json.write_text(json.dumps(coco_ann))
    data_yaml.write_text(yaml.safe_dump({"names": {i: n for i, n in enumerate(names)}}))
    r = subprocess.run([sys.executable, str(PARSER),
                        "--eval-pth", str(eval_pth),
                        "--ann-json", str(ann_json),
                        "--data-yaml", str(data_yaml),
                        "--out-json", str(out_json),
                        "--out-txt", str(out_txt)],
                       capture_output=True, text=True, check=True)
    assert "[OK]" in r.stdout, f"parser stdout: {r.stdout}\nstderr: {r.stderr}"
    return json.loads(out_json.read_text())


def test_per_class_ap_and_pr(tmp_path: Path) -> None:
    names = ["cls0", "cls1", "cls2"]
    eval_dict = make_synthetic_eval(num_classes=len(names))
    coco = make_synthetic_coco([(10, 20), (5, 8), (0, 0)])
    out = run_parser(eval_dict, coco, names, tmp_path)
    per_class = {row["class"]: row for row in out["per_class"]}
    assert abs(per_class["cls0"]["ap50"] - 1.0) < 1e-9
    assert abs(per_class["cls0"]["ap5095"] - 1.0) < 1e-9
    assert abs(per_class["cls0"]["p_bestf1"] - 1.0) < 1e-9
    assert abs(per_class["cls0"]["r_bestf1"] - 1.0) < 1e-9
    assert per_class["cls0"]["imgs"] == 10
    assert per_class["cls0"]["insts"] == 20
    ap50_cls1 = per_class["cls1"]["ap50"]
    assert 0.49 < ap50_cls1 < 0.51, f"cls1 ap50={ap50_cls1}, expected ~0.505"
    assert abs(per_class["cls1"]["p_bestf1"] - 0.5) < 1e-3
    assert abs(per_class["cls1"]["r_bestf1"] - 0.5) < 1e-3
    assert per_class["cls1"]["imgs"] == 5
    assert per_class["cls1"]["insts"] == 8
    assert per_class["cls2"]["ap50"] == 0.0
    assert per_class["cls2"]["p_bestf1"] == 0.0
    assert per_class["cls2"]["r_bestf1"] == 0.0
    assert per_class["cls2"]["imgs"] == 0
    assert per_class["cls2"]["insts"] == 0


def test_overall_uniform_mean(tmp_path: Path) -> None:
    names = ["a", "b", "c"]
    eval_dict = make_synthetic_eval(num_classes=len(names))
    coco = make_synthetic_coco([(10, 20), (5, 8), (3, 6)])
    out = run_parser(eval_dict, coco, names, tmp_path)
    per_class = {row["class"]: row for row in out["per_class"]}
    expected_ap50 = (per_class["a"]["ap50"] + per_class["b"]["ap50"] + per_class["c"]["ap50"]) / 3
    assert abs(out["overall"]["ap50"] - expected_ap50) < 1e-9, \
        f"overall ap50 should be uniform mean of valid (insts>0) classes"
    assert out["overall"]["imgs"] == len(coco["images"])
    assert out["overall"]["insts"] == 20 + 8 + 6


def test_zero_classes_excluded_from_overall(tmp_path: Path) -> None:
    """Classes with insts=0 contribute 0.0 to overall mean by the divisor logic
    in _parse_deim_per_class.py (valid_classes counter); verify by comparison."""
    names = ["a", "b"]
    eval_dict = make_synthetic_eval(num_classes=len(names))
    coco = make_synthetic_coco([(10, 20), (0, 0)])
    out = run_parser(eval_dict, coco, names, tmp_path)
    per_class = {row["class"]: row for row in out["per_class"]}
    assert per_class["b"]["insts"] == 0
    assert abs(out["overall"]["ap50"] - per_class["a"]["ap50"]) < 1e-9, \
        "overall should equal cls_a alone (cls_b excluded by insts>0 gate)"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_per_class_ap_and_pr(tmp)
        print("[OK] test_per_class_ap_and_pr")
        test_overall_uniform_mean(tmp)
        print("[OK] test_overall_uniform_mean")
        test_zero_classes_excluded_from_overall(tmp)
        print("[OK] test_zero_classes_excluded_from_overall")
        print("\nAll 3 tests passed.")
