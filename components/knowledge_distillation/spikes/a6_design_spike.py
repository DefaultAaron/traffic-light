"""A6 cross-arch KD design spike — DEIM-D-FINE-M (teacher) → YOLO26-s (student).

Pre-R2 R1 rehearsal scoped by `docs/_archive/pre_r2_kickoff_checklist.md (2026-05-12 归档)` §2.5 row "KD A6 R1 spike".
GPU-free; 1-batch synthetic-fixture sanity probe.

Spike answers (design doc: `docs/planning/kd_a6_design_spike.md`):
  - YOLO26 student head structure (does DFL exist?)
  - DEIM FDR teacher head structure (from source analysis, no DEIM ckpt needed)
  - PKD FPN projection conv shape compatibility
  - DEIM Integral collapse shape compatibility (FDR → 4-coord)
  - Path recommendation: β (aux DFL head) / γ (Integral collapse) / δ (drop distribution)

Output:
  - `runs/rehearsal_kd_A6_design_spike.json` (rehearsal_kind = synthetic_fixture)

Run:
  uv run python -m components.knowledge_distillation.spikes.a6_design_spike \
      [--yolo-ckpt runs/detect/yolo26s-r1/weights/best.pt] \
      [--output runs/rehearsal_kd_A6_design_spike.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SPIKE_VERSION = "1.0"
SPIKE_KIND = "synthetic_fixture"
DEFAULT_YOLO_CKPT = "runs/detect/yolo26s-r1/weights/best.pt"
DEFAULT_OUTPUT = "runs/rehearsal_kd_A6_design_spike.json"

# DEIM-D-FINE-M static facts derived from source:
#   DEIM/engine/deim/dfine_decoder.py:506-507 (MLP regression head)
#   DEIM/configs/base/dfine_hgnetv2.yml:61    (reg_max=32)
DEIM_REG_MAX = 32
DEIM_HIDDEN_DIM = 256
DEIM_NUM_CLASSES = 7
DEIM_FDR_LOGITS_PER_QUERY = 4 * (DEIM_REG_MAX + 1)  # 132
DEIM_NUM_QUERIES = 300


def probe_yolo_head(ckpt_path: Path) -> dict:
    """Load YOLO26-s ckpt and read Detect-head shape facts."""
    from ultralytics import YOLO  # local import — ultralytics is in uv env only

    m = YOLO(str(ckpt_path))
    h = m.model.model[-1]
    # FPN feature channels: read in_channels of the first conv in each cv2 block
    # (cv2[i][0] consumes the FPN feature for scale i; cv2[i][-1] is the box-output proj)
    def _first_in_channels(block):
        for mod in block.modules():
            if isinstance(mod, nn.Conv2d):
                return int(mod.in_channels)
        raise RuntimeError("no Conv2d found in cv2 block")
    fpn_channels = [_first_in_channels(b) for b in h.cv2]
    head_stem_in_channels = [int(b[-1].in_channels) for b in h.cv2]
    return {
        "head_class": type(h).__name__,
        "reg_max": int(h.reg_max),
        "nc": int(h.nc),
        "no": int(h.no),
        "stride": [int(s) for s in h.stride.tolist()],
        "fpn_channels": fpn_channels,  # FPN feature channels feeding the head (P3/P4/P5)
        "head_stem_in_channels": head_stem_in_channels,  # internal head stem dim
        "cv2_out_channels": [int(b[-1].out_channels) for b in h.cv2],
        "cv3_out_channels": [int(b[-1].out_channels) for b in h.cv3],
        "has_dfl_head": int(h.reg_max) > 1,
    }


def check_pkd_projection_conv(
    student_channels: list[int],
    teacher_channels: int,
    imgsz: int = 640,
    strides: tuple[int, int, int] = (8, 16, 32),
) -> dict:
    """Build per-scale 1×1 Conv2d(C_s -> 256) and verify shape on simulated FPN."""
    feats_s = [
        torch.randn(1, c, imgsz // s, imgsz // s) for c, s in zip(student_channels, strides)
    ]
    feats_t = [
        torch.randn(1, teacher_channels, imgsz // s, imgsz // s) for s in strides
    ]
    proj_convs = [nn.Conv2d(c, teacher_channels, kernel_size=1) for c in student_channels]
    out_shapes = []
    losses = []
    for fs, ft, conv in zip(feats_s, feats_t, proj_convs):
        proj = conv(fs)
        if proj.shape != ft.shape:
            raise RuntimeError(
                f"projection shape mismatch: student {tuple(proj.shape)} vs teacher {tuple(ft.shape)}"
            )
        out_shapes.append(tuple(proj.shape))
        losses.append(F.mse_loss(proj, ft).item())
    return {
        "student_channels": student_channels,
        "teacher_channels": teacher_channels,
        "strides": list(strides),
        "imgsz": imgsz,
        "projected_shapes": [list(s) for s in out_shapes],
        "per_scale_mse": losses,
        "shape_check": "pass",
    }


def check_integral_collapse(
    reg_max: int = DEIM_REG_MAX,
    num_queries: int = DEIM_NUM_QUERIES,
) -> dict:
    """Mirror DEIM Integral (dfine_decoder.py:246-268) and verify shape.

    Forward path: FDR logits (B, L, 4*(reg_max+1))
      -> reshape (B*L, 4, reg_max+1)
      -> softmax over last dim
      -> dot with non-uniform project weights (shape reg_max+1)
      -> reshape (B, L, 4) box coords

    A spike does NOT need DEIM's exact weighting_function. We use a uniform
    increasing weight 0..reg_max (linear; DEIM uses a non-uniform learned one,
    but shape/numerical behavior is equivalent for the shape check).
    """
    B, L = 1, num_queries
    fdr_logits = torch.randn(B, L, 4 * (reg_max + 1))
    project = torch.linspace(0.0, float(reg_max), reg_max + 1)

    shape_in = fdr_logits.shape
    x = F.softmax(fdr_logits.reshape(-1, reg_max + 1), dim=1)
    x = F.linear(x, project.unsqueeze(0)).reshape(-1, 4)
    box_coords = x.reshape(list(shape_in[:-1]) + [-1])

    if tuple(box_coords.shape) != (B, L, 4):
        raise RuntimeError(
            f"Integral collapse shape mismatch: got {tuple(box_coords.shape)}, expected {(B, L, 4)}"
        )
    return {
        "reg_max": reg_max,
        "fdr_logits_per_query": 4 * (reg_max + 1),
        "input_shape": list(shape_in),
        "output_shape": list(box_coords.shape),
        "output_dtype": str(box_coords.dtype),
        "output_finite": bool(torch.isfinite(box_coords).all().item()),
        "shape_check": "pass",
    }


def recommend_path(
    yolo_head: dict, projection: dict, integral: dict
) -> tuple[str, str, str]:
    """Apply pre-committed selection rule from §七 + design doc §七.

    Returns (selected_path, priority, rationale).
    """
    if yolo_head["has_dfl_head"]:
        return (
            "beta_with_native_dfl",
            "P1_with_path_beta",
            "YOLO26 student has native DFL head — direct FDR↔DFL alignment is viable.",
        )
    if projection["shape_check"] == "pass" and integral["shape_check"] == "pass":
        return (
            "gamma",
            "P1_with_path_gamma",
            (
                "YOLO26 student has no DFL head (reg_max=1). Path γ (Integral collapse on teacher"
                " FDR → L1 + GIoU bbox KD + cls-logit KL + PKD FPN projection) is shape-compatible"
                " and preserves DEIM long-tail signal via Hungarian-matched positives. Lowest-risk"
                " path consistent with A6's differentiation thesis. δ remains as fallback if"
                " 1-week PoC fails long-tail recall ≥ +5 pp criterion."
            ),
        )
    return (
        "blocked",
        "P2_demote",
        "Projection conv or Integral collapse shape check failed; A6 demoted to P2.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="A6 cross-arch KD design spike")
    ap.add_argument("--yolo-ckpt", default=DEFAULT_YOLO_CKPT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    yolo_ckpt = Path(args.yolo_ckpt)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "schema_version": SPIKE_VERSION,
        "rehearsal_kind": SPIKE_KIND,
        "spike_name": "kd_A6_design_spike",
        "spike_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yolo_ckpt": str(yolo_ckpt),
        "deim_dfine_m_static_facts": {
            "source": "DEIM/engine/deim/dfine_decoder.py:506-507 + DEIM/configs/base/dfine_hgnetv2.yml",
            "reg_max": DEIM_REG_MAX,
            "num_classes": DEIM_NUM_CLASSES,
            "hidden_dim": DEIM_HIDDEN_DIM,
            "fdr_logits_per_query": DEIM_FDR_LOGITS_PER_QUERY,
            "default_num_queries": DEIM_NUM_QUERIES,
        },
    }

    try:
        report["yolo26s_head"] = probe_yolo_head(yolo_ckpt)
        report["yolo_head_probe_status"] = "pass"
    except Exception as exc:
        report["yolo26s_head"] = None
        report["yolo_head_probe_status"] = "fail"
        report["yolo_head_probe_traceback"] = traceback.format_exc()

    if report.get("yolo_head_probe_status") == "pass":
        student_channels = report["yolo26s_head"]["fpn_channels"]
        strides = tuple(report["yolo26s_head"]["stride"])  # type: ignore[arg-type]
        try:
            report["pkd_projection_check"] = check_pkd_projection_conv(
                student_channels, DEIM_HIDDEN_DIM, strides=strides
            )
        except Exception:
            report["pkd_projection_check"] = {
                "shape_check": "fail",
                "traceback": traceback.format_exc(),
            }
    else:
        report["pkd_projection_check"] = {
            "shape_check": "skipped",
            "reason": "yolo_head_probe_failed",
        }

    try:
        report["integral_collapse_check"] = check_integral_collapse()
    except Exception:
        report["integral_collapse_check"] = {
            "shape_check": "fail",
            "traceback": traceback.format_exc(),
        }

    selected_path, priority, rationale = recommend_path(
        report.get("yolo26s_head") or {"has_dfl_head": False},
        report["pkd_projection_check"],
        report["integral_collapse_check"],
    )
    report["incompatibility_found"] = (
        None
        if (report.get("yolo26s_head") or {}).get("has_dfl_head")
        else "yolo26_no_dfl_head"
    )
    report["selected_path"] = selected_path
    report["a6_priority_recommendation"] = priority
    report["selected_path_rationale"] = rationale

    spike_pass = (
        report.get("yolo_head_probe_status") == "pass"
        and report["pkd_projection_check"].get("shape_check") == "pass"
        and report["integral_collapse_check"].get("shape_check") == "pass"
        and selected_path != "blocked"
    )
    report["spike_pass"] = spike_pass
    report["next_step"] = (
        "Begin 1-week γ-path PoC on R1 data: implement runner with Integral collapse"
        " bbox KD + cls-logit KL + PKD FPN projection; long-tail recall ≥ +5 pp gate."
        if spike_pass
        else "Demote A6 to P2; route alternatives to §七 retrospective."
    )

    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"A6 design spike: {'PASS' if spike_pass else 'FAIL'} → {output}")
    print(f"  selected_path = {selected_path}")
    print(f"  a6_priority   = {priority}")
    return 0 if spike_pass else 1


if __name__ == "__main__":
    sys.exit(main())
