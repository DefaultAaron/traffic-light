"""Localization Distillation (LD) on FDR distribution — KD cell A2b (DEIM family).

DEIM-D-FINE's FDR head emits `4 * (reg_max + 1)` logits per query that softmax
into 4 corner-distance discrete distributions. LD distills the teacher's FDR
distribution into the student's via KL with optional temperature, applied
per-query × per-side.

YOLO26 has `reg_max = 1` (no DFL head) — this loss is NOT applicable on YOLO
students. Use `cls_logit_kl` instead for YOLO KD (cell A2a).

Shapes:
    student_fdr_logits, teacher_fdr_logits:
        (B, L, 4 * (reg_max + 1))   — DEIM regression head raw output
    pos_mask (optional):
        (B, L)                       — per-query mask (Hungarian-matched positives)

Math (per-query, per-side):
    KL_T(s || t) = T² · Σ softmax(t/T) · [log softmax(t/T) - log softmax(s/T)]

Reduction: mean over (matched-positions × 4 sides) by default.

Spec: docs/planning/additional_components_plan.md §七 cell A2b
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fdr_localization_kl(
    student_fdr_logits: torch.Tensor,
    teacher_fdr_logits: torch.Tensor,
    reg_max: int,
    temperature: float = 1.0,
    pos_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """LD on FDR — KL divergence over `reg_max + 1` bins, per query × per side.

    Args:
        student_fdr_logits: (B, L, 4*(reg_max+1)) DEIM student FDR logits.
        teacher_fdr_logits: (B, L, 4*(reg_max+1)) DEIM teacher FDR logits (no_grad).
        reg_max: max discrete bin index (DEIM default 32 → 33 bins per side).
        temperature: softmax T (T > 0).
        pos_mask: (B, L) selecting Hungarian-matched positives. If None, all queries.
        reduction: "mean" | "sum" | "none".

    Returns:
        Scalar (mean/sum) or (B, L, 4) tensor (reduction="none").
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if reg_max < 1:
        raise ValueError(f"reg_max must be ≥ 1, got {reg_max}")
    if student_fdr_logits.shape != teacher_fdr_logits.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_fdr_logits.shape)} vs "
            f"teacher {tuple(teacher_fdr_logits.shape)}"
        )
    expected_last = 4 * (reg_max + 1)
    if student_fdr_logits.shape[-1] != expected_last:
        raise ValueError(
            f"last dim must be 4*(reg_max+1)={expected_last}, "
            f"got {student_fdr_logits.shape[-1]} for reg_max={reg_max}"
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be one of mean/sum/none, got {reduction!r}")

    B, L, _ = student_fdr_logits.shape
    T = float(temperature)

    # Reshape (B, L, 4*(reg_max+1)) → (B, L, 4, reg_max+1)
    student = student_fdr_logits.reshape(B, L, 4, reg_max + 1)
    teacher = teacher_fdr_logits.reshape(B, L, 4, reg_max + 1)

    log_student = F.log_softmax(student / T, dim=-1)
    teacher_p = F.softmax(teacher / T, dim=-1)

    per_bin = F.kl_div(log_student, teacher_p, reduction="none")  # (B, L, 4, reg_max+1)
    per_side = per_bin.sum(dim=-1) * (T * T)  # (B, L, 4)

    if pos_mask is not None:
        if pos_mask.shape != (B, L):
            raise ValueError(
                f"pos_mask shape must be (B, L)=({B}, {L}), got {tuple(pos_mask.shape)}"
            )
        mask = pos_mask.to(per_side.dtype).unsqueeze(-1)  # (B, L, 1) → broadcast to sides
        per_side = per_side * mask
        denom = (mask.sum() * 4.0).clamp(min=1.0)
    else:
        denom = torch.tensor(float(per_side.numel()), device=per_side.device, dtype=per_side.dtype)

    if reduction == "none":
        return per_side
    if reduction == "sum":
        return per_side.sum()
    return per_side.sum() / denom
