"""Classification-logit KL divergence — KD cells A2a (YOLO) and A2b (DEIM).

Soft-target KL on per-anchor (YOLO) or per-query (DEIM) cls logits with
optional temperature T. Reduction is mean over the active position set; the
inactive (padding) positions are masked when a `pos_mask` is supplied.

Math:
    KL_T(s || t) = T² · Σ softmax(t/T) · [log softmax(t/T) - log softmax(s/T)]

The T² scaling restores the cls-loss-gradient magnitude after T-smoothing,
following Hinton et al. 2015. Multiply by `kd_lambda` at the caller site.

Shapes accepted:
    student_logits, teacher_logits: (..., num_classes) — last-dim aligned.
    pos_mask (optional): broadcastable to leading dims of `*_logits` (excl. last).
                          If None, every position contributes equally.

Spec: docs/planning/additional_components_plan.md §七 cells A2a / A2b
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cls_logit_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    pos_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL divergence on classification logits with optional temperature smoothing.

    Args:
        student_logits: (..., C) student raw logits (requires_grad=True).
        teacher_logits: (..., C) teacher raw logits (no_grad — caller responsibility).
        temperature: Softmax temperature T (T > 0). T=1 → standard KL.
        pos_mask: Boolean / 0-1 mask of shape (...) (no class dim) selecting
                  positions to include in the reduction. Broadcasts.
        reduction: "mean" | "sum" | "none". For "none", returns per-position scalar.

    Returns:
        Scalar (mean/sum) or per-position tensor.

    Notes:
        - `F.kl_div` requires log_softmax of the input (student) and
          softmax/probabilities of the target (teacher).
        - `F.kl_div(..., reduction="none")` returns per-element (after softmax)
          contribution; we sum over the class dim before reducing over positions.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_logits.shape)} vs teacher {tuple(teacher_logits.shape)}"
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be one of mean/sum/none, got {reduction!r}")

    T = float(temperature)
    log_student = F.log_softmax(student_logits / T, dim=-1)
    teacher = F.softmax(teacher_logits / T, dim=-1)

    # Per-element KL contribution; sum over class dim → per-position scalar.
    per_class = F.kl_div(log_student, teacher, reduction="none")  # (..., C)
    per_pos = per_class.sum(dim=-1) * (T * T)  # (...)

    if pos_mask is not None:
        if pos_mask.shape != per_pos.shape:
            try:
                pos_mask = pos_mask.expand_as(per_pos)
            except RuntimeError as exc:
                raise ValueError(
                    f"pos_mask shape {tuple(pos_mask.shape)} not broadcastable to "
                    f"per-position shape {tuple(per_pos.shape)}"
                ) from exc
        per_pos = per_pos * pos_mask.to(per_pos.dtype)
        denom = pos_mask.to(per_pos.dtype).sum().clamp(min=1.0)
    else:
        denom = torch.tensor(float(per_pos.numel()), device=per_pos.device, dtype=per_pos.dtype)

    if reduction == "none":
        return per_pos
    if reduction == "sum":
        return per_pos.sum()
    return per_pos.sum() / denom
