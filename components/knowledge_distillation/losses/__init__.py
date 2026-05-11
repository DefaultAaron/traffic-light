"""KD loss modules — landed per-cell when scheduled.

Active loss functions:
    cls_logit_kl(student_logits, teacher_logits, temperature, pos_mask, reduction)
        — cls-logit KL with T scaling. Used by A2a (YOLO) and A2b (DEIM cls path).
    fdr_localization_kl(student_fdr_logits, teacher_fdr_logits, reg_max, ...)
        — LD on FDR distribution. Used by A2b (DEIM only; YOLO has no DFL).

Per-batch KD signals only. Multi-phase / multi-teacher / checkpoint-resolution
orchestration lives in the sibling ``schedules/`` package, not here.

Deferred modules (added when their cells are scheduled):
    pkd.py             — Pearson Correlation feature KD (A3, A4, A6 γ path)
    mgd.py             — Masked Generative Distillation (A6 backup)
    projection_mlp.py  — 1×1 conv channel projection for cross-arch feature alignment
                         (A6 γ path; removed at TRT export — training-only auxiliary)
    kd_weight_ramp.py  — per-step/epoch scalar KD-loss-weight schedule
                         (warmup ramp 0 → full over Stage-1; the actual
                         Stage-0/1/2 PHASE transitions belong in schedules/,
                         this module only owns the loss-weight scalar)
"""

from .cls_logit_kl import cls_logit_kl
from .fdr_localization_kl import fdr_localization_kl

__all__ = ["cls_logit_kl", "fdr_localization_kl"]

