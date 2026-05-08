"""KD loss modules — landed per-cell when scheduled.

Planned modules (per v1.2 §5.2 / §5.3):
    cls_logit_kl.py    — classification logit KL (A2a, A2b)
    ld_fdr.py          — Localization Distillation on FDR (A2b only; YOLO26 has no DFL)
    pkd.py             — Pearson Correlation feature KD (A3, A4, A6)
    mgd.py             — Masked Generative Distillation (A6 backup)
    projection_mlp.py  — 1×1 conv channel projection for cross-arch feature alignment
                         (A6; removed at TRT export — training-only auxiliary)
    kd_warmup.py       — Stage-1 KD weight ramp + Stage-2 KD-off finetune scheduler
"""
