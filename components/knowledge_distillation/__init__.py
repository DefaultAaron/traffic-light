"""Knowledge Distillation pipeline (R2 in-round, v1.2 LOCKED).

Authoritative spec: ``docs/planning/knowledge_distillation_pipeline.md`` (v1.2).

Cells (§5.1):
    A0  DEIM-only, GO-LSD off, no external KD          (P0, DEIM only)
    A1  Scratch baseline + wall-clock anchor           (P0, all)
    A2a YOLO26-s ← YOLO26-m, cls-logit KL only         (P0, YOLO)
    A2b DEIM-S  ← DEIM-M,    LD on FDR + cls-logit KL  (P0, DEIM)
    A3  PKD feature-level                              (P0, all)
    A4  A2 + A3 combo                                  (P1, all; gated)
    A5  MTPD progressive 2-teacher                     (P2, all; gated)
    A6  Cross-arch PKD (DEIM ↔ YOLO)                   (P2, all; gated)
    A7  TAKD via M, ESKD checkpoint, L-tier teacher    (P2, all; gated)

Acceptance gates (§6, all must pass for KD ship-decision):
    #1 total mAP non-regression — KD-cell lower-CI > A1_CI_low
       AND KD-cell lower-CI > A1_point − 0.5 pp (95% confident the regression
       is < 0.5 pp). A1 itself is the reference and does not pass this gate.
    #2 per-class safety AP delta ≥ −0.5 pp on classes with full_val_support ≥ 30
    #3 no new FP on R1 demo8/11/13 background frames
    #4 single-cell wall-clock < 2.0 × T_scratch_A1
    #5 TRT-engine eval-parity (0.01 pp) at the R2 ship_precision

Decision JSON: ``runs/_kd_decisions.json`` (schema ``scripts/_kd_decision_schema.json``).
Per-cell decision executor: ``scripts/_kd_decide_cell.py`` (NEW, deferred).
"""
