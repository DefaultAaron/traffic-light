"""Knowledge Distillation pipeline (R2 in-round, v1.3; v1.2 LOCKED base).

Authoritative spec: ``docs/planning/knowledge_distillation_pipeline.md`` (v1.3 —
v1.2 LOCKED matrix + acceptance gates retained verbatim; v1.3 adds runner-file
descriptive renames, §11 usage section, §12 execution checklist, and trims
non-load-bearing prose).

Cells (§5.1) — file in runners/:
    A0  deim_baseline_golsd_off.py     DEIM-only, GO-LSD off, no external KD       (P0, DEIM)
    A1  scratch_baseline.py            Scratch baseline + wall-clock anchor        (P0, all)
    A2a yolo_logit_kd.py               YOLO26-s ← YOLO26-m, cls-logit KL only      (P0, YOLO)
    A2b deim_logit_localization_kd.py  DEIM-S ← DEIM-M, LD on FDR + cls-logit KL   (P0, DEIM)
    A3  pearson_feature_kd.py          PKD feature-level (Pearson Correlation)     (P0, all)
    A4  logit_plus_feature_kd.py       A2 + A3 combo                               (P1, gated)
    A5  progressive_multi_teacher.py   MTPD progressive 2-teacher                  (P2, gated)
    A6  cross_arch_feature_kd.py       Cross-arch PKD (DEIM ↔ YOLO)                (P2, gated)
    A7  takd_large_teacher.py          L-tier teacher with two-of-three capacity-
                                       gap mitigations (TAKD via M, ESKD ckpt,
                                       projection MLP)                             (P2, gated)

Acceptance gates (§6, all must pass for KD ship-decision):
    #1 total mAP non-regression — KD-cell lower-CI > A1_CI_low
       AND KD-cell lower-CI > A1_point − 0.5 pp (95% confident the regression
       is < 0.5 pp). A1 itself is the reference and does not pass this gate.
    #2 per-class safety AP delta ≥ −0.5 pp on classes with full_val_support ≥ 30
    #3 no new FP on R1 demo8/11/13 background frames
    #4 single-cell wall-clock < 2.0 × T_scratch_A1
    #5 TRT-engine eval-parity (0.01 pp) at the R2 ship_precision

Subpackages:
    runners/    per-cell training entrypoints (A0-A7) + uniform CLI / repro /
                KD-implementation contracts (see runners/__init__.py)
    losses/     KD loss modules (cls-logit KL, LD-on-FDR, PKD, MGD, projection
                MLP, kd_weight_ramp scalar) — landed per-cell when scheduled
    schedules/  multi-phase / multi-teacher orchestrators (MTPD progressive,
                TAKD assistant, ESKD checkpoint loader, A0 GO-LSD toggle)
    gates/      §6 acceptance-gate evaluators

Deferred deliverables (NEW, not yet present):
    runs/_kd_decisions.json                   per-cell decision records
    scripts/_kd_decision_schema.json          schema (≥80% field overlap with
                                              _r2_decision_schema.json)
    scripts/_kd_decide_cell.py                decision-rule executor
"""
