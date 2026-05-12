"""Knowledge Distillation pipeline (R2 in-round; plan §七 v2 locked 2026-05-11,
A6 path γ wording sync 2026-05-12).

Authoritative spec: ``docs/planning/additional_components_plan.md`` §七
(supersedes the pre-trim ``knowledge_distillation_pipeline.md`` per the
2026-05-10 v2.0 plan-trim consolidation).

Cells (§七 cell matrix) — file in runners/:
    A0  deim_baseline_golsd_off.py     DEIM-only baseline, GO-LSD off              (P0, DEIM)
    A1  scratch_baseline.py            Scratch baseline + wall-clock anchor        (P0, all)
    A2a yolo_logit_kd.py               YOLO26-s ← YOLO26-m, cls-logit KL only      (P0, YOLO)
    A2b deim_logit_localization_kd.py  DEIM-S ← DEIM-M, LD on FDR + cls-logit KL   (P0, DEIM)
    A3  pearson_feature_kd.py          PKD feature-level (Pearson Correlation)     (P0, all)
    A4  logit_plus_feature_kd.py       A2 + A3 combo                               (P1, gated)
    A5  progressive_multi_teacher.py   MTPD progressive 2-teacher                  (P2, gated)
    A6  cross_arch_feature_kd.py       Cross-arch DEIM-M → YOLO26-s; path γ
                                       (cls KL + FDR Integral collapse →
                                       L1+GIoU bbox KD + PKD FPN projection)       (P1, gated)
    A7  takd_large_teacher.py          DEIM-L teacher with two-of-three capacity-
                                       gap mitigations (TAKD via M, ESKD ckpt,
                                       projection MLP); YOLO26-l disqualified      (P1, gated)

Acceptance gates (§七 KD 验收门; Gates #1-#5 block KD ship-decision; #6 advisory):
    #1 total mAP non-regression — KD-cell lower-CI > A1_CI_low
       AND KD-cell lower-CI > A1_point − 0.5 pp; ship-decision forces seed5.
       A1 itself is the reference and does not pass this gate.
    #2 per-class safety AP delta ≥ −0.5 pp on classes with full_val_support ≥ 30
    #3 no new FP on R1 demo8/11/13 background frames (shares §四 manifest)
    #4 single-cell wall-clock < 2.0 × T_scratch_A1
    #5 TRT-engine eval-parity (0.01 pp) at R2 ship_precision; sidecar
       carries kd_cell_id / kd_method / kd_teacher_artifact_sha256
    #6 deploy-stability trigger — **NON-BLOCKING for KD ship-decision**.
       Post-R2-close: ship-flagged students (YOLO26-s OR DEIM-D-FINE-S)
       run demo4/10/12/15 burst-jitter + demo8/11/13 FP measurement;
       any regression vs R1 baseline registers a tuning task in
       ``docs/planning/pre_deploy_AGV_integration.md`` and writes
       ``runs/_pre_deploy_tuning_decisions.json``

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
