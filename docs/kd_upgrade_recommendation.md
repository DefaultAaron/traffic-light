# KD Upgrade Recommendation for YOLO26 and DEIM-D-FINE

## Review Summary

The current KD section is already directionally strong. It includes:

- S-only deployment students: `YOLO26-s` and `DEIM-D-FINE-S`.
- Same-family teacher paths: `YOLO26-m -> YOLO26-s` and `DEIM-D-FINE-M -> DEIM-D-FINE-S`.
- DEIM-side localization distillation using FDR / distribution-style supervision.
- Feature-level KD through PKD.
- Combined logit + feature KD.
- Cross-architecture DEIM-M -> YOLO26-s as a high-value path.
- Seed / CI-based ship gates, safety-class AP gates, hard-negative FP gates, cost gates, and TRT sidecar requirements.

However, the plan does not yet fully reflect the recommended KD implementation details for YOLO, DEIM, and cross-family distillation. The upgrade below keeps the current A0-A7 structure but makes the KD losses and cross-distillation paths safer and more actionable.

---

## Main Gaps

### Gap 1: YOLO same-family KD is too weak if it only uses class-logit KL

Current A2a:

```text
YOLO26-s <- YOLO26-m
method: cls-logit KL
```

This should be upgraded. For dense one-stage detectors, class-logit KD alone usually underuses the teacher. The YOLO KD path should include:

```text
L = L_yolo_gt
  + λ_cls * L_clsKD
  + λ_box * L_boxKD
  + λ_feat * L_featureKD
```

Recommended components:

| Component | Recommendation |
|---|---|
| class KD | confidence-filtered KL on foreground / high-quality teacher candidates |
| box KD | DFL / IoU-weighted localization KD |
| feature KD | FPN/PAN multi-scale feature KD using MGD / FGD / CWD-style masking, foreground weighting, or channel-wise normalization |
| filtering | do not distill all background cells equally; use teacher confidence, GT foreground mask, or top-k per image |

Suggested default weights:

```text
λ_cls  = 0.5
λ_box  = 2.0
λ_feat = 1.0
temperature = 2-4
teacher_conf_min = 0.3-0.5
```

---

### Gap 2: DEIM-D-FINE KD should explicitly separate query KD, FDR/box distribution KD, and encoder feature KD

Current A2b:

```text
DEIM-D-FINE-S <- DEIM-D-FINE-M
method: LD on FDR + cls-logit KL
```

This is good, but should be made more explicit:

```text
L = L_deim_gt
  + λ_query * L_queryKD
  + λ_fdr   * L_FDR_distributionKD
  + λ_cls   * L_clsKD
  + λ_enc   * L_encoderFeatureKD
```

Recommended components:

| Component | Recommendation |
|---|---|
| FDR KD | primary signal; distill refined localization distributions, not only decoded boxes |
| query KD | match teacher and student queries using Hungarian matching or teacher-GT aligned assignment |
| class KD | apply after query matching; avoid unmatched/no-object domination |
| encoder feature KD | optional; use projected feature KD only if feature resolutions/channels differ |
| GO-LSD | keep A0 as GO-LSD-off baseline, but main DEIM training should preserve the native D-FINE self-distillation path unless A0 is intentionally measuring its contribution |

Suggested default weights:

```text
λ_fdr   = 2.0
λ_query = 1.0
λ_cls   = 0.5
λ_enc   = 0.5
```

---

### Gap 3: Cross-architecture KD should not begin with direct feature/logit alignment only

Current A6:

```text
YOLO26-s <- DEIM-D-FINE-M
method: cross-arch logit + FDR↔DFL distribution alignment + projection MLP
```

This is valuable, but risky as the first cross-architecture implementation. YOLO and DEIM have different prediction structures:

```text
YOLO: dense grid / point / anchor-like predictions
DEIM-D-FINE: set prediction / decoder queries
```

Directly aligning YOLO head tensors to DEIM query tensors can be unstable. Add a pseudo-label bridge before feature/query alignment.

Recommended A6 upgrade:

```text
A6a: DEIM-D-FINE-M -> YOLO26-s pseudo-label KD
A6b: DEIM-D-FINE-M -> YOLO26-s pseudo-label KD + FDR-to-DFL box distribution KD
A6c: DEIM-D-FINE-M -> YOLO26-s pseudo-label KD + projected feature KD
```

Pseudo-label bridge:

```text
DEIM teacher predictions
-> confidence filtering
-> class-aware NMS or WBF
-> top-k boxes per image
-> YOLO target assignment
-> train YOLO with GT + pseudo targets
```

Recommended pseudo-label rules:

```text
teacher_conf_min = 0.4 initially
top_k = 100-300 boxes/image
NMS IoU = 0.5-0.7
pseudo weight = teacher_conf * localization_quality
ignore ambiguous pseudo boxes overlapping GT with wrong class
```

This makes cross-KD much easier to debug and safer for deployment.

---

### Gap 4: Reverse cross-distillation is missing

The current matrix has DEIM-M -> YOLO26-s, but does not include the reverse direction:

```text
YOLO26-m -> DEIM-D-FINE-S
```

This should be added as an optional P2 cell, not a P0/P1 blocker.

Recommended new cell:

```text
A8: DEIM-D-FINE-S <- YOLO26-m
method: YOLO pseudo-label bridge + query assignment + optional projected feature KD
priority: P2
trigger: only if YOLO-M has better precision/stability on hard-negative demos or DEIM-S has persistent false positives
```

Why P2:

- It may help DEIM-S learn YOLO's conservative false-positive behavior.
- It is less likely to improve localization than DEIM -> YOLO.
- It should not block the main deployment path.

---

## Recommended Updated Cell Matrix

| Cell | Student | Teacher | Method Stack | Priority | Change |
|---|---|---|---|---|---|
| A0 | DEIM-D-FINE-S | — | GO-LSD off; no external KD | P0 | keep |
| A1 | YOLO26-s + DEIM-D-FINE-S | — | scratch baseline | P0 | keep |
| A2a | YOLO26-s | YOLO26-m | cls KL + DFL/box KD + foreground-filtered candidate KD | P0 | upgrade |
| A2b | DEIM-D-FINE-S | DEIM-D-FINE-M | FDR distribution KD + query KD + cls KL | P0 | upgrade |
| A3 | S dual paths | same-family M | MGD/FGD/CWD-style feature KD, not only PKD | P0 | upgrade |
| A4 | S dual paths | same-family M | A2 + A3 | P1 | keep |
| A5 | S dual paths | same-family M -> complementary-family M | progressive 2-teacher | P2 | keep but drawdown first |
| A6a | YOLO26-s | DEIM-D-FINE-M | pseudo-label KD bridge | P1 | add before current A6 |
| A6b | YOLO26-s | DEIM-D-FINE-M | pseudo-label KD + FDR-to-DFL localization KD | P1 | add |
| A6c | YOLO26-s | DEIM-D-FINE-M | pseudo-label KD + projected feature KD | P1/P2 | replace risky direct cross-feature-first path |
| A7 | DEIM-D-FINE-S | DEIM-D-FINE-L | TAKD / ESKD / projection MLP | P1 | keep |
| A8 | DEIM-D-FINE-S | YOLO26-m | YOLO pseudo-label KD + query assignment | P2 | new reverse cross-KD |

---

## Runner Mapping Upgrade

Replace or extend the current runner names as follows:

```text
A2a -> components.knowledge_distillation.runners.yolo_localization_logit_kd
A2b -> components.knowledge_distillation.runners.deim_query_fdr_kd
A3  -> components.knowledge_distillation.runners.detector_feature_kd_mgd_fgd
A6a -> components.knowledge_distillation.runners.deim_to_yolo_pseudo_kd
A6b -> components.knowledge_distillation.runners.deim_to_yolo_pseudo_fdr_dfl_kd
A6c -> components.knowledge_distillation.runners.deim_to_yolo_projected_feature_kd
A8  -> components.knowledge_distillation.runners.yolo_to_deim_pseudo_query_kd
```

Keep the existing unified CLI, but add:

```text
--teacher-conf-min
--teacher-topk
--pseudo-nms-iou
--pseudo-weight-mode {constant,conf,conf_iou}
--kd-loss-weights
--feature-kd-method {pkd,mgd,fgd,cwd}
--cross-kd-stage {pseudo,distribution,feature}
```

---

## Decision Gate Upgrade

The existing gates are good. Add three KD-specific diagnostic gates:

| Gate | Requirement |
|---|---|
| #7 Rare-class recall bucket | KD must not reduce rare safety-class recall by > 0.5 pp; if rare-class AP improves but recall drops, mark as `defer` |
| #8 Small-object bucket | For traffic-light distant/small-object bucket, KD must improve recall or keep delta ≥ -0.5 pp |
| #9 Pseudo-label noise audit | For cross-KD cells, manually inspect or auto-audit a fixed sample of pseudo labels; if obvious wrong pseudo-label rate > 5-10%, do not ship |

Suggested pseudo-label audit file:

```text
runs/_kd_pseudo_label_audit.json
```

Required fields:

```json
{
  "cell_id": "A6a",
  "teacher": "deim_dfine_m",
  "student": "yolo26_s",
  "sample_size": 200,
  "wrong_label_rate": 0.0,
  "bad_box_rate": 0.0,
  "missed_gt_conflict_rate": 0.0,
  "decision": "pass|defer|drop"
}
```

---

## Recommended Execution Order

Use this order to reduce risk:

```text
1. A1 scratch baselines
2. A2a upgraded YOLO localization/logit KD
3. A2b upgraded DEIM query/FDR KD
4. A3 feature KD with MGD/FGD/CWD option
5. A4 same-family combined KD
6. A6a DEIM -> YOLO pseudo-label KD
7. A6b DEIM -> YOLO pseudo + localization distribution KD
8. A7 DEIM-L -> DEIM-S, if DEIM-L is ready
9. A6c projected cross-feature KD only after A6a/A6b pass
10. A8 YOLO -> DEIM reverse KD only if DEIM-S needs YOLO-style FP suppression
```

---

## Minimal Patch to Current Plan

If time is limited, make only these edits:

1. Upgrade A2a from `cls-logit KL` to:

```text
cls-logit KL + DFL/box KD + foreground-filtered candidate KD
```

2. Upgrade A2b from `LD on FDR + cls-logit KL` to:

```text
FDR distribution KD + query KD + cls-logit KL
```

3. Replace A3 `PKD feature-level` with:

```text
feature KD selectable from PKD / MGD / FGD / CWD
```

4. Split A6 into:

```text
A6a: DEIM -> YOLO pseudo-label KD
A6b: DEIM -> YOLO pseudo-label KD + FDR-to-DFL KD
A6c: DEIM -> YOLO projected feature KD
```

5. Add optional P2 reverse cell:

```text
A8: YOLO -> DEIM pseudo-label/query KD
```

6. Add pseudo-label audit gate for all cross-KD cells.

---

## Final Recommendation

The current plan includes the core recommendation direction, especially same-family KD and the high-value DEIM -> YOLO cross-distillation path. The main upgrade is to make KD less abstract and more detector-specific:

- YOLO KD should emphasize localization and multi-scale feature transfer, not only class KL.
- DEIM-D-FINE KD should explicitly distill FDR distributions and query-level assignments.
- Cross-KD should start from pseudo-label bridging before direct feature/query alignment.
- Reverse YOLO -> DEIM KD should be added as optional P2 for false-positive suppression.
- Cross-KD must include pseudo-label noise auditing before ship decisions.

---

## Adoption Decision (2026-05-12, post-review)

This proposal was cross-checked by two independent reviewers run in parallel:
- `paper-researcher` for academic-evidence verification (citation-grounded; refuses to invent results)
- `codex-plan-conflictor` for plan-shape adversarial review against the locked §七 matrix

Full audit trail: `research/surveys/kd_upgrade_verification.md` (verdict table per question, BibTeX, decision-rule loophole analysis).

### Cross-checked verdicts

| GPT recommendation | Paper evidence | Conflictor verdict | **Adoption** |
|---|---|---|---|
| A2a + DFL/box KD | CONTRADICTED (YOLO26 `reg_max=1`, no DFL head) | REJECT | **REJECT** (architecturally infeasible) |
| A2a foreground/conf filtering | STRONGLY-SUPPORTED (FGD, LD, GID) | folded into A2a REJECT | **DEFER → R3 additive ablation** |
| A2a / A3 feature KD method swap (MGD/FGD/CWD) | WEAKLY-SUPPORTED | REJECT (no local evidence PKD insufficient) | **DEFER → R3 challenger** (only if PKD fails predeclared diagnostic) |
| A2b + query KD + encoder feat | WEAKLY-SUPPORTED (DETRDistill, KD-DETR) | DEFER-TO-R3 (LANDED, no local evidence weak) | **DEFER → R3** |
| A6 split A6a/A6b/A6c | A6b CONTRADICTED, A6a NO-EVIDENCE | ADOPT-WITH-AMENDMENT | **AMEND A6 wording only**; pseudo-label → R3 fallback |
| A8 reverse YOLO-M → DEIM-S | NO-EVIDENCE | REJECT (R1 demo supports opposite direction) | **REJECT** |
| New Gates #7 / #8 / #9 | n/a | REJECT (gate-creep; overlaps Gates #2 / #3 / #6; gameable) | **REJECT** (6-gate contract preserved) |
| Suggested execution order | n/a | REJECT | **REJECT** |
| "Minimal Patch" subsection | n/a | REJECT (not minimal) | **REJECT** |

### Applied amendment (single change to locked plan)

`docs/planning/additional_components_plan.md` §七:
1. Cell matrix A6 row: method-stack wording synced to A6 spike's path γ (FDR Integral collapse → scalar bbox L1+GIoU + cls-logit KL + PKD projection conv)
2. A6 priority-elevation note: spike-direction wording updated; pseudo-label bridge noted as R3 fallback (not a new cell, not in R2 scope)

No new cells, no new gates, no runner renames, no drawdown changes.

### R3 carry-forward (parking lot — not R2 work)

Four items from this proposal have merit but are out-of-scope for R2. Carry forward to a dedicated R3 KD-ablation plan (or `docs/planning/R3_precision_reproducibility.md` companion):
1. A2a foreground-filtered / teacher-conf-min class KD (STRONGLY-SUPPORTED in literature)
2. A3 challenger ablation: MGD vs FGD vs CWD vs PKD (only if PKD R2 result fails a predeclared diagnostic)
3. A2b query-level + encoder feature KD (only if A2b R2 result lower-CI does not beat A1 by ≥ 0.5 pp)
4. A6 pseudo-label bridge fallback (only if A6 path γ PoC fails the +5 pp long-tail recall threshold)

### Rationale for narrow adoption

- **WHAT-not-WHY plan policy**: GPT proposal is rationale-heavy; importing it into a locked plan would pollute it. Citations and verdicts live here + in `research/surveys/kd_upgrade_verification.md`, not in §七.
- **LANDED runner integrity**: A2a / A2b runners completed multi-iter B2 + C3 cycles; reopening them mid-round creates review debt and reproducibility churn without local evidence of weakness.
- **Locked 6-gate contract**: Gates #7 / #8 / #9 partially overlap Gates #2 / #3 / #6 and introduce gameable thresholds (bucket definitions, audit-sample bias, NMS-dependent wrong-label rate).
- **Architectural constraint**: YOLO26 `reg_max=1` is an empirically confirmed property (A6 spike output); DFL-dependent recommendations cannot land without a separate YOLO head-scaffolding spike, which is R3+ scope.
- **R1 demo evidence**: DEIM-S/M frame-to-frame stability < YOLO; this supports A6's DEIM→YOLO direction, not A8's reverse. Deploy-side FP suppression for DEIM-S is already covered by Gate #6 deploy-tuning trigger.
