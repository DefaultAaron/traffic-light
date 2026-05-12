"""Cross-architecture KD: DEIM-D-FINE-M → YOLO26-s (cell A6; path γ).

Path γ method stack (selected by A6 design spike on 2026-05-11; plan §七
amended 2026-05-12):

    cls-logit KL              : same form as A2a; teacher = DEIM-D-FINE-M
                                top-level pred_logits; student = YOLO26-s
                                one2many scores; cls-head shape must match
    DEIM FDR Integral collapse: teacher pred_corners (B, L, 4, reg_max+1)
                                → softmax-weighted bin sum → scalar 4-tuple
                                bbox; consumed by L1 + GIoU KD against
                                YOLO26 4 direct box channels
    PKD FPN projection conv   : 1×1 conv MLP (student stride 8/16/32 →
                                teacher channels) + Pearson correlation
                                feature loss; train-only auxiliary,
                                removed at TRT export

YOLO26 student has reg_max=1 (no native DFL head) — direct FDR↔DFL
distribution alignment is architecturally infeasible. Path γ is the
shape-feasible alternative that preserves DEIM's long-tail localization
signal via Integral collapse rather than discarding it.

Head/decoder/query alignment is OUT OF SCOPE. DETR set-prediction has no
natural mapping to YOLO grid + NMS, and the project does not promise
DETRDistill / KD-DETR cross-arch query KD; those remain DEIM↔DEIM only.

Trigger: A4 通过全部 6 gate (per plan §七 cell matrix).
PoC criterion: long-tail recall ≥ +5 pp on R2 val. Fail → demote to P2;
pseudo-label bridge (DEIM teacher pred → conf-filtered top-k → YOLO target
assignment) is the R3 fallback per `docs/kd_upgrade_recommendation.md`
adoption decision (2026-05-12), NOT in R2 scope.

Spec: docs/planning/additional_components_plan.md §七 A6 row + priority note
Design: docs/planning/kd_a6_design_spike.md (path γ rationale + α/β/δ rejection)
Spike output: runs/rehearsal_kd_A6_design_spike.json (selected_path=gamma)
Status: scaffold — lands when A4 passes all 6 gates.
"""

from __future__ import annotations

import sys

_CELL = "A6"
_SUMMARY = "cross-arch DEIM-M → YOLO26-s; cls KL + FDR Integral collapse → L1+GIoU bbox KD + PKD FPN projection (path γ)"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/additional_components_plan.md §七 + "
        "docs/planning/kd_a6_design_spike.md."
    )


if __name__ == "__main__":
    sys.exit(main())
