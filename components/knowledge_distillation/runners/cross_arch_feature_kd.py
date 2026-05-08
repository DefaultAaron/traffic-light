"""Cross-architecture feature KD (DEIM ↔ YOLO): arch-agnostic feature-level only via PKD (cell A6).

Spec (v1.2 §4.3):
    direction       teacher = complementary-family M; student = same-family S
    feature levels  three-layer FPN-style pyramid; stride 8/16/32 reshape on DEIM
                    side; spatial bilinear resample on ±1 stride mismatch
    projection      1×1 conv MLP (student channels → teacher channels), removed
                    at TRT export — training-only auxiliary
    losses          PKD primary; MGD as backup; FGD foreground-mask NOT used
                    (cross-arch fg/bg semantics insufficiently validated)

Head/decoder/query alignment is OUT OF SCOPE — DETR set-prediction has no natural
mapping to YOLO grid + NMS (v1.2 §4.2). DETRDistill / KD-DETR remain DEIM↔DEIM only.

Trigger: A4 passes §6 + team capacity.
Legal gate (§8): A6 student-weight publish-gate DEFERRED to commercial-deploy
stage. Field-test stage is non-blocking, advisory-only.

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A6 (P2, all families).
Status: scaffold — lands when A6 is scheduled (gated on A4 full pass).
"""

from __future__ import annotations

import sys

_CELL = "A6"
_SUMMARY = "cross-arch PKD (DEIM ↔ YOLO), feature-level only"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/knowledge_distillation_pipeline.md §5.1."
    )


if __name__ == "__main__":
    sys.exit(main())
