"""Phase 1-A: PoC — small-data train (10% data, 20 epochs) on the selected detector.

Phase 1-A is the **go/no-go gate** before full retrain. Per §1.5: 10% R2 data,
20 epochs, scratch init, clip-size 4. Compare TSM-on vs TSM-off on a small val
subset bucketed by bbox height (small / medium / large).

Gate condition (§1.5 + gates/concept_validation_gate.py):
    small_target_recall_delta ≥ 2 pp  OR  mAP_delta ≥ 1 pp
    Either condition is sufficient to advance to Phase 1-B. Failing both → loop
    back to plan §0.2 problem-attribution re-evaluation (DO NOT iterate
    hyperparams; per §1.5 the failure means TSM is not the right tool here).

Trigger condition:
    R2 main detector selection final + on-vehicle replay surfaced small-target
    or occluded miss failure modes (§0.2 row 1 or row 4). Phase 1-A NEVER
    starts before the main track is done.

Spec: docs/planning/temporal_optimization_plan.md §1.5 Phase 1-A.
Status: scaffold — lands when Phase 1-A is scheduled.
"""

from __future__ import annotations

import sys

_PHASE = "1-A"
_SUMMARY = "PoC: 10% data, 20 epochs, scratch init, clip-size 4"


def main() -> int:
    raise NotImplementedError(
        f"TSM phase {_PHASE} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/temporal_optimization_plan.md §1.5."
    )


if __name__ == "__main__":
    sys.exit(main())
