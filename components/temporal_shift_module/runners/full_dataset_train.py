"""Phase 1-B: Full train — full R2 dataset, scratch init, same epoch / aug as main detector.

Phase 1-B promotes the Phase 1-A PoC to production candidate. Full R2 dataset,
same epoch / patience / augmentation as the main detector's R2 baseline so the
"baseline + TSM" comparison is apples-to-apples.

Gate condition (§1.5 + gates/full_train_acceptance_gate.py):
    small_target_recall ≥ 0.6 (the §0.2 启动阈值 absolute floor — not a delta)
    AND end_to_end_orin_latency_ms < 26 (locked input_source / resolution /
    tracker_mode / output_mode columns matching scripts/run_demos.sh; same
    caveat columns as the R2 precision parity plan).

Trigger condition:
    Phase 1-A passed Gate-1A. NO direct entry — Phase 1-B without 1-A
    short-circuits the go/no-go gate.

Hyperparam coupling to main detector (§1.4):
    Reuses main detector's R2 epoch count, patience, augmentation YAML. The
    only deltas are: clip-collator dataloader, scratch init enforced (no
    main-detector best.pt warm-start), and the shift module patched into
    BasicBlock.forward.

Spec: docs/planning/temporal_optimization_plan.md §1.5 Phase 1-B.
Status: scaffold — lands when Phase 1-B is scheduled.
"""

from __future__ import annotations

import sys

_PHASE = "1-B"
_SUMMARY = "Full train: full R2 dataset, scratch init, main-detector hyperparams"

# Activation-gate scope (v1.4): same plan-§0.2 row-1 four-tag set as Phase 1-A
# (Phase 1-B is a generalization, not a re-scoping). Future per-phase
# narrowing replaces this constant, NOT prose docstring.
PHASE_FAILURE_MODE_SCOPE = frozenset(
    {"small_target_miss", "far_distance_miss", "occluded_miss", "motion_blur"}
)


def main() -> int:
    raise NotImplementedError(
        f"TSM phase {_PHASE} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/temporal_optimization_plan.md §1.5."
    )


if __name__ == "__main__":
    sys.exit(main())
