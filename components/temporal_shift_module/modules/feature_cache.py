"""FeatureCache — per-stage previous-frame 1/8-channel container for streaming.

Spec: docs/planning/temporal_optimization_plan.md §1.5 Phase 1-C (per-camera,
per-stage cache; ROS2 node holds one instance per camera) + §1.6 risk row 2
(memory budget < 5 MB / camera; drop P5 if pressured).
Status: scaffold — lands when Phase 1-C is scheduled.

Lifecycle contract (when implementation lands):
    new camera   → cache initialized empty (returns zeros on first read)
    each frame   → write_after_forward(stage_id, slice_1_8th)
    stream gap   → reset() clears all stages (boundary becomes zero-pad)
    shutdown     → no special handling; container is GC'd with camera node
"""

from __future__ import annotations

import sys

_MODULE = "FeatureCache"


def main() -> int:
    raise NotImplementedError(
        f"TSM module {_MODULE} is a scaffold stub. "
        "Implementation lands when Phase 1-C is scheduled — "
        "see docs/planning/temporal_optimization_plan.md §1.5 + §1.6."
    )


if __name__ == "__main__":
    sys.exit(main())
