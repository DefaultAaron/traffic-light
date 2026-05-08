"""Clip-of-N collator — wraps native detector dataloader → (B, T, C, H, W).

Spec: docs/planning/temporal_optimization_plan.md §1.4 (clip-of-N=4) + §4.1
(R2 data SOP: 30 fps continuous, train/val split by video file).
Status: scaffold — lands when Phase 1-A is scheduled.
"""

from __future__ import annotations

import sys

_MODULE = "ClipCollator"


def main() -> int:
    raise NotImplementedError(
        f"TSM data {_MODULE} is a scaffold stub. "
        "Implementation lands when Phase 1-A is scheduled — "
        "see docs/planning/temporal_optimization_plan.md §1.4 + §4.1."
    )


if __name__ == "__main__":
    sys.exit(main())
