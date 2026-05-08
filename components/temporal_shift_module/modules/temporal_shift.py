"""TemporalShift module — 1/8-channel forward shift with zero-pad boundary.

Spec: docs/planning/temporal_optimization_plan.md §1.1 (mechanism) + §1.5
Phase 1-C (ONNX-friendly Slice+Concat-only requirement).
Status: scaffold — lands when Phase 1-A is scheduled.
"""

from __future__ import annotations

import sys

_MODULE = "TemporalShift"


def main() -> int:
    raise NotImplementedError(
        f"TSM module {_MODULE} is a scaffold stub. "
        "Implementation lands when Phase 1-A is scheduled — "
        "see docs/planning/temporal_optimization_plan.md §1.1 + §1.5."
    )


if __name__ == "__main__":
    sys.exit(main())
