"""Cell A2a — YOLO26-s ← YOLO26-m, classification logit KL only.

YOLO26 has DFL removed, so localization-distillation (LD) does not apply on the
YOLO path; A2a is cls-logit KL only. DEIM students go through A2b instead.

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A2a (P0, YOLO family).
Status: scaffold — lands when A2a is scheduled in R2 in-round.
"""

from __future__ import annotations

import sys

_CELL = "A2a"
_SUMMARY = "YOLO26-s ← YOLO26-m, cls-logit KL only"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/knowledge_distillation_pipeline.md §5.1."
    )


if __name__ == "__main__":
    sys.exit(main())
