"""Pearson Correlation feature KD (PKD): selection winner ← same-family M (cell A3).

Hint layers per v1.2 §5.3:
    YOLO student : neck P3/P4/P5 (stride 8/16/32) + head pre-logit
    DEIM student : encoder memory + decoder intermediate

Spec: docs/planning/additional_components_plan.md §七 row A3 (P0, all families).
Status: scaffold — lands when A3 is scheduled in R2 in-round.
"""

from __future__ import annotations

import sys

_CELL = "A3"
_SUMMARY = "PKD feature-level, same-family M teacher"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/additional_components_plan.md §七."
    )


if __name__ == "__main__":
    sys.exit(main())
