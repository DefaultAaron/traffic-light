"""Cell A3 — selection winner ← same-family M, PKD feature-level (Pearson Correlation).

Hint layers per v1.2 §5.3:
    YOLO student : neck P3/P4/P5 (stride 8/16/32)
    DEIM student : encoder memory + decoder intermediate

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A3 (P0, all families).
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
        "see docs/planning/knowledge_distillation_pipeline.md §5.1."
    )


if __name__ == "__main__":
    sys.exit(main())
