"""Scratch baseline + wall-clock anchor for KD-acceptance gates (cell A1; DEIM path: GO-LSD on).

Role: Control + anchor for KD-acceptance §6 gates #1 and #4. T_scratch_A1
records its wall-clock; A2+ cells must finish under 2.0 × T_scratch_A1.
A1 itself does NOT pass §6#1 against itself — it is the reference.

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A1 (P0, all families).
Status: scaffold — lands when A1 is scheduled in R2 in-round.
"""

from __future__ import annotations

import sys

_CELL = "A1"
_SUMMARY = "scratch baseline + wall-clock anchor"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/knowledge_distillation_pipeline.md §5.1."
    )


if __name__ == "__main__":
    sys.exit(main())
