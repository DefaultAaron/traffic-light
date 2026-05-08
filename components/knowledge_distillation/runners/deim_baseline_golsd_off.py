"""Cell A0 — DEIM-D-FINE-S baseline, GO-LSD off, no external KD.

Role: DEIM-path net-separation baseline. Required to isolate external-KD
contribution from D-FINE's intrinsic GO-LSD self-distillation (v1.2 §5.5).

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A0 (P0, DEIM only).
Status: scaffold — lands when A0 is scheduled in R2 in-round.
"""

from __future__ import annotations

import sys

_CELL = "A0"
_SUMMARY = "DEIM-only baseline, GO-LSD off, no external KD"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/knowledge_distillation_pipeline.md §5.1."
    )


if __name__ == "__main__":
    sys.exit(main())
