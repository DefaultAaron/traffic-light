"""Cell A2b — DEIM-D-FINE-S ← DEIM-D-FINE-M, LD on FDR + classification logit KL.

DEIM keeps DFL/FDR, so LD applies. Cls-logit KL stacked on top. A0 (GO-LSD off)
is the net-separation control for this cell — see v1.2 §5.5.

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A2b (P0, DEIM family).
Status: scaffold — lands when A2b is scheduled in R2 in-round.
"""

from __future__ import annotations

import sys

_CELL = "A2b"
_SUMMARY = "DEIM-S ← DEIM-M, LD on FDR + cls-logit KL"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/knowledge_distillation_pipeline.md §5.1."
    )


if __name__ == "__main__":
    sys.exit(main())
