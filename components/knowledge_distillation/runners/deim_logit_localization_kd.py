"""DEIM logit + localization KD: DEIM-S ← DEIM-M, LD on FDR + cls-logit KL (cell A2b).

DEIM keeps DFL/FDR, so external LD on FDR is structurally applicable; cls-logit KL
is stacked on top. Per v1.3 §5.5, A2b is the **GO-LSD-vs-external-LD overlap
ablation**, not a primary KD recommendation — D-FINE's intrinsic GO-LSD already
covers the localization channel, so external LD's marginal benefit is expected
to be small. The net-separation arithmetic (DEIM path):
    A1 − A0  = GO-LSD intrinsic self-distillation contribution
    A2b − A1 = external (LD on FDR + cls-logit KL) JOINT marginal contribution

If A2b − A1 ≤ §6#1 noise threshold, the §5.5 overlap hypothesis is empirically
confirmed; subsequent DEIM P1+ cells (A3 / A4) should prioritize classification
and encoder-feature channels to avoid the localization overlap.

NOTE: A2b − A1 isolates LD AND cls-logit KL together; isolating LD alone would
require a fourth ablation cell (out of scope at v1.3 — accept joint signal).

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A2b + §5.5 (P0, DEIM family).
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
