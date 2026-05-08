"""DEIM-D-FINE-S baseline with GO-LSD off and no external KD (cell A0).

Role: DEIM-path net-separation baseline. Required to isolate external-KD
contribution from D-FINE's intrinsic GO-LSD self-distillation (v1.2 §5.5).

Implementation note: GO-LSD is intrinsic to D-FINE training. Disable it via
DEIM trainer config — locate the GO-LSD loss term in `DEIM/engine/deim/`
and zero its weight or skip its construction. Toggle helper will land at
`components/knowledge_distillation/schedules/golsd_toggle.py` when A0 is
scheduled.

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
