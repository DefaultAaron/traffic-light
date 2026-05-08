"""Cell A4 — A2 + A3 combo (logit KD stacked on PKD feature KD).

Trigger (v1.2 §5.1): max(A2a/A2b, A3) mAP@0.5:0.95 lower-CI > A1 point estimate
AND no safety-class with full_val_support ≥ 30 has AP delta < −0.5 pp.
Tie-break (≤ 0.1 pp gap on main metric): pick cheaper wall-clock.

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A4 (P1, all families).
Status: scaffold — lands when A4 is scheduled (gated on A2/A3 outcomes).
"""

from __future__ import annotations

import sys

_CELL = "A4"
_SUMMARY = "A2 + A3 combo (logit + feature KD)"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/knowledge_distillation_pipeline.md §5.1."
    )


if __name__ == "__main__":
    sys.exit(main())
