"""Progressive multi-teacher KD (MTPD): 2-teacher sequence, same-family M then cross-family M (cell A5).

Sequence (v1.2 §3.2): teacher 1 = same-family M (stable features),
teacher 2 = complementary-family M. Synchronous 4-teacher is explicitly
excluded (cost × 4 + cross-venv overhead + Cao 2023 same-class constraint).

Trigger: A4 passes all 5 §6 acceptance gates.

Spec: docs/planning/knowledge_distillation_pipeline.md §5.1 row A5 (P2, all families).
Status: scaffold — lands when A5 is scheduled (gated on A4 full pass).
"""

from __future__ import annotations

import sys

_CELL = "A5"
_SUMMARY = "MTPD progressive 2-teacher"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/knowledge_distillation_pipeline.md §5.1."
    )


if __name__ == "__main__":
    sys.exit(main())
