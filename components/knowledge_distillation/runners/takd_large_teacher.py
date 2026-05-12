"""TAKD with L-tier teacher: assistant-bridged distillation + ESKD checkpoint (cell A7).

Capacity-conditional cell. 4090 D capacity analysis locks the teacher tier
to L; X/XL excluded (main-bs forces headroom < 2 GB or wall-clock > 2×
scratch — violates Gate #4 cost). **Per plan §七 教师规模规则 (R1 results,
2026-05-11): YOLO26-l is DISQUALIFIED as teacher (R1 best mAP50=0.850 <
YOLO26-m 0.869); only DEIM-D-FINE-L is a candidate L-tier teacher.** A7
runs DEIM-D-FINE-S ← DEIM-D-FINE-L only.

Mandatory capacity-gap mitigations (plan §七 教师规模规则: "TAKD / ESKD /
投影 MLP 三选二"; evidence Cho 2019 / Mirzadeh 2020): two of three must be
enabled simultaneously —
TAKD intermediate (M assistant), ESKD early-stopped teacher checkpoint,
projection MLP. Orchestrators live under
`components/knowledge_distillation/schedules/` (takd_assistant.py,
eskd_loader.py).

OOM fallback: first-epoch peak > 22 GB → auto-drop bs (YOLO 8→6, DEIM 4→3);
still > 22 GB → A7 degrades to an M-teacher cell, log capacity failure.

Trigger: (a) rare-class AP misses R3 ship floor, OR (b) 4090 D capacity allows.

Spec: docs/planning/additional_components_plan.md §七 row A7 (P1, DEIM-L teacher only).
Status: scaffold — lands when A7 is scheduled (DEIM-L training complete unlocks
automatically; YOLO26-l disqualified).
"""

from __future__ import annotations

import sys

_CELL = "A7"
_SUMMARY = "TAKD via M + ESKD checkpoint, L-tier teacher only"


def main() -> int:
    raise NotImplementedError(
        f"KD cell {_CELL} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/additional_components_plan.md §七."
    )


if __name__ == "__main__":
    sys.exit(main())
