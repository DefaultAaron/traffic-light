"""Gate-1B: small_target_recall ≥ 0.6 AND end_to_end_orin_latency_ms < 26.

AND condition — both required. Recall threshold is an absolute floor (the
§0.2 启动阈值), not a delta vs Phase 1-A. Latency MUST be measured with the
locked caveat columns input_source / resolution / tracker_mode / output_mode.

Spec: docs/planning/temporal_optimization_plan.md §1.5 Phase 1-B 通过判定 +
§0.2 启动阈值.
Status: scaffold — lands when Phase 1-B is scheduled.
"""

from __future__ import annotations

import sys

_GATE = "1-B"
_RULE = "small_target_recall ≥ 0.6 AND end_to_end_orin_latency_ms < 26"


def main() -> int:
    raise NotImplementedError(
        f"TSM gate {_GATE} ({_RULE}) is a scaffold stub. "
        "Implementation lands when Phase 1-B is scheduled — "
        "see docs/planning/temporal_optimization_plan.md §1.5 + §0.2."
    )


if __name__ == "__main__":
    sys.exit(main())
