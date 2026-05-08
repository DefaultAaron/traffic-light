"""Gate-1A: small_target_recall_delta ≥ 2 pp OR mAP_delta ≥ 1 pp.

OR condition — either suffices to advance to Phase 1-B. Failing both is the
explicit §1.5 "回到 §0.2 重新评估问题归因" trigger; do NOT auto-iterate
hyperparams from this gate's failure path.

Spec: docs/planning/temporal_optimization_plan.md §1.5 Phase 1-A 通过判定.
Status: scaffold — lands when Phase 1-A is scheduled.
"""

from __future__ import annotations

import sys

_GATE = "1-A"
_RULE = "small_target_recall_delta ≥ 2 pp OR mAP_delta ≥ 1 pp"


def main() -> int:
    raise NotImplementedError(
        f"TSM gate {_GATE} ({_RULE}) is a scaffold stub. "
        "Implementation lands when Phase 1-A is scheduled — "
        "see docs/planning/temporal_optimization_plan.md §1.5."
    )


if __name__ == "__main__":
    sys.exit(main())
