"""Internal shared constants + helpers for ``components.hard_negative_mining``.

Single load-bearing source of truth for plan-pinned literals and the
SHA256 hex predicate used across multiple submodules. Lifted here to
remove the cross-file ClassVar drift hazard B2 review I7 / I9 / S2 /
S3 flagged on the iter-0 scaffold.

Module-private (leading underscore): NOT re-exported from the package
``__init__``. Submodules import directly via the deep path.
"""

from __future__ import annotations

from typing import Final

# Plan §4.7 + §4.1 thresholds — ALL hard-pinned. Bool-exclusion at the
# consuming dataclass __post_init__; this module just owns the values.
PLAN_LOCKED_CONFIDENCE_THRESHOLD: Final[float] = 0.25
PLAN_LOCKED_NMS_IOU_THRESHOLD: Final[float] = 0.5
# Plan §4.5 verification floor.
PLAN_LOCKED_MIN_SAMPLE_FRACTION: Final[float] = 0.10
# Defensive ceiling on the "acceptable" within-sample TP-missed rate
# (plan §4.5 doesn't cap explicitly; > 0.20 indicates a noisy mining
# pass that should be re-tuned, not waved through).
PLAN_TPM_RATE_CEILING: Final[float] = 0.20
# §3.7/§4.7 drop threshold for total mAP regression. Tolerance fields
# at every layer (YAML loader, ArmMetrics, DecisionInputs) cap at this
# value — beyond it the deploy guard tolerates regressions the drop
# trigger also rejects, making the rule incoherent.
PLAN_MAP_TOLERANCE_CEILING_PP: Final[float] = 0.5

# Plan §4.1 mining-source set. Subset of the manifest label-source
# set (which additionally includes "real_light_set" — the recall-
# denominator population that is NOT mined).
PLAN_MINING_SOURCES: Final[tuple[str, ...]] = (
    "demo8",
    "demo11",
    "demo13",
    "r2_self",
)
# Plan §4.7 manifest label-source set. Superset of mining sources by
# exactly one element ("real_light_set"), which is the recall-
# denominator population. The asymmetry is intentional and is the
# reason these two tuples are NOT the same value.
PLAN_MANIFEST_LABEL_SOURCES: Final[tuple[str, ...]] = (
    *PLAN_MINING_SOURCES,
    "real_light_set",
)


def is_hex_sha256(value: object) -> bool:
    """True iff ``value`` is a 64-char lowercase hex string.

    Matches the regex ``^[0-9a-f]{64}$`` baked into the JSON Schema.
    Lifted here so config.py / data/eval_manifest.py / gates/ablation_gate.py
    consume one implementation rather than three.
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()
