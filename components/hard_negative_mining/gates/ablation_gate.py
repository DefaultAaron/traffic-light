"""c-stage acceptance metrics for the §四 hard-negative-mining ablation.

Computes the three pre-committed metrics from
``docs/planning/additional_components_plan.md`` §4.7 on a single
already-trained arm's eval output:

  1. ``fp_drop_frac``                — relative FP count reduction on
                                        the §4.7 frozen manifest. Signed
                                        fraction; +0.5 = 50% reduction.
                                        Computed as
                                        ``(base_fp - cand_fp) / max(base_fp, 1)``
                                        — the ``max(., 1)`` floor avoids
                                        zero-divide; b-stage logs a
                                        WARNING when ``base_fp == 0``
                                        (the FP rate is already at floor
                                        and the metric is undefined; the
                                        decision rule defaults to drop
                                        in that pathological case via
                                        the catch-all).
  2. ``real_light_recall_delta_pp``  — change in real-light recall on
                                        the §4.7 frozen manifest's
                                        ``has_real_light=True`` subset,
                                        in percentage points. Computed
                                        as ``(candidate - baseline) * 100``;
                                        signed (negative = regression).
  3. ``total_map_delta_pp``          — total mAP@0.5 delta on the frozen
                                        R2 val set (NOT the §4.7 manifest)
                                        in percentage points. Computed
                                        as ``(candidate - baseline) * 100``;
                                        signed (negative = regression).

All deltas are ``arm - baseline`` where the baseline is the no_hn
arm. The baseline arm itself carries all-zero deltas +
``is_baseline_reference=True`` (self-comparison).

Frozen manifest contract (plan §4.7 verbatim — anti-gaming):
  * The §四 frozen manifest at
    ``runs/_hard_negative_eval_manifest.json`` is shared with §3.7;
    ``fp_manifest_sha256`` MUST match across every per-arm eval JSON
    in a single ablation run.
  * Confidence threshold = 0.25; NMS IoU = 0.5; both pinned at the
    manifest layer (see ``data/eval_manifest.FrozenEvalManifest``).
  * The denominator (recall denominator AND FP denominator) is FROZEN
    at fit time. The runner refuses to consume per-arm eval JSONs
    whose manifest hash differs from the loaded manifest (or differs
    between baseline and candidate).

Scaffold (a-stage): API signatures only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from components.hard_negative_mining._internals import (
    PLAN_MAP_TOLERANCE_CEILING_PP,
    is_hex_sha256,
)


@dataclass(frozen=True)
class ArmMetrics:
    """Aggregated c-stage metrics for ONE arm minus the baseline arm.

    Mirrors ``components/copy_paste_balance/gates/ablation_gate.ArmMetrics``
    structurally but with §4.7 metric set:
      * ``fp_drop_frac`` (signed; +0.5 = 50% drop)
      * ``real_light_recall_delta_pp``
      * ``total_map_delta_pp``
      * ``map_no_regression`` (upstream verdict for THIS arm vs baseline)
      * ``map_regression_tolerance_pp``
      * ``data_yaml_sha256``
      * ``eval_manifest_sha256`` (R2 frozen val manifest)
      * ``fp_manifest_sha256`` (§4.7 + §3.7 shared frozen FP manifest)

    Cross-arm invariants (the runner enforces — gate trusts):
      * Eval manifest hash identical across arms (R2 val set).
      * FP manifest hash identical across arms (§4.7 frozen manifest).
      * data.yaml hash identical across arms (class-label provenance).
      * map_regression_tolerance_pp identical (runner-side knob).
      * arm_id == "no_hn" iff is_baseline_reference == True.
    """

    arm_id: str                                 # "no_hn" | "with_hn"
    is_baseline_reference: bool
    fp_drop_frac: float
    real_light_recall_delta_pp: float
    total_map_delta_pp: float
    map_no_regression: bool
    map_regression_tolerance_pp: float
    eval_manifest_sha256: str
    fp_manifest_sha256: str
    data_yaml_sha256: str

    _ALLOWED_ARM_IDS: ClassVar[tuple[str, ...]] = ("no_hn", "with_hn")
    # Schema MetricsBlock ranges (mirror _hard_negative_decision_schema.json):
    _DELTA_PP_MIN: ClassVar[float] = -100.0
    _DELTA_PP_MAX: ClassVar[float] = 100.0
    # B2 review I2 2026-05-10: ``fp_drop_frac`` lower bound ``-10.0`` is
    # a TRIPWIRE, not a constraint of the math. The math admits any
    # value in (-∞, 1]; a candidate that increased FP by > 10x relative
    # to baseline is so catastrophic that the run should hard-fail at
    # the dataclass boundary rather than serialize a deploy/defer/drop
    # decision. The schema's matching range field carries the same
    # rationale comment so the two layers stay coherent.
    _FP_DROP_FRAC_MIN: ClassVar[float] = -10.0
    _FP_DROP_FRAC_MAX: ClassVar[float] = 1.0
    _MAP_TOLERANCE_CEILING_PP: ClassVar[float] = PLAN_MAP_TOLERANCE_CEILING_PP

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str):
            raise ValueError(
                f"arm_id must be str; got {type(self.arm_id).__name__}="
                f"{self.arm_id!r}"
            )
        if self.arm_id not in self._ALLOWED_ARM_IDS:
            raise ValueError(
                f"arm_id must be one of {self._ALLOWED_ARM_IDS}; got "
                f"{self.arm_id!r}"
            )
        if not isinstance(self.is_baseline_reference, bool):
            raise ValueError(
                f"is_baseline_reference must be bool; got "
                f"{type(self.is_baseline_reference).__name__}"
            )
        # Cross-knob: arm_id == "no_hn" iff is_baseline_reference. The runner
        # constructs the baseline ArmMetrics from a self-comparison; the
        # invariant must hold both ways.
        if (self.arm_id == "no_hn") != self.is_baseline_reference:
            raise ValueError(
                f"is_baseline_reference must be True iff arm_id == 'no_hn'; "
                f"got arm_id={self.arm_id!r}, "
                f"is_baseline_reference={self.is_baseline_reference}"
            )
        # Numeric delta fields: bool-exclusion + finite + range.
        for field_name, field_value, lo, hi in (
            ("real_light_recall_delta_pp", self.real_light_recall_delta_pp,
             self._DELTA_PP_MIN, self._DELTA_PP_MAX),
            ("total_map_delta_pp", self.total_map_delta_pp,
             self._DELTA_PP_MIN, self._DELTA_PP_MAX),
            ("fp_drop_frac", self.fp_drop_frac,
             self._FP_DROP_FRAC_MIN, self._FP_DROP_FRAC_MAX),
        ):
            if not isinstance(field_value, float) or isinstance(field_value, bool):
                raise ValueError(
                    f"{field_name} must be float; got "
                    f"{type(field_value).__name__}={field_value!r}"
                )
            if not math.isfinite(field_value):
                raise ValueError(
                    f"{field_name} must be finite; got {field_value!r}"
                )
            if not (lo <= field_value <= hi):
                raise ValueError(
                    f"{field_name} must be in [{lo}, {hi}]; got {field_value}"
                )
        # Baseline-reference invariant: all delta fields exactly 0.
        # B2 final-review S2 2026-05-10: this block mirrors the schema's
        # BaselineMetricsBlock allOf overlay (const-zero deltas + const-true
        # map_no_regression) at _hard_negative_decision_schema.json. Both
        # layers MUST move in lock-step if §4.7 is ever amended to allow
        # non-zero baseline deltas.
        if self.is_baseline_reference:
            for field_name, field_value in (
                ("fp_drop_frac", self.fp_drop_frac),
                ("real_light_recall_delta_pp", self.real_light_recall_delta_pp),
                ("total_map_delta_pp", self.total_map_delta_pp),
            ):
                if field_value != 0.0:
                    raise ValueError(
                        f"is_baseline_reference=True requires {field_name} == 0; "
                        f"got {field_value} (the no_hn arm is constructed by "
                        f"self-comparison so every delta must be exactly 0)"
                    )
        # Manifest hashes: 64-char lowercase hex.
        if not is_hex_sha256(self.eval_manifest_sha256):
            raise ValueError(
                f"eval_manifest_sha256 must be 64-char lowercase hex; got "
                f"{self.eval_manifest_sha256!r}"
            )
        if not is_hex_sha256(self.fp_manifest_sha256):
            raise ValueError(
                f"fp_manifest_sha256 must be 64-char lowercase hex; got "
                f"{self.fp_manifest_sha256!r}"
            )
        if not is_hex_sha256(self.data_yaml_sha256):
            raise ValueError(
                f"data_yaml_sha256 must be 64-char lowercase hex; got "
                f"{self.data_yaml_sha256!r}"
            )
        # map_no_regression: must be plain bool. For the baseline arm,
        # trivially True (self-comparison; mAP delta is 0).
        if not isinstance(self.map_no_regression, bool):
            raise ValueError(
                f"map_no_regression must be bool; got "
                f"{type(self.map_no_regression).__name__}="
                f"{self.map_no_regression!r}"
            )
        if self.is_baseline_reference and not self.map_no_regression:
            raise ValueError(
                f"is_baseline_reference=True requires map_no_regression=True "
                f"(baseline is constructed by self-comparison; mAP delta is "
                f"exactly 0, no regression possible); got "
                f"map_no_regression={self.map_no_regression}"
            )
        # map_regression_tolerance_pp: float-with-bool-exclusion + finite +
        # range [0, PLAN_MAP_TOLERANCE_CEILING_PP] (mirrors DecisionInputs
        # ceiling at the §3.7/§4.7 drop threshold).
        if not isinstance(self.map_regression_tolerance_pp, float) or isinstance(
            self.map_regression_tolerance_pp, bool
        ):
            raise ValueError(
                f"map_regression_tolerance_pp must be float; got "
                f"{type(self.map_regression_tolerance_pp).__name__}="
                f"{self.map_regression_tolerance_pp!r}"
            )
        if not math.isfinite(self.map_regression_tolerance_pp):
            raise ValueError(
                f"map_regression_tolerance_pp must be finite; got "
                f"{self.map_regression_tolerance_pp!r}"
            )
        if not (0.0 <= self.map_regression_tolerance_pp <= self._MAP_TOLERANCE_CEILING_PP):
            raise ValueError(
                f"map_regression_tolerance_pp must be in "
                f"[0, {self._MAP_TOLERANCE_CEILING_PP}]; got "
                f"{self.map_regression_tolerance_pp} (cap mirrors the "
                f"§3.7 / §4.7 drop threshold for total mAP regression — "
                f"a tolerance > {self._MAP_TOLERANCE_CEILING_PP} makes "
                f"the deploy guard broader than the drop trigger, which "
                f"is incoherent)"
            )


def compute_arm_metrics(
    *,
    arm_id: str,
    is_baseline_reference: bool,
    baseline_fp_count: int,
    candidate_fp_count: int,
    baseline_real_light_recall: float,
    candidate_real_light_recall: float,
    baseline_total_map: float,
    candidate_total_map: float,
    eval_manifest_sha256: str,
    fp_manifest_sha256: str,
    map_no_regression: bool,
    map_regression_tolerance_pp: float,
    data_yaml_sha256: str,
) -> ArmMetrics:
    """Compute the §4.7 metric block on a single arm vs the baseline.

    Calling convention (b-stage spells out — order matters for the
    signed-delta invariant):

      1. ``fp_drop_frac = (baseline_fp_count - candidate_fp_count)
                           / max(baseline_fp_count, 1)``
         — positive = candidate has fewer FPs (the desired direction).
         The ``max(., 1)`` floor avoids zero-divide; b-stage logs a
         WARNING when ``baseline_fp_count == 0`` (rare-FP rate already
         at floor; the metric is undefined and the rule defers to the
         drop catch-all).
      2. ``real_light_recall_delta_pp = (candidate - baseline) * 100.0``
         — positive = candidate has higher recall (the desired direction).
      3. ``total_map_delta_pp = (candidate - baseline) * 100.0``
         — positive = candidate has higher mAP.

    Baseline-arm contract (B2 review C4 2026-05-10 — explicit kwarg
    instead of the iter-0 implicit-from-arm_id rule):
      * The caller MUST pass ``is_baseline_reference=True`` AND
        ``arm_id == "no_hn"`` together (cross-checked at the body's
        guard before ``ArmMetrics`` is constructed). For a baseline
        arm, the caller MUST also pass identical baseline & candidate
        counts/recalls/maps so all three deltas are exactly 0.
      * ``arm_id == "with_hn"`` MUST be paired with
        ``is_baseline_reference=False``; mismatch raises ``ValueError``
        with a single, self-explaining message rather than the
        confusing ``ArmMetrics.__post_init__`` "fp_drop_frac == 0"
        cascade B2 flagged on iter-0.

    The function returns an ``ArmMetrics`` ready to feed into the
    d-stage decision gate. Output validation is delegated to
    ``ArmMetrics.__post_init__`` — the body of this function performs
    the cross-knob arm/is_baseline_reference check then constructs
    the dataclass.

    Args:
        arm_id: ``"no_hn"`` (baseline) or ``"with_hn"`` (candidate).
        is_baseline_reference: explicit baseline flag; MUST equal
            ``arm_id == "no_hn"``.
        baseline_fp_count: FP count from the no_hn arm on the §4.7
            frozen manifest.
        candidate_fp_count: same, this arm.
        baseline_real_light_recall: real-light recall from the no_hn arm
            on the §4.7 manifest's has_real_light=True subset (absolute,
            range [0, 1]).
        candidate_real_light_recall: same, this arm.
        baseline_total_map: total mAP@0.5 from the no_hn arm on the
            frozen R2 val set (absolute, range [0, 1]).
        candidate_total_map: same, this arm.
        eval_manifest_sha256: hash of the frozen R2 val manifest.
        fp_manifest_sha256: hash of the §4.7 + §3.7 shared frozen FP
            manifest.
        map_no_regression: upstream eval verdict for THIS arm vs baseline.
            For arm_id == "no_hn" the caller MUST pass True
            (self-comparison).
        map_regression_tolerance_pp: tolerance applied; runner-side knob.
            Same value MUST be passed for every arm in a single run.
        data_yaml_sha256: hash of the data.yaml the eval was computed
            against. Same value across all arms.

    Returns:
        ``ArmMetrics`` ready to feed into the d-stage decision gate.

    Raises:
        ValueError: malformed inputs (NaN, negative counts, recall
            outside [0,1], etc.) OR
            ``(arm_id == "no_hn") != is_baseline_reference``.
    """
    # arm_id / is_baseline_reference cross-check per B2 review C4 — a
    # single self-explaining message rather than the confusing
    # ArmMetrics.__post_init__ cascade if the two disagree.
    if not isinstance(arm_id, str):
        raise ValueError(
            f"arm_id must be str; got {type(arm_id).__name__}={arm_id!r}"
        )
    if not isinstance(is_baseline_reference, bool):
        raise ValueError(
            f"is_baseline_reference must be bool; got "
            f"{type(is_baseline_reference).__name__}={is_baseline_reference!r}"
        )
    if (arm_id == "no_hn") != is_baseline_reference:
        raise ValueError(
            f"arm_id / is_baseline_reference mismatch: arm_id={arm_id!r}, "
            f"is_baseline_reference={is_baseline_reference}. The contract is "
            f"is_baseline_reference == (arm_id == 'no_hn'); pass them "
            f"together so the helper can build the baseline cell without "
            f"the dataclass-level cascade surfacing 'fp_drop_frac == 0' "
            f"errors that obscure the actual mismatch."
        )

    # FP counts: non-negative ints (bool-excluded).
    for name, val in (
        ("baseline_fp_count", baseline_fp_count),
        ("candidate_fp_count", candidate_fp_count),
    ):
        if not isinstance(val, int) or isinstance(val, bool):
            raise ValueError(
                f"{name} must be int; got {type(val).__name__}={val!r}"
            )
        if val < 0:
            raise ValueError(f"{name} must be >= 0; got {val}")

    # Recalls + total mAPs: finite floats in [0, 1].
    for name, val in (
        ("baseline_real_light_recall", baseline_real_light_recall),
        ("candidate_real_light_recall", candidate_real_light_recall),
        ("baseline_total_map", baseline_total_map),
        ("candidate_total_map", candidate_total_map),
    ):
        if not isinstance(val, float) or isinstance(val, bool):
            raise ValueError(
                f"{name} must be float; got {type(val).__name__}={val!r}"
            )
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite; got {val!r}")
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]; got {val}")

    # FP drop frac: positive = candidate has FEWER FPs (desired). The
    # max(.,1) floor avoids zero-divide when baseline already has zero
    # FPs (the metric is undefined there; the §4.7 rule's catch-all
    # handles the pathological case).
    fp_drop_frac = float(baseline_fp_count - candidate_fp_count) / float(
        max(baseline_fp_count, 1)
    )

    # Δrecall, ΔmAP in pp; the math returns 0 for self-comparison and
    # ArmMetrics.__post_init__ enforces the zero-delta invariant when
    # is_baseline_reference=True. No need to short-circuit.
    real_light_recall_delta_pp = (
        candidate_real_light_recall - baseline_real_light_recall
    ) * 100.0
    total_map_delta_pp = (
        candidate_total_map - baseline_total_map
    ) * 100.0

    return ArmMetrics(
        arm_id=arm_id,
        is_baseline_reference=is_baseline_reference,
        fp_drop_frac=fp_drop_frac,
        real_light_recall_delta_pp=real_light_recall_delta_pp,
        total_map_delta_pp=total_map_delta_pp,
        map_no_regression=map_no_regression,
        map_regression_tolerance_pp=map_regression_tolerance_pp,
        eval_manifest_sha256=eval_manifest_sha256,
        fp_manifest_sha256=fp_manifest_sha256,
        data_yaml_sha256=data_yaml_sha256,
    )
