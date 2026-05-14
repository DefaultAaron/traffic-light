"""c-stage acceptance metrics for the copy-paste / class-balance ablation.

Computes the four pre-committed metrics from
``docs/planning/additional_components_plan.md`` §3.7 on a single
already-trained arm's eval output:

  1. ``rare_class_mean_delta_AP``   — mean ΔAP across rare classes
                                       (full_val_support < threshold).
  2. ``rare_class_max_delta_AP``    — max ΔAP across rare classes (used for
                                       the "≥1 rare class +5pp" deploy
                                       alternative in §3.7).
  3. ``rare_safety_min_delta_AP``   — min ΔAP across rare ∩ safety classes
                                       (used for the "all rare safety
                                       classes ≥ −1pp" guardrail).
  4. ``rare_related_fp_delta_frac`` — relative change in rare-class-related
                                       FP count on the §四 frozen manifest
                                       (signed fraction; +0.10 = 10% rise).
  5. ``total_map_delta_pp``         — total mAP@0.5 delta in percentage
                                       points (signed; negative = regression).

All deltas are ``arm_metric - baseline_metric`` where ``baseline_metric`` is
the no_aug arm's value. ΔAP fields are in percentage points (multiply
absolute AP delta by 100), matching plan §3.7 prose ("+5 pp", "−1 pp").

Frozen FP manifest contract (plan §3.7 + §4.7 share the same manifest):
  * The rare-related FP count is computed on
    ``runs/_hard_negative_eval_manifest.json`` (or the §三 equivalent
    if §四 hasn't landed it). Threshold: confidence ≥ 0.25, NMS IoU = 0.5.
  * The runner reads the manifest path + threshold from the YAML; mismatch
    between fit time and eval time is a hard fail (§3.7 anti-gaming
    safeguard — denominator must not change).

Eligible-classes rule:
  * Rare set = ``{c : full_val_support(c) < rare_class_threshold}``.
  * Rare-safety set = rare ∩ ``safety_class_names``.
  * If a rare class has zero instances on either side (no validation support),
    it is excluded from mean/min computations and surfaced in
    ``ArmMetrics.zero_support_rare_classes`` for the d-stage rule's
    ``executor_error`` gate.

Scaffold (a-stage): API signatures only.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import ClassVar, Iterable

# B2 iter-3 review: cross-module hash predicate lives in _internals so
# the runner doesn't reach across module boundary into a private symbol.
# B2 iter-4 review: import the public name directly without an alias —
# call style at local callsites now reads ``is_hex_sha256(value)``,
# mirroring HN's sister gate exactly.
from components.copy_paste_balance._internals import is_hex_sha256


@dataclass(frozen=True)
class PerClassAP:
    """One class's AP@0.5 + support count from one arm's eval.

    Both ``ap_at_0_5`` and ``full_val_support`` come from the eval JSON
    that was produced by the trainer. The ablation gate consumes pairs
    of these (one for the no_aug baseline, one for an experimental arm)
    and computes the delta metrics.

    Validation discipline (C3 NEW-MAJOR 1 2026-05-09): same boundary
    rigor as ``CopyPasteBalanceYamlConfig`` / ``AblationConfig`` — this
    is the c-stage metric boundary feeding deploy/defer/drop, so a
    malformed row that bypasses validation can silently alter the
    headline decision before the runner's output-schema validation
    catches it.
    """

    class_id: int
    class_name: str
    ap_at_0_5: float            # absolute AP, range [0, 1]
    full_val_support: int       # instance count on the eval split

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, int) or isinstance(self.class_id, bool):
            raise ValueError(
                f"class_id must be int; got "
                f"{type(self.class_id).__name__}={self.class_id!r}"
            )
        if self.class_id < 0:
            raise ValueError(f"class_id must be >= 0; got {self.class_id}")
        if not isinstance(self.class_name, str):
            raise ValueError(
                f"class_name must be str; got {type(self.class_name).__name__}"
            )
        if not self.class_name:
            raise ValueError("class_name must be non-empty")
        if not isinstance(self.ap_at_0_5, float) or isinstance(self.ap_at_0_5, bool):
            raise ValueError(
                f"ap_at_0_5 must be float; got "
                f"{type(self.ap_at_0_5).__name__}={self.ap_at_0_5!r}"
            )
        if not math.isfinite(self.ap_at_0_5):
            raise ValueError(f"ap_at_0_5 must be finite; got {self.ap_at_0_5!r}")
        if not (0.0 <= self.ap_at_0_5 <= 1.0):
            raise ValueError(
                f"ap_at_0_5 must be in [0, 1]; got {self.ap_at_0_5}"
            )
        if not isinstance(self.full_val_support, int) or isinstance(self.full_val_support, bool):
            raise ValueError(
                f"full_val_support must be int; got "
                f"{type(self.full_val_support).__name__}={self.full_val_support!r}"
            )
        if self.full_val_support < 0:
            raise ValueError(
                f"full_val_support must be >= 0; got {self.full_val_support}"
            )


@dataclass(frozen=True)
class ArmMetrics:
    """Aggregated c-stage metrics for ONE arm minus the baseline arm.

    All delta fields are computed as ``arm - baseline``; the baseline arm
    itself is represented by an ``ArmMetrics`` whose deltas are all
    exactly zero (with ``is_baseline_reference=True`` so the d-stage rule
    can short-circuit).

    Cross-arm invariants (the runner enforces — d-stage gate trusts):
      * The class IDs and names are identical across arms (same
        ``data.yaml`` was used to train all three arms).
      * The eval manifest hash is identical across arms (same
        ``runs/_r2_val_manifest.sha256`` consumed).
      * The frozen FP manifest hash is identical across arms.
      * The rare-class set is computed on the BASELINE arm's
        ``full_val_support`` only — switching rare-set membership per
        arm would let an arm that depleted a rare class's support
        appear "deployed" without actually improving the rare class.
    """

    arm_id: str                                 # "no_aug" | "cp_only" | "cp_balanced"
    is_baseline_reference: bool
    rare_class_mean_delta_AP_pp: float
    rare_class_max_delta_AP_pp: float
    rare_safety_min_delta_AP_pp: float
    rare_related_fp_delta_frac: float
    total_map_delta_pp: float
    rare_class_ids: tuple[int, ...]             # plan §3.7 rare set (baseline-derived)
    rare_safety_class_ids: tuple[int, ...]      # rare ∩ safety
    zero_support_rare_classes: tuple[int, ...]  # excluded from mean/min
    eval_manifest_sha256: str                   # frozen val manifest hash
    fp_manifest_sha256: str                     # §3.7+§4.7 shared manifest hash
    # C3 iter-9 NEW-MAJOR 2026-05-09: decision-provenance fields.
    # Without these per-cell, a hand-built artifact can claim
    # ``decision: "deploy"`` while hiding the upstream mAP verdict /
    # tolerance / data.yaml hash that made the decision possible.
    map_no_regression: bool                     # upstream eval verdict (THIS arm vs baseline)
    map_regression_tolerance_pp: float          # tolerance applied; runner-side knob
    data_yaml_sha256: str                       # eval source data.yaml hash; class-label provenance

    _ALLOWED_ARM_IDS: ClassVar[tuple[str, ...]] = ("no_aug", "cp_only", "cp_balanced")
    # Schema MetricsBlock ranges (mirror _copy_paste_decision_schema.json):
    _DELTA_AP_PP_MIN: ClassVar[float] = -100.0
    _DELTA_AP_PP_MAX: ClassVar[float] = 100.0
    _FP_DELTA_FRAC_MIN: ClassVar[float] = -1.0
    _FP_DELTA_FRAC_MAX: ClassVar[float] = 10.0

    def __post_init__(self) -> None:
        # C3 NEW-MAJOR 1 2026-05-09: same boundary discipline as the
        # rest of the scaffold. The schema enforces these ranges on the
        # serialized output; mirror at the dataclass boundary so the
        # decision-rule executor doesn't see malformed input that the
        # schema would reject post-hoc.
        if not isinstance(self.arm_id, str):
            raise ValueError(
                f"arm_id must be str; got {type(self.arm_id).__name__}={self.arm_id!r}"
            )
        if self.arm_id not in self._ALLOWED_ARM_IDS:
            raise ValueError(
                f"arm_id must be one of {self._ALLOWED_ARM_IDS}; got {self.arm_id!r}"
            )
        if not isinstance(self.is_baseline_reference, bool):
            raise ValueError(
                f"is_baseline_reference must be bool; got "
                f"{type(self.is_baseline_reference).__name__}"
            )
        # Cross-knob: arm_id == "no_aug" iff is_baseline_reference. The runner
        # constructs the baseline ArmMetrics from a self-comparison; the
        # invariant must hold both ways or downstream rare-set derivation
        # picks the wrong arm.
        if (self.arm_id == "no_aug") != self.is_baseline_reference:
            raise ValueError(
                f"is_baseline_reference must be True iff arm_id == 'no_aug'; "
                f"got arm_id={self.arm_id!r}, is_baseline_reference={self.is_baseline_reference}"
            )
        # Numeric delta fields: bool-exclusion + finite + range.
        for field_name, field_value, lo, hi in (
            ("rare_class_mean_delta_AP_pp", self.rare_class_mean_delta_AP_pp,
             self._DELTA_AP_PP_MIN, self._DELTA_AP_PP_MAX),
            ("rare_class_max_delta_AP_pp", self.rare_class_max_delta_AP_pp,
             self._DELTA_AP_PP_MIN, self._DELTA_AP_PP_MAX),
            ("rare_safety_min_delta_AP_pp", self.rare_safety_min_delta_AP_pp,
             self._DELTA_AP_PP_MIN, self._DELTA_AP_PP_MAX),
            ("total_map_delta_pp", self.total_map_delta_pp,
             self._DELTA_AP_PP_MIN, self._DELTA_AP_PP_MAX),
            ("rare_related_fp_delta_frac", self.rare_related_fp_delta_frac,
             self._FP_DELTA_FRAC_MIN, self._FP_DELTA_FRAC_MAX),
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
        # Baseline-reference invariant: all delta fields are exactly 0
        # (constructed by self-comparison; non-zero would indicate a
        # programming error in the runner's baseline arm construction).
        if self.is_baseline_reference:
            for field_name, field_value in (
                ("rare_class_mean_delta_AP_pp", self.rare_class_mean_delta_AP_pp),
                ("rare_class_max_delta_AP_pp", self.rare_class_max_delta_AP_pp),
                ("rare_safety_min_delta_AP_pp", self.rare_safety_min_delta_AP_pp),
                ("rare_related_fp_delta_frac", self.rare_related_fp_delta_frac),
                ("total_map_delta_pp", self.total_map_delta_pp),
            ):
                if field_value != 0.0:
                    raise ValueError(
                        f"is_baseline_reference=True requires {field_name} == 0; "
                        f"got {field_value} (the no_aug arm is constructed by "
                        f"self-comparison so every delta must be exactly 0)"
                    )
        # Class-id tuple fields: tuple-of-int, unique, in [0, ∞). The schema
        # enforces uniqueItems on the serialized output; mirror here.
        for field_name, field_value in (
            ("rare_class_ids", self.rare_class_ids),
            ("rare_safety_class_ids", self.rare_safety_class_ids),
            ("zero_support_rare_classes", self.zero_support_rare_classes),
        ):
            if not isinstance(field_value, tuple):
                raise ValueError(
                    f"{field_name} must be tuple; got "
                    f"{type(field_value).__name__}={field_value!r} "
                    f"(loader / runner must coerce list → tuple before construction)"
                )
            for i, cid in enumerate(field_value):
                if not isinstance(cid, int) or isinstance(cid, bool):
                    raise ValueError(
                        f"{field_name}[{i}] must be int; got "
                        f"{type(cid).__name__}={cid!r}"
                    )
                if cid < 0:
                    raise ValueError(
                        f"{field_name}[{i}]={cid} must be >= 0"
                    )
            if len(set(field_value)) != len(field_value):
                raise ValueError(
                    f"{field_name} must contain no duplicates; got {field_value}"
                )
        # Subset invariants: rare_safety ⊆ rare; zero_support ⊆ rare. The
        # runner derives both from rare_class_ids; mismatches indicate a
        # construction bug.
        rare_set = set(self.rare_class_ids)
        if not set(self.rare_safety_class_ids).issubset(rare_set):
            raise ValueError(
                f"rare_safety_class_ids must be a subset of rare_class_ids; "
                f"got rare_safety={self.rare_safety_class_ids}, "
                f"rare={self.rare_class_ids}"
            )
        if not set(self.zero_support_rare_classes).issubset(rare_set):
            raise ValueError(
                f"zero_support_rare_classes must be a subset of rare_class_ids; "
                f"got zero_support={self.zero_support_rare_classes}, "
                f"rare={self.rare_class_ids}"
            )
        # Manifest hashes: 64-char lowercase hex (matches schema regex).
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
        # C3 iter-9 NEW-MAJOR (decision-provenance) 2026-05-09:
        # map_no_regression must be plain bool. For the baseline arm
        # the value is trivially True (self-comparison has zero mAP
        # delta), and the dataclass enforces that consistency below.
        if not isinstance(self.map_no_regression, bool):
            raise ValueError(
                f"map_no_regression must be bool; got "
                f"{type(self.map_no_regression).__name__}={self.map_no_regression!r}"
            )
        if self.is_baseline_reference and not self.map_no_regression:
            raise ValueError(
                f"is_baseline_reference=True requires map_no_regression=True "
                f"(baseline is constructed by self-comparison; mAP delta is "
                f"exactly 0, no regression possible); got map_no_regression={self.map_no_regression}"
            )
        # map_regression_tolerance_pp: float-with-bool-exclusion + finite
        # + range [0, 0.5] (mirrors DecisionInputs ceiling at the §3.7
        # drop threshold; rule is incoherent if tolerance > drop).
        if not isinstance(self.map_regression_tolerance_pp, float) or isinstance(self.map_regression_tolerance_pp, bool):
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
        if not (0.0 <= self.map_regression_tolerance_pp <= 0.5):
            raise ValueError(
                f"map_regression_tolerance_pp must be in [0, 0.5]; got "
                f"{self.map_regression_tolerance_pp} (cap mirrors "
                f"DecisionInputs.DROP_TOTAL_MAP_REGRESSION_PP)"
            )


def compute_arm_metrics(
    *,
    arm_id: str,
    baseline_per_class: Iterable[PerClassAP],
    candidate_per_class: Iterable[PerClassAP],
    baseline_total_map: float,
    candidate_total_map: float,
    baseline_rare_fp_count: int,
    candidate_rare_fp_count: int,
    rare_class_threshold: int,
    safety_class_ids: tuple[int, ...],
    eval_manifest_sha256: str,
    fp_manifest_sha256: str,
    # C3 iter-10 NEW-MAJOR 2026-05-09: producer signature must accept the
    # three iter-9 ArmMetrics fields, otherwise b-stage cannot build a
    # legal ArmMetrics from this helper. For the baseline self-comparison
    # the caller passes map_no_regression=True.
    map_no_regression: bool,
    map_regression_tolerance_pp: float,
    data_yaml_sha256: str,
) -> ArmMetrics:
    """Compute the §3.7 metric block on a single arm vs the baseline.

    Calling convention (b-stage spells out — order matters for the rare
    set membership invariant):

      1. Compute rare set from BASELINE per-class supports only:
         ``rare_class_ids = sorted(c.class_id for c in baseline_per_class
                                   if c.full_val_support < rare_class_threshold)``
      2. ``rare_safety_class_ids = sorted(set(rare_class_ids) & set(safety_class_ids))``
      3. Pair candidate to baseline by class_id; reject if the sets differ.
      4. Compute mean / max ΔAP_pp across rare; min ΔAP_pp across rare-safety.
      5. ``rare_related_fp_delta_frac = (cand_fp - base_fp) / max(base_fp, 1)``
         — the ``max(., 1)`` floor avoids zero-divide; b-stage logs a WARNING
         when ``base_fp == 0`` (rare-FP rate already at floor).

    For the BASELINE arm itself: pass the same iterable for both
    ``baseline_per_class`` and ``candidate_per_class`` (and same totals).
    The function returns an ``ArmMetrics`` with all-zero deltas and
    ``is_baseline_reference=True``.

    Args:
        arm_id: ``"no_aug"`` (baseline) or ``"cp_only"`` or ``"cp_balanced"``.
        baseline_per_class: per-class AP from the no_aug arm.
        candidate_per_class: per-class AP from THIS arm.
        baseline_total_map: total mAP@0.5 from the no_aug arm.
        candidate_total_map: total mAP@0.5 from this arm.
        baseline_rare_fp_count: rare-class-related FP count on the §3.7+§4.7
            shared frozen manifest, no_aug arm.
        candidate_rare_fp_count: same, this arm.
        rare_class_threshold: full_val_support threshold for "rare" (plan
            default 30).
        safety_class_ids: pre-committed safety-critical class IDs (subset
            of all class IDs).
        eval_manifest_sha256: hash of the frozen R2 val manifest.
        fp_manifest_sha256: hash of the frozen §3.7+§4.7 shared FP manifest.
        map_no_regression: upstream eval verdict for THIS arm vs baseline
            (computed from candidate_total_map - baseline_total_map vs
            map_regression_tolerance_pp). For arm_id == "no_aug" the
            caller MUST pass True (self-comparison).
        map_regression_tolerance_pp: the tolerance threshold the runner
            applied when computing ``map_no_regression``. Recorded into
            the ArmMetrics for audit. Same value MUST be passed for
            every arm in a single ablation run.
        data_yaml_sha256: hash of the data.yaml the eval was computed
            against. Class-label provenance — same value across all arms.

    Returns:
        ``ArmMetrics`` ready to feed into the d-stage decision gate.

    Raises:
        ValueError: class-set mismatch between baseline / candidate; empty
            inputs; malformed safety_class_ids; etc.
    """
    # Materialize iterables (caller may pass generators; we walk twice).
    baseline_list = list(baseline_per_class)
    candidate_list = list(candidate_per_class)

    if not baseline_list:
        raise ValueError("baseline_per_class must not be empty")
    if not candidate_list:
        raise ValueError("candidate_per_class must not be empty")
    for i, p in enumerate(baseline_list):
        if not isinstance(p, PerClassAP):
            raise ValueError(
                f"baseline_per_class[{i}] must be PerClassAP; got "
                f"{type(p).__name__}={p!r}"
            )
    for i, p in enumerate(candidate_list):
        if not isinstance(p, PerClassAP):
            raise ValueError(
                f"candidate_per_class[{i}] must be PerClassAP; got "
                f"{type(p).__name__}={p!r}"
            )

    # arm_id is validated by ArmMetrics.__post_init__; this helper derives
    # is_baseline_reference from it (CPB has three arm_ids and the
    # dataclass invariant pins "no_aug ↔ baseline", so the helper signal
    # is unambiguous without an explicit kwarg — diverges from §四 sister
    # by design; see HN ablation_gate B2 review C4 rationale).
    if not isinstance(arm_id, str):
        raise ValueError(
            f"arm_id must be str; got {type(arm_id).__name__}={arm_id!r}"
        )
    is_baseline_reference = arm_id == "no_aug"

    # rare_class_threshold: strictly positive int (bool-excluded).
    if not isinstance(rare_class_threshold, int) or isinstance(
        rare_class_threshold, bool
    ):
        raise ValueError(
            f"rare_class_threshold must be int; got "
            f"{type(rare_class_threshold).__name__}={rare_class_threshold!r}"
        )
    if rare_class_threshold <= 0:
        raise ValueError(
            f"rare_class_threshold must be > 0; got {rare_class_threshold}"
        )

    # safety_class_ids: tuple of distinct non-negative ints (mirrors
    # ArmMetrics's rare_class_ids discipline for the dataclass boundary).
    if not isinstance(safety_class_ids, tuple):
        raise ValueError(
            f"safety_class_ids must be tuple; got "
            f"{type(safety_class_ids).__name__}={safety_class_ids!r}"
        )
    for i, cid in enumerate(safety_class_ids):
        if not isinstance(cid, int) or isinstance(cid, bool):
            raise ValueError(
                f"safety_class_ids[{i}] must be int; got "
                f"{type(cid).__name__}={cid!r}"
            )
        if cid < 0:
            raise ValueError(
                f"safety_class_ids[{i}]={cid} must be >= 0"
            )
    if len(set(safety_class_ids)) != len(safety_class_ids):
        raise ValueError(
            f"safety_class_ids must contain no duplicates; got "
            f"{safety_class_ids}"
        )

    # FP counts: non-negative ints (bool-excluded).
    for name, val in (
        ("baseline_rare_fp_count", baseline_rare_fp_count),
        ("candidate_rare_fp_count", candidate_rare_fp_count),
    ):
        if not isinstance(val, int) or isinstance(val, bool):
            raise ValueError(
                f"{name} must be int; got {type(val).__name__}={val!r}"
            )
        if val < 0:
            raise ValueError(f"{name} must be >= 0; got {val}")

    # Total mAPs: finite floats in [0, 1] (absolute AP range).
    for name, val in (
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

    # Class-set parity: same class IDs in the same order. The order is
    # part of the contract — per-class arrays from the trainer are
    # produced in class-ID order and downstream schema validation expects
    # consistent indexing.
    baseline_ids = tuple(p.class_id for p in baseline_list)
    candidate_ids = tuple(p.class_id for p in candidate_list)
    if baseline_ids != candidate_ids:
        # Build a diagnostic that names the divergence rather than dumping
        # both arrays.
        base_set = set(baseline_ids)
        cand_set = set(candidate_ids)
        only_in_baseline = sorted(base_set - cand_set)
        only_in_candidate = sorted(cand_set - base_set)
        if base_set == cand_set:
            diff = "same class set, different order"
        else:
            diff = (
                f"baseline-only={only_in_baseline}, "
                f"candidate-only={only_in_candidate}"
            )
        raise ValueError(
            f"baseline_per_class / candidate_per_class class IDs differ "
            f"({diff}). Class-set drift between arms indicates a trainer "
            f"or data_yaml mismatch — the rare set is derived from baseline "
            f"and the rule needs paired-class deltas."
        )

    # Rare set from BASELINE per-class supports only (per plan §3.7 — the
    # rare population is baseline-derived and propagated to every
    # candidate cell; switching the rare set per arm would let an arm
    # that depleted a rare class's support appear "deployed").
    baseline_by_id = {p.class_id: p for p in baseline_list}
    candidate_by_id = {p.class_id: p for p in candidate_list}
    rare_class_ids = tuple(
        sorted(
            p.class_id
            for p in baseline_list
            if p.full_val_support < rare_class_threshold
        )
    )
    safety_set = set(safety_class_ids)
    rare_safety_class_ids = tuple(
        sorted(set(rare_class_ids) & safety_set)
    )

    # Zero-support rare classes: support == 0 on EITHER side. Excluded
    # from rare mean/max/safety_min computation but kept in
    # rare_class_ids for cross-arm equality + downstream audit.
    zero_support_rare_classes = tuple(
        sorted(
            cid
            for cid in rare_class_ids
            if baseline_by_id[cid].full_val_support == 0
            or candidate_by_id[cid].full_val_support == 0
        )
    )
    zero_support_set = set(zero_support_rare_classes)
    eligible_rare = [
        cid for cid in rare_class_ids if cid not in zero_support_set
    ]
    eligible_rare_safety = [
        cid
        for cid in rare_safety_class_ids
        if cid not in zero_support_set
    ]

    # Per-class delta in percentage points. The math returns 0 for the
    # baseline self-comparison; ArmMetrics.__post_init__ then enforces
    # the zero-delta invariant for is_baseline_reference=True. No need
    # to short-circuit here.
    def _delta_pp(cid: int) -> float:
        return (
            candidate_by_id[cid].ap_at_0_5 - baseline_by_id[cid].ap_at_0_5
        ) * 100.0

    if eligible_rare:
        rare_deltas = [_delta_pp(cid) for cid in eligible_rare]
        rare_mean = sum(rare_deltas) / len(rare_deltas)
        rare_max = max(rare_deltas)
    else:
        # No eligible rare classes (either rare set is empty OR every
        # rare class has zero support somewhere). The deploy guard's
        # 5pp threshold will trivially fail; the rule falls through to
        # defer or drop. Recording 0.0 keeps the dataclass invariants
        # satisfied (range / finiteness).
        rare_mean = 0.0
        rare_max = 0.0
    if eligible_rare_safety:
        rare_safety_min = min(_delta_pp(cid) for cid in eligible_rare_safety)
    else:
        # B2 review MAJOR-2 fix: empty eligible_rare_safety is one of two
        # cases — both must be visible to the operator:
        #   (a) rare_safety_class_ids is empty (no rare ∩ safety overlap)
        #       — legitimate but unusual; means the deploy gate's
        #       `≥ -1pp` rare-safety guard collapses to "no rare safety
        #       to protect". Likely a YAML misconfiguration if R2 had
        #       safety classes that became rare.
        #   (b) rare_safety_class_ids is non-empty but every entry has
        #       zero support on either side — pathological; the
        #       safety guard silently passes on a population that
        #       can't be measured.
        # Both produce rare_safety_min=0.0 which trivially satisfies the
        # deploy guard. Surface via UserWarning so the runner / operator
        # sees the issue at eval time rather than discovering it from a
        # spurious deploy decision.
        if rare_safety_class_ids:
            warnings.warn(
                f"compute_arm_metrics({arm_id=}): all rare-safety classes "
                f"({list(rare_safety_class_ids)}) have zero support on "
                f"either baseline or candidate (zero_support="
                f"{list(zero_support_rare_classes)}). rare_safety_min "
                f"defaults to 0.0 and trivially passes the deploy "
                f"`≥ -1pp` safety guard — a wrongful deploy on this arm "
                f"would not be caught by the safety mechanism.",
                UserWarning,
                stacklevel=2,
            )
        elif rare_class_ids:
            # rare_class_ids non-empty but safety∩rare is empty — the
            # YAML's safety_class_ids has no overlap with the
            # baseline-derived rare set. Usually means R2's safety list
            # doesn't include any rare class; review safety_class_ids.
            warnings.warn(
                f"compute_arm_metrics({arm_id=}): rare set "
                f"{list(rare_class_ids)} has zero overlap with the YAML "
                f"safety_class_ids — the deploy gate's rare-safety "
                f"guard collapses to 'no rare safety to protect'. Verify "
                f"that the safety class list is intentional.",
                UserWarning,
                stacklevel=2,
            )
        rare_safety_min = 0.0

    # FP delta as signed fraction; max(.,1) floor avoids zero-divide
    # when baseline already has zero rare-FPs (the metric is undefined
    # there; the §3.7 rule's drop catch-all handles the pathological case).
    rare_related_fp_delta_frac = float(
        candidate_rare_fp_count - baseline_rare_fp_count
    ) / float(max(baseline_rare_fp_count, 1))

    # Total mAP delta in pp.
    total_map_delta_pp = (
        candidate_total_map - baseline_total_map
    ) * 100.0

    return ArmMetrics(
        arm_id=arm_id,
        is_baseline_reference=is_baseline_reference,
        rare_class_mean_delta_AP_pp=rare_mean,
        rare_class_max_delta_AP_pp=rare_max,
        rare_safety_min_delta_AP_pp=rare_safety_min,
        rare_related_fp_delta_frac=rare_related_fp_delta_frac,
        total_map_delta_pp=total_map_delta_pp,
        rare_class_ids=rare_class_ids,
        rare_safety_class_ids=rare_safety_class_ids,
        zero_support_rare_classes=zero_support_rare_classes,
        eval_manifest_sha256=eval_manifest_sha256,
        fp_manifest_sha256=fp_manifest_sha256,
        map_no_regression=map_no_regression,
        map_regression_tolerance_pp=map_regression_tolerance_pp,
        data_yaml_sha256=data_yaml_sha256,
    )
