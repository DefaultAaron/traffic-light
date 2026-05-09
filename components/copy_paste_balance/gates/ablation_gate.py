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
from dataclasses import dataclass
from typing import ClassVar, Iterable


def _is_hex_sha256(value: object) -> bool:
    """True iff ``value`` is a 64-char lowercase hex string."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


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
        if not _is_hex_sha256(self.eval_manifest_sha256):
            raise ValueError(
                f"eval_manifest_sha256 must be 64-char lowercase hex; got "
                f"{self.eval_manifest_sha256!r}"
            )
        if not _is_hex_sha256(self.fp_manifest_sha256):
            raise ValueError(
                f"fp_manifest_sha256 must be 64-char lowercase hex; got "
                f"{self.fp_manifest_sha256!r}"
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

    Returns:
        ``ArmMetrics`` ready to feed into the d-stage decision gate.

    Raises:
        ValueError: class-set mismatch between baseline / candidate; empty
            inputs; malformed safety_class_ids; etc.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
