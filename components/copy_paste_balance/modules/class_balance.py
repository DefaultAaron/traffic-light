"""Class-balanced loss weights — Cui et al. 2019 effective-number-of-samples.

Reference: Cui, Jia, Lin, Song, Belongie, "Class-Balanced Loss Based on
Effective Number of Samples," CVPR 2019.

Formula (load-bearing — do NOT re-derive in b-stage):

    effective_num(c)  = (1 - β^N(c)) / (1 - β)         for β ∈ [0, 1)
    weight(c)         = 1 / effective_num(c)
    weight_normalized = weight * (num_classes / sum(weight))

where N(c) is the per-class instance count from the training set. The
final normalization step makes ``mean(weight_normalized) == 1.0`` so the
weighted loss has the same average magnitude as the unweighted loss
(stabilizes optimizer learning rate).

β interpretation (Cui 2019 §3):

  * β = 0       : uniform weights (all classes equal). The "no class-balance"
                   baseline arm in this ablation uses this.
  * β = 0.99    : mild reweighting; common starting point for moderately
                   imbalanced datasets.
  * β = 0.999   : Cui paper's CIFAR-LT headline value.
  * β = 0.9999  : aggressive reweighting; risk of rare-class FP explosion
                   per plan §3.5.

The c-stage sensitivity sweep over β ∈ {0.0, 0.99, 0.999, 0.9999} is
plan-pinned. The β=0.0 case is the cp_only arm (copy-paste on,
class-balance off); β > 0 cells are the cp_balanced arm.

Apply mode dispatch (b-stage decides per detector family):

  * **ultralytics_class_weights**: per-class weight tensor passed via a
    training callback that patches the loss module's class-weight buffer.
    Plan default for Ultralytics. Requires version-pinned callback API
    check at b-stage.
  * **ultralytics_cls_scalar**: collapse the per-class vector to a scalar
    Ultralytics ``cls=`` gain (mean of weights). Lossy fallback when the
    callback-patch path isn't viable; documented as approximation only.
    NEVER the default — only used when the runner reports the version
    pin failed.
  * **deim_focal_alpha**: replace DEIM's focal-loss α with the
    per-class weights. b-stage path; DEIM uses focal loss by default per
    plan §3.1.

Scaffold (a-stage): API frozen as ``NotImplementedError`` stubs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ClassBalanceApplyMode(str, Enum):
    """How the per-class weight vector flows into the trainer."""

    ULTRALYTICS_CLASS_WEIGHTS = "ultralytics_class_weights"
    ULTRALYTICS_CLS_SCALAR = "ultralytics_cls_scalar"
    DEIM_FOCAL_ALPHA = "deim_focal_alpha"


@dataclass(frozen=True)
class ClassBalanceWeights:
    """Pre-computed per-class loss weights from class counts + β.

    Construction is a one-shot computation: counts + β → weight vector
    (normalized so mean is 1.0). The dataclass holds the result so
    repeated construction inside a training loop doesn't re-do the
    arithmetic.

    β=0 semantics (B2 review S5 2026-05-09): β=0 is a valid degenerate
    case the formula handles via short-circuit. The Cui formula
    ``effective = (1 - β^N) / (1 - β)`` simplifies to ``effective = 1``
    for every class when β=0, so all per-class weights collapse to 1
    after normalization (uniform). The runner's β=0 branch can skip
    the formula entirely; the dataclass accepts β=0 as the cp_only-arm
    default. See ``from_counts`` step 1 commentary for the same point
    spelled out at the formula level.

    Cross-check (b-stage validates at construction): weights are strictly
    positive, mean is 1.0 ± 1e-6, length matches num_classes.
    """

    num_classes: int
    beta: float
    apply_mode: ClassBalanceApplyMode
    max_weight_ratio: float                 # clamp ratio of max(w) / min(w)
    weights: tuple[float, ...]              # length C, mean ≈ 1.0

    def __post_init__(self) -> None:
        # System-boundary validation — direct construction (e.g. tests, b-stage
        # harness) MUST surface malformed input as ValueError, not a numpy /
        # downstream-trainer error 30 minutes into a training run.
        if self.num_classes is None:
            raise ValueError("num_classes must be set explicitly; got None")
        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool):
            raise ValueError(
                f"num_classes must be int; got "
                f"{type(self.num_classes).__name__}={self.num_classes!r}"
            )
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0; got {self.num_classes}")
        if not isinstance(self.beta, float) or isinstance(self.beta, bool):
            raise ValueError(
                f"beta must be float; got "
                f"{type(self.beta).__name__}={self.beta!r}"
            )
        if not math.isfinite(self.beta):
            raise ValueError(f"beta must be finite; got {self.beta!r}")
        # β=1.0 makes the formula divide by zero (1 - β^N) / (1 - β); reject
        # at the boundary rather than producing inf/NaN weights downstream.
        if not (0.0 <= self.beta < 1.0):
            raise ValueError(
                f"beta must be in [0, 1); got {self.beta} "
                f"(β=1.0 is forbidden — formula divides by (1 - β))"
            )
        if not isinstance(self.apply_mode, ClassBalanceApplyMode):
            raise ValueError(
                f"apply_mode must be ClassBalanceApplyMode enum; got "
                f"{type(self.apply_mode).__name__}={self.apply_mode!r}"
            )
        if not isinstance(self.max_weight_ratio, float) or isinstance(self.max_weight_ratio, bool):
            raise ValueError(
                f"max_weight_ratio must be float; got "
                f"{type(self.max_weight_ratio).__name__}={self.max_weight_ratio!r}"
            )
        if not math.isfinite(self.max_weight_ratio):
            raise ValueError(
                f"max_weight_ratio must be finite; got {self.max_weight_ratio!r}"
            )
        if self.max_weight_ratio < 1.0:
            raise ValueError(
                f"max_weight_ratio must be >= 1.0 (1.0 = no clamp); got "
                f"{self.max_weight_ratio}"
            )
        if not isinstance(self.weights, tuple):
            raise ValueError(
                f"weights must be tuple; got {type(self.weights).__name__}"
            )
        if len(self.weights) != self.num_classes:
            raise ValueError(
                f"weights length ({len(self.weights)}) must equal num_classes "
                f"({self.num_classes})"
            )
        for i, w in enumerate(self.weights):
            if not isinstance(w, float) or isinstance(w, bool):
                raise ValueError(
                    f"weights[{i}] must be float; got "
                    f"{type(w).__name__}={w!r}"
                )
            if not math.isfinite(w):
                raise ValueError(f"weights[{i}] must be finite; got {w!r}")
            if w <= 0.0:
                raise ValueError(
                    f"weights[{i}] must be > 0; got {w} "
                    f"(zero/negative loss weight breaks gradient flow)"
                )
        # C3 iter-6 NEW-MAJOR 2026-05-09: enforce the documented
        # postconditions on the weight vector at construction. Without
        # these, a hand-built `(100.0, 100.0, ...)` or an over-clamped
        # rare-class vector would pass construction and silently change
        # the loss scale or rare-class pressure during training,
        # corrupting the cp_balanced ablation arm while still looking
        # schema-clean downstream. b-stage's `from_counts` MUST normalize
        # so mean=1 and clamp so max/min <= max_weight_ratio; mirror at
        # the dataclass boundary so direct construction (tests / b-stage
        # harness) can't bypass it.
        mean_w = sum(self.weights) / len(self.weights)
        if not math.isclose(mean_w, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"mean(weights) must be 1.0 ± 1e-6 (the Cui 2019 "
                f"normalization step makes the weighted loss have the same "
                f"average magnitude as the unweighted loss); got mean={mean_w}"
            )
        ratio = max(self.weights) / min(self.weights)
        # +1e-9 tolerance so a vector that was clamped exactly at the limit
        # (max(w)/min(w) == max_weight_ratio after fp arithmetic) doesn't
        # spuriously fail. Anything past that is a real clamp violation.
        if ratio > self.max_weight_ratio + 1e-9:
            raise ValueError(
                f"max(weights) / min(weights) must be <= max_weight_ratio "
                f"({self.max_weight_ratio}); got ratio={ratio} "
                f"(plan §3.5 risk: large weights → rare-class FP explosion; "
                f"clamp must be applied BEFORE normalization in `from_counts`)"
            )

    @classmethod
    def from_counts(
        cls,
        *,
        counts: np.ndarray,
        beta: float,
        apply_mode: ClassBalanceApplyMode,
        max_weight_ratio: float = 10.0,
    ) -> "ClassBalanceWeights":
        """Compute Cui 2019 effective-number-of-samples weights from counts.

        Implementation outline (b-stage spells out — order matters for
        numerical stability):

          1. ``effective = (1 - beta ** counts) / (1 - beta)`` element-wise.
             For ``beta == 0``: ``effective = 1.0`` element-wise (degenerate
             case → uniform weights). The runner's β=0 branch can short-
             circuit this entire computation.
          2. ``raw_weights = 1.0 / effective`` (zero-count classes need a
             floor: clamp counts to ``max(counts, 1)`` BEFORE step 1; b-stage
             logs a WARNING per zero-count class so the dataset gap surfaces).
          3. Clamp ``raw_weights`` so ``max(raw_weights) / min(raw_weights)
             <= max_weight_ratio`` per plan §3.5 risk (rare-class FP
             explosion).
          4. Normalize: ``weights = raw_weights * num_classes / sum(raw_weights)``.
             Post-condition: ``mean(weights) == 1.0`` exactly (within fp64
             precision); b-stage asserts this.

        Args:
            counts: shape (C,) integer instance counts from the training set.
            beta: Cui 2019 β parameter; must be in [0, 1).
            apply_mode: how the resulting weight vector flows into the
                trainer (forwarded onto the returned dataclass; the
                constructor doesn't pre-apply mode-specific transforms).
            max_weight_ratio: clamp on max(w)/min(w).

        Returns:
            ``ClassBalanceWeights`` with normalized per-class weights.

        Raises:
            ValueError: counts has wrong shape / dtype / negatives;
                β out of range; max_weight_ratio < 1.0.
            NotImplementedError: a-stage scaffold.
        """
        raise NotImplementedError("b-stage")
