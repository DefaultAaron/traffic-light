"""Transition matrix for the HMM smoother.

Implements the C×C transition prior A_{ij} = P(y_t = j | y_{t-1} = i),
estimated from observed track sequences (see ``components.hmm_smoother.data
.transition_counts``) and post-processed with:

* Laplace smoothing — ``α`` pseudo-counts per cell. Plan default ``α=0.1``
  (``docs/planning/temporal_optimization_plan.md`` §2.2 conflictor iter-2
  amendment 2026-05-09): α=1.0 over a 14-class state space injects ≈ 50%
  uniform prior on sparse TL transition data; α=0.1 injects ~10% of typical
  row support. c-stage MUST sweep α ∈ {0.01, 0.1, 1.0} and the deploy
  decision MUST be anchored on a value that is stable across the sweep.
* Illegal-transition handling — controlled by
  ``illegal_transition_policy``:
    - ``"hard_zero"``  : forbidden cells set to 0 then row-renormalized
                          (plan default; aligns with §0.2 row 3 trigger
                          condition "eliminate illegal state transitions").
    - ``"downweight"`` : forbidden cells keep Laplace pseudo-count only;
                          let the data drive deviation from the prior.

Scaffold (a-stage): public surface frozen here as type-checked
``NotImplementedError`` stubs. b-stage replaces bodies with the
matrix-build path; no API change permitted without re-running the §2.2.1
adversarial loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class IllegalTransitionPolicy(str, Enum):
    """How to treat transitions listed in ``illegal_transition_set``."""

    HARD_ZERO = "hard_zero"
    DOWNWEIGHT = "downweight"


@dataclass(frozen=True)
class TransitionConfig:
    """Pre-committed knobs for transition-matrix construction.

    Mirrors ``configs/temporal_hmm.yaml`` so a YAML load → dataclass build is
    a one-liner once b-stage adds the loader. The dataclass is frozen so any
    accidental mid-run mutation surfaces as ``FrozenInstanceError`` rather
    than a silent prior shift.

    YAML→dataclass adapter note (B2 review S2 2026-05-09): YAML deserializes
    lists, not tuples. The loader MUST coerce ``illegal_transition_set``
    via ``tuple(map(tuple, yaml_value))`` and the policy field via
    ``IllegalTransitionPolicy(yaml_value)`` before constructing this
    dataclass — otherwise frozen-dataclass validation will surface as a
    confusing type error from the immutability check rather than the
    YAML loader.
    """

    num_classes: int
    laplace_alpha: float = 0.1
    illegal_transition_set: tuple[tuple[int, int], ...] = ()
    illegal_transition_policy: IllegalTransitionPolicy = (
        IllegalTransitionPolicy.HARD_ZERO
    )

    def __post_init__(self) -> None:
        # Codex stop-gate 2026-05-09: parallel hardening to match
        # HmmYamlConfig.__post_init__ — same bool-subclass / NaN / type-guard
        # hazards apply at this constructor boundary too. Direct construction
        # bypasses the YAML-config layer's validation, so both system
        # boundaries need the same discipline.
        if self.num_classes is None:
            raise ValueError(
                "num_classes must be set explicitly; got None"
            )
        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool):
            raise ValueError(
                f"num_classes must be int; got "
                f"{type(self.num_classes).__name__}={self.num_classes!r}"
            )
        if self.num_classes <= 0:
            raise ValueError(
                f"num_classes must be > 0; got {self.num_classes}"
            )
        if not isinstance(self.laplace_alpha, float) or isinstance(self.laplace_alpha, bool):
            raise ValueError(
                f"laplace_alpha must be float; got "
                f"{type(self.laplace_alpha).__name__}={self.laplace_alpha!r}"
            )
        if not math.isfinite(self.laplace_alpha):
            raise ValueError(
                f"laplace_alpha must be finite; got {self.laplace_alpha!r}"
            )
        if self.laplace_alpha < 0.0:
            raise ValueError(
                f"laplace_alpha must be >= 0; got {self.laplace_alpha}"
            )
        if not isinstance(self.illegal_transition_policy, IllegalTransitionPolicy):
            raise ValueError(
                f"illegal_transition_policy must be IllegalTransitionPolicy enum; "
                f"got {type(self.illegal_transition_policy).__name__}"
            )
        if not isinstance(self.illegal_transition_set, tuple):
            raise ValueError(
                f"illegal_transition_set must be tuple; got "
                f"{type(self.illegal_transition_set).__name__}={self.illegal_transition_set!r}"
            )
        for pair in self.illegal_transition_set:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    f"illegal_transition_set entries must be 2-tuples; got "
                    f"{type(pair).__name__}={pair!r}"
                )
            src, dst = pair
            for cell_label, cell in (("src", src), ("dst", dst)):
                if not isinstance(cell, int) or isinstance(cell, bool):
                    raise ValueError(
                        f"illegal_transition cell {cell_label} must be int; got "
                        f"{type(cell).__name__}={cell!r} in pair {pair!r}"
                    )
            if not (0 <= src < self.num_classes and 0 <= dst < self.num_classes):
                raise ValueError(
                    f"illegal_transition cell ({src},{dst}) out of range "
                    f"[0, {self.num_classes})"
                )


class TransitionMatrix:
    """Row-stochastic C×C transition matrix.

    Construction order (b-stage will implement, a-stage exposes signature):

      1. Start from raw integer count matrix ``counts`` (C×C, dtype=int64),
         e.g. produced by ``data.transition_counts.estimate_from_jsonl``.
      2. Add ``laplace_alpha`` to every cell (or to legal cells only when
         ``policy == HARD_ZERO``; illegal cells go to 0 in that case).
      3. Row-normalize. Empty rows (no observed transitions, all-zero after
         hard-zeroing) collapse to a uniform-over-legal-targets fallback to
         keep the matrix row-stochastic.
      4. Validate: every row sums to 1 within 1e-6; every entry is in [0, 1];
         every illegal cell under ``HARD_ZERO`` is exactly 0.

    The resulting matrix is exposed as ``A`` (read-only ``np.ndarray`` via
    ``.copy()`` only — internal storage is private to discourage in-place
    mutation across forward/backward passes).
    """

    def __init__(self, config: TransitionConfig):
        self._config = config
        self._A: np.ndarray | None = None  # populated by build()

    @property
    def config(self) -> TransitionConfig:
        return self._config

    @property
    def num_classes(self) -> int:
        return self._config.num_classes

    def build(self, counts: np.ndarray) -> None:
        """Construct the row-stochastic transition matrix from raw counts.

        Args:
            counts: shape (C, C) integer count matrix; ``counts[i, j]`` is
                the number of observed ``i → j`` transitions in the
                training/calibration track set.

        Raises:
            ValueError: shape mismatch with ``num_classes``, negative cells,
                or post-build validation failure (rows not summing to 1).
            NotImplementedError: a-stage scaffold.
        """
        raise NotImplementedError("b-stage")

    @property
    def A(self) -> np.ndarray:
        """Return a defensive copy of the transition matrix.

        Raises:
            RuntimeError: ``build()`` has not been called.
            NotImplementedError: a-stage scaffold.
        """
        raise NotImplementedError("b-stage")

    @classmethod
    def from_npy(cls, path: str, config: TransitionConfig) -> "TransitionMatrix":
        """Load a pre-fitted matrix from a .npy file with sanity checks.

        Re-runs the row-stochastic / illegal-cell validation against
        ``config`` so a stale on-disk artifact can't silently violate the
        active config (e.g. policy flipped from downweight → hard_zero
        between fit time and ablation time).

        Args:
            path: path to a ``np.save``'d (C, C) float matrix.
            config: the active ``TransitionConfig``; the loaded matrix must
                be compatible with this config under the validation rules.

        Raises:
            ValueError: shape / value / policy mismatch with ``config``.
            NotImplementedError: a-stage scaffold.
        """
        raise NotImplementedError("b-stage")
