"""Observation likelihood model for the HMM smoother.

The HMM observes per-frame detector output and needs an emission likelihood
``b_i(o_t) = P(o_t | y_t = i)`` per state ``i``. Two source formats matter:

  1. Soft :  ``raw_confidence`` PLUS a full softmax ``class_probs`` vector
              (length C). This is the preferred path when the detector
              exposes class probabilities (TRTDetector ``class_probs``
              field after R1 tracker change).
  2. Hard :  ``raw_class_id`` (int) PLUS ``raw_confidence`` only. We
              synthesize an emission distribution that places
              ``raw_confidence`` mass on the argmax class and spreads the
              residual ``1 - raw_confidence`` uniformly over the other
              C-1 classes. Used when JSONL replays predate the
              soft-output tracker change, OR when the detector path
              doesn't expose probs.

b-stage will implement the actual likelihood computation; a-stage exposes
the signature so the inference algorithms can be written against it.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np


class ObservationMode(str, Enum):
    """Where the per-frame emission distribution comes from.

    Pre-committed at config time so a JSONL replay can't silently switch
    paths mid-stream: an ``ObservationModel`` configured for ``SOFT`` that
    encounters a frame missing ``class_probs`` raises rather than falling
    back to ``HARD``. (Mixed-source replays must be split before driving
    the smoother.)
    """

    SOFT = "soft"
    HARD = "hard"


class ObservationModel:
    """Per-frame emission likelihood ``P(observation | state)``.

    Construction binds a fixed ``num_classes`` and ``ObservationMode``;
    every frame fed to ``emission()`` MUST match that mode. The output
    is a length-C array that the HMM inference algorithms multiply
    elementwise against the forward / backward state distributions.
    """

    def __init__(
        self,
        num_classes: int,
        mode: ObservationMode = ObservationMode.SOFT,
        epsilon: float = 1e-6,
    ):
        # Codex stop-gate 2026-05-09: parallel hardening to match
        # HmmYamlConfig.__post_init__ — direct ObservationModel construction
        # is a public boundary and needs the same validation discipline.
        if num_classes is None:
            raise ValueError("num_classes must be set explicitly; got None")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool):
            raise ValueError(
                f"num_classes must be int; got "
                f"{type(num_classes).__name__}={num_classes!r}"
            )
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0; got {num_classes}")
        if not isinstance(mode, ObservationMode):
            raise ValueError(
                f"mode must be ObservationMode enum; got "
                f"{type(mode).__name__}={mode!r}"
            )
        if not isinstance(epsilon, float) or isinstance(epsilon, bool):
            raise ValueError(
                f"epsilon must be float; got "
                f"{type(epsilon).__name__}={epsilon!r}"
            )
        if not math.isfinite(epsilon):
            raise ValueError(f"epsilon must be finite; got {epsilon!r}")
        if not (0.0 < epsilon < 1.0 / num_classes):
            raise ValueError(
                f"epsilon must be in (0, 1/num_classes={1.0 / num_classes:g}); "
                f"got {epsilon}"
            )
        self._num_classes = num_classes
        self._mode = mode
        self._epsilon = epsilon

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def mode(self) -> ObservationMode:
        return self._mode

    def emission(
        self,
        *,
        class_probs: np.ndarray | None = None,
        raw_class_id: int | None = None,
        raw_confidence: float | None = None,
    ) -> np.ndarray:
        """Return the emission likelihood vector for one frame's observation.

        Exactly one of the following parameter combinations must be supplied,
        and it must agree with the mode bound at construction:

          * mode=SOFT  : ``class_probs`` (shape (C,), sums to 1).
          * mode=HARD  : ``raw_class_id`` (int in [0, C)) AND ``raw_confidence``
                          (float in (0, 1]).

        ``epsilon`` is added to every entry of the emission vector before it
        is returned, so a frame whose argmax is "yellow" with confidence 1.0
        does NOT produce a likelihood-zero state for "red" (which would make
        any subsequent red detection unreachable in Viterbi). The post-add
        vector is renormalized to sum to 1.

        HARD-mode ε guidance (B2 review I2 2026-05-09): the YAML default
        ``observation_epsilon: 1e-6`` is appropriate for SOFT mode (where
        the detector softmax usually floors at 1e-3 to 1e-2), but is too
        tight for HARD mode at C ≥ 10 — a 100-frame track with a persistent
        argmax can rail-pin the MAP path against any transition prior.
        For HARD-mode replays, configure ``observation_epsilon: 1e-3`` (or
        scale to ``1 / num_classes / 100``). The constructor permits
        ``epsilon`` up to ``1 / num_classes`` so this scaling is in range
        for C up to a few thousand.

        Returns:
            length-C float array with strictly positive entries summing to 1.

        Raises:
            ValueError: input combination disagrees with ``mode``, or shape /
                range / sum violations on the supplied input.
            NotImplementedError: a-stage scaffold.
        """
        raise NotImplementedError("b-stage")
