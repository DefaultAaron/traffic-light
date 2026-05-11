"""HMM inference algorithms: forward-backward (smoothing) + Viterbi (MAP).

Plan §2.2 calls out both as acceptable inference paths. The c-stage A/B
ablation runs the same `(TransitionMatrix, ObservationModel)` pair through
both modes and reports per-track flicker rate / illegal-transition count
for each, so the deploy decision (§2.2 决策规则) can pick whichever path the
data favors — typically Viterbi for MAP-stable label sequences, but
forward-backward when downstream consumers want a calibrated posterior.

Scaffold (a-stage): both functions are signature-frozen ``NotImplementedError``
stubs. b-stage fills in vectorized numpy bodies; latency target per §2.2 is
< 0.01 ms, well inside numpy's window for C ≤ 14 and per-track sequence
length ≤ a few hundred frames.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from components.hmm_smoother.modules.observation import ObservationModel
from components.hmm_smoother.modules.transition import TransitionMatrix


class HmmInferenceMode(str, Enum):
    """Selector for the runner / gates layer; not consumed by the algos themselves."""

    FORWARD_BACKWARD = "forward_backward"
    VITERBI = "viterbi"


def forward_backward(
    *,
    observations: np.ndarray,
    transition: TransitionMatrix,
    observation_model: ObservationModel,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Posterior class distribution per frame via forward-backward.

    Returns the smoothed marginals ``γ_t(i) = P(y_t = i | o_{1:T})`` rather
    than only the forward filter — that's what the c-stage flicker-rate
    metric is computed on (argmax of γ vs argmax of EMA-smoothed prob).

    Numerical stability: b-stage MUST normalize the forward and backward
    messages at every step (or work entirely in log-space). The plan calls
    HMM "purely matrix-multiply", but unscaled multiplication over a
    100-frame track collapses to subnormal floats; the canonical scaled
    form (Rabiner 1989 §IV) is required.

    Args:
        observations: shape (T, ?) — emission inputs, format determined by
            ``observation_model.mode``. b-stage validates shape against
            ``observation_model.num_classes``.
        transition: pre-built ``TransitionMatrix``.
        observation_model: bound ``ObservationModel``.
        initial: optional length-C initial state distribution; defaults to
            uniform 1/C if omitted.

    Returns:
        shape (T, C) float array; each row is a non-negative distribution
        summing to 1 (within 1e-6 tolerance).

    Note: callers feeding the result into
    ``components.hmm_smoother.gates.ablation_gate.TrackSequence`` must take
    ``argmax(γ, axis=1).astype(np.int64)`` first — the gate consumes int
    sequences, not posteriors. ``posterior_argmax_sequence`` (below) is the
    canonical helper; use it rather than open-coding.

    Raises:
        ValueError: shape / range mismatches on inputs, or empty sequence.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")


def posterior_argmax_sequence(posterior: np.ndarray) -> np.ndarray:
    """Convert a forward-backward posterior to an int64 class sequence.

    Convenience wrapper so the runner doesn't open-code ``argmax`` and
    end up disagreeing with the Viterbi path on dtype / tie-breaking.
    ``np.argmax`` is left-most-wins on ties, matching ``viterbi``'s
    documented tie-break (lowest state index wins).

    Args:
        posterior: shape (T, C) float array; each row a probability
            distribution over states.

    Returns:
        shape (T,) int64 array.

    Raises:
        ValueError: empty input or non-2D shape.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")


def viterbi(
    *,
    observations: np.ndarray,
    transition: TransitionMatrix,
    observation_model: ObservationModel,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """MAP class sequence via the Viterbi algorithm.

    Returns the single most-likely state sequence given the observations,
    transition prior, and emission model — NOT per-frame argmax of the
    posterior (use ``forward_backward`` if that's what you want). The two
    outputs disagree on borderline frames where the smoothed marginal
    prefers state A but the joint MAP path runs through state B.

    Implementation notes for b-stage:
      * Work in log-space throughout — Viterbi is the canonical underflow
        offender for long sequences.
      * Tie-breaking when two predecessor states give equal log-probability
        is left-most (lowest state index) for determinism / parity-test
        friendliness; document this in the function body when filling it in.

    Args:
        observations: shape (T, ?) — see ``forward_backward``.
        transition: pre-built ``TransitionMatrix``.
        observation_model: bound ``ObservationModel``.
        initial: optional length-C initial state distribution; defaults to
            uniform 1/C if omitted.

    Returns:
        shape (T,) int64 array — the MAP class sequence.

    Raises:
        ValueError: shape / range mismatches on inputs, or empty sequence.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
