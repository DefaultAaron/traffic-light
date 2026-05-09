"""c-stage acceptance metrics for the HMM ablation.

Computes the three pre-committed metrics from
``docs/planning/temporal_optimization_plan.md`` §2.2.1 c on a paired
(baseline, candidate) replay:

  1. ``flicker_rate``        — per eligible track: argmax-change count
                                 divided by adjacent-valid-frame-pair count;
                                 then averaged across eligible tracks.
  2. ``illegal_transition``  — absolute count of (i → j) transitions in
                                 ``illegal_transition_set`` over eligible
                                 tracks. Plan note: count, not rate.
  3. ``map_no_regression``   — placeholder hook; for the ablation we replay
                                 a fixed JSONL so per-class AP doesn't
                                 change between baseline and candidate
                                 (same boxes, only labels differ). The mAP
                                 check is enforced upstream by the runner
                                 against the frozen ``runs/_r2_val_manifest``;
                                 this module exposes the pass/fail field
                                 so the d-stage rule can read it.

Eligible-tracks rule (plan §2.2.1 c):
  * track length ≥ 5 frames in the replay window
  * no ByteTrack ID switch inside the window — switches are detected
    upstream by the runner; this module trusts the ``eligible`` flag

Scaffold (a-stage): API signatures only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TrackSequence:
    """One eligible track as a flat sequence of per-frame class assignments.

    Both baseline (EMA) and candidate (HMM) sequences are the SMOOTHED
    output, not the raw detector argmax. The flicker / illegal-transition
    metrics are computed on the smoothed stream.
    """

    track_id: int
    eligible: bool
    smoothed_classes: tuple[int, ...]      # length T


@dataclass(frozen=True)
class GateMetrics:
    """Aggregated c-stage metrics for one side of the A/B.

    ``total_transitions`` (B2 review I1 2026-05-09) is the denominator
    used by the d-stage gate's illegal-rate computation
    (``illegal_transition_count / total_transitions``). It is the count
    of adjacent valid frame-pair transitions on ELIGIBLE tracks only,
    consistent with how ``flicker_rate`` is computed.

    Cross-side invariants (C3 iter-1 NEW-MAJOR 2 + iter-2 NEW-MAJOR 3
    2026-05-09): the same eligible-tracks rule applied to the same
    JSONL EVAL split MUST produce identical denominator-class fields on
    both baseline and candidate sides — the smoothing choice changes
    labels, not the set of (track, frame-pair) units the labels are
    computed over. The d-stage executor enforces ALL of:

      * ``baseline.total_transitions == candidate.total_transitions``
      * ``baseline.eligible_track_count == candidate.eligible_track_count``
      * ``baseline.total_track_count == candidate.total_track_count``

    Any mismatch surfaces as ``HmmDecision.EXECUTOR_ERROR``. Either
    side with ``total_transitions == 0`` (no eligible tracks long
    enough to span even one frame pair) is also ``EXECUTOR_ERROR`` —
    malformed input, not a legitimate run.

    The serialized schema (``_hmm_decision_schema.json``) splits this
    field into ``baseline_total_transitions`` + ``candidate_total_transitions``
    so denominator drift, when it occurs, is grep-able in the artifact;
    the executor reads both before applying the equality invariant.
    """

    flicker_rate: float
    illegal_transition_count: int
    total_transitions: int
    eligible_track_count: int
    total_track_count: int


def compute_metrics(
    sequences: Iterable[TrackSequence],
    *,
    illegal_transition_set: tuple[tuple[int, int], ...],
) -> GateMetrics:
    """Compute the three c-stage metrics on a single side of the A/B.

    Args:
        sequences: eligible-flagged smoothed track sequences (baseline OR
            candidate, not both — call once per side).
        illegal_transition_set: pre-committed set of forbidden ``(src, dst)``
            class-id pairs from ``configs/temporal_hmm.yaml``. Same set used
            on both sides of the A/B; otherwise the comparison is invalid.

    Returns:
        ``GateMetrics`` aggregating across eligible tracks; ineligible
        tracks contribute to ``total_track_count`` but not to the
        per-track flicker mean or illegal-transition tally.

    Raises:
        ValueError: empty input, or illegal-transition cells out of range
            relative to the smoothed class IDs observed.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
