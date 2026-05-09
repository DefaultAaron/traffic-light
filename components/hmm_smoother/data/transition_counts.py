"""Estimate raw transition counts from JSONL track replays.

Input format (mirrors ``inference/cpp/src/demo.cpp`` ``writeTrackJson`` and the
Python ``demo --json`` mode): one JSON record per line with at least the
fields::

    {
      "frame": int,
      "tracking_id": int,
      "raw_class_id": int,
      "raw_confidence": float,
      "class_probs": [float, ...]   # optional; only present in soft-mode
    }

The fitting pipeline (b-stage):

  1. Group records by ``tracking_id`` and sort within group by ``frame``.
  2. Drop tracks shorter than ``min_track_length`` (plan §2.2.1 c-stage
     eligible-tracks rule: length ≥ 5 frames). Note that ID-switch handling
     happens upstream in the runner — this module trusts the ``tracking_id``
     field and does NOT re-derive it.
  3. For each remaining track, walk consecutive frame pairs; for each pair,
     increment ``counts[i, j]`` where ``i = raw_class_id_{t}`` and
     ``j = raw_class_id_{t+1}``. Frame gaps (a track missed a frame) are
     handled per ``gap_policy`` (see below).
  4. Emit the (C, C) integer count matrix.

Gap policy is pre-committed at fit time:

  * ``"strict"``  — drop transition pairs that span a frame gap > 1.
                     Conservative; biases counts toward dense tracks.
  * ``"bridge"``  — count the (i → j) pair regardless of gap size, treating
                     the missed frame as if the state held. Aligns with the
                     EMA baseline's missed-frame behavior; default.

Scaffold (a-stage): function signature only.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np


class GapPolicy(str, Enum):
    STRICT = "strict"
    BRIDGE = "bridge"


def estimate_from_jsonl(
    jsonl_path: str | Path,
    *,
    num_classes: int,
    min_track_length: int = 5,
    gap_policy: GapPolicy = GapPolicy.BRIDGE,
) -> np.ndarray:
    """Build the raw transition-count matrix from a JSONL track replay.

    Source-of-truth precedence (B2 review S5 2026-05-09): the function-
    level defaults here (``min_track_length=5``, ``gap_policy=BRIDGE``)
    match the plan §2.2.1 c eligible-tracks rule and the
    ``configs/temporal_hmm.yaml`` defaults at scaffold time. **At
    runtime the YAML wins**: the runner MUST pass these args explicitly
    from the loaded YAML, not rely on the function defaults. Treat the
    Python defaults as documentation of the intended values, not as an
    independent source of truth.

    Args:
        jsonl_path: path to a JSONL emitted by ``demo --json``.
        num_classes: state-space size; must match the detector's class count
            and the eventual ``TransitionConfig.num_classes``.
        min_track_length: tracks with fewer than this many frames are dropped
            entirely (plan §2.2.1 c eligible-tracks rule, default 5).
        gap_policy: how to treat missed frames inside a track (see
            module docstring).

    Returns:
        shape (num_classes, num_classes) int64 count matrix.

    Raises:
        ValueError: malformed JSONL, ``raw_class_id`` outside [0, num_classes),
            or no tracks survive ``min_track_length``.
        FileNotFoundError: ``jsonl_path`` does not exist.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
