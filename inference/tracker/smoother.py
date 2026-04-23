"""TrackSmoother: ByteTrack association + per-track EMA class voting.

Sits on top of the per-frame `TRTDetector`. Input is a list of `Detection`
objects (xyxy in source-image pixels, single argmax class + score). Output
is a list of `TrackedDetection` with a persistent `tracking_id` and a
smoothed per-class probability vector.

Design notes in `docs/integration/tracker_voting_guide.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from inference.trt_pipeline import Detection
from inference.tracker.basetrack import BaseTrack
from inference.tracker.byte_tracker import BYTETracker, STrack


@dataclass
class TrackedDetection(Detection):
    """Per-frame detection with persistent track identity + EMA-smoothed class.

    Inherits `class_id`, `confidence`, `x1/y1/x2/y2` from `Detection` with the
    smoothed class/confidence. `raw_class_id` / `raw_confidence` preserve the
    single-frame argmax for offline diffing.
    """

    tracking_id: int = -1
    age: int = 0                 # frames since this track was first seen
    hits: int = 0                # frames where this track had a measurement
    raw_class_id: int = -1
    raw_confidence: float = 0.0
    class_probs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["tracking_id"] = self.tracking_id
        d["age"] = self.age
        d["hits"] = self.hits
        d["raw_class_id"] = self.raw_class_id
        d["raw_confidence"] = self.raw_confidence
        d["class_probs"] = list(self.class_probs)
        return d


class TrackSmoother:
    """ByteTrack-based tracker with per-track EMA class voting.

    The tracker runs class-agnostic IoU matching (class-id does not gate
    association — that's the whole point, since single-frame class flips are
    what we want to smooth out). Each confirmed track maintains a softmax-
    shaped `class_probs` vector updated as:

        p <- alpha * one_hot(raw_class_id) + (1 - alpha) * p

    Tracks with `hits < min_hits` are suppressed from the output to filter
    one-shot false positives.

    Note on ID scope: track IDs come from `BaseTrack._count`, a class-level
    counter shared across all `BYTETracker` / `TrackSmoother` instances in
    the process. For multi-camera deployment you must either (a) keep each
    camera in its own process, or (b) namespace the emitted IDs downstream.
    `reset()` resets this counter — not safe to call while another smoother
    is live.
    """

    def __init__(
        self,
        num_classes: int = 7,
        alpha: float = 0.3,
        track_thresh: float = 0.25,
        high_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_hits: int = 3,
        frame_rate: int = 30,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive; got {num_classes}")
        if high_thresh < track_thresh:
            raise ValueError(
                f"high_thresh ({high_thresh}) must be >= track_thresh ({track_thresh})"
            )

        self.num_classes = num_classes
        self.alpha = alpha
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits

        self._tracker = BYTETracker(
            track_thresh=high_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=False,
            frame_rate=frame_rate,
        )
        self._state: dict[int, _TrackState] = {}

    def reset(self) -> None:
        """Clear all tracks and reset the global id counter."""
        self._tracker.reset()
        self._state.clear()

    def update(
        self, detections: list[Detection], frame_idx: int
    ) -> list[TrackedDetection]:
        # Drop detections below the low cutoff — everything above goes to the
        # tracker, which splits them into high/low internally.
        dets_filtered = [d for d in detections if d.confidence >= self.track_thresh]

        if dets_filtered:
            dets_arr = np.array(
                [[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in dets_filtered],
                dtype=float,
            )
        else:
            dets_arr = np.zeros((0, 5), dtype=float)

        active = self._tracker.update(dets_arr)
        current = self._tracker.frame_id

        output: list[TrackedDetection] = []

        for strack in active:
            tid = strack.track_id
            state = self._state.get(tid)
            got_measurement = strack.frame_id == current

            if got_measurement:
                det_idx = _best_iou_match(strack.tlbr, dets_filtered)
                if det_idx is not None:
                    raw_cls = dets_filtered[det_idx].class_id
                    raw_conf = dets_filtered[det_idx].confidence
                    box = _detection_box(dets_filtered[det_idx])
                    if state is None:
                        state = _TrackState(
                            first_frame=frame_idx,
                            last_seen_frame=frame_idx,
                            hits=1,
                            class_probs=_one_hot(raw_cls, self.num_classes),
                            last_raw_class_id=raw_cls,
                        )
                        self._state[tid] = state
                    else:
                        state.hits += 1
                        state.last_seen_frame = frame_idx
                        state.class_probs = (
                            self.alpha * _one_hot(raw_cls, self.num_classes)
                            + (1.0 - self.alpha) * state.class_probs
                        )
                        state.last_raw_class_id = raw_cls
                else:
                    # Tracker says this strack was updated, but we can't locate
                    # the matched detection in our filtered list. Treat as a
                    # measurement-less step so we don't double-count hits or
                    # echo-smooth class_probs toward the stale class.
                    if state is None:
                        continue
                    raw_cls = state.last_raw_class_id
                    raw_conf = 0.0
                    box = strack.tlbr
            else:
                if state is None:
                    # Track exists in tracker state but we never saw it with a
                    # measurement. Skip; it'll be picked up when it re-acquires.
                    continue
                raw_cls = state.last_raw_class_id
                raw_conf = 0.0
                box = strack.tlbr

            if state.hits < self.min_hits:
                continue

            smoothed_cls = int(np.argmax(state.class_probs))
            smoothed_conf = float(state.class_probs[smoothed_cls])

            output.append(
                TrackedDetection(
                    class_id=smoothed_cls,
                    confidence=smoothed_conf,
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    tracking_id=tid,
                    age=frame_idx - state.first_frame + 1,
                    hits=state.hits,
                    raw_class_id=int(raw_cls),
                    raw_confidence=float(raw_conf),
                    class_probs=state.class_probs.tolist(),
                )
            )

        # GC state for tracks that have fallen out of the tracker entirely.
        live_ids = {t.track_id for t in self._tracker.tracked_stracks + self._tracker.lost_stracks}
        stale = [tid for tid in self._state if tid not in live_ids]
        for tid in stale:
            del self._state[tid]

        return output


@dataclass
class _TrackState:
    first_frame: int
    last_seen_frame: int
    hits: int
    class_probs: np.ndarray
    last_raw_class_id: int


def _one_hot(class_id: int, num_classes: int) -> np.ndarray:
    v = np.zeros(num_classes, dtype=float)
    if 0 <= class_id < num_classes:
        v[class_id] = 1.0
    return v


def _detection_box(d: Detection) -> tuple[float, float, float, float]:
    return (d.x1, d.y1, d.x2, d.y2)


def _best_iou_match(
    box: np.ndarray, detections: list[Detection], min_iou: float = 0.1
) -> Optional[int]:
    """Find the detection index whose xyxy has highest IoU with `box`.
    Returns None if no detection clears `min_iou`.
    """
    if not detections:
        return None
    bx1, by1, bx2, by2 = box
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    best_iou = min_iou
    best_idx: Optional[int] = None
    for i, d in enumerate(detections):
        ix1 = max(bx1, d.x1)
        iy1 = max(by1, d.y1)
        ix2 = min(bx2, d.x2)
        iy2 = min(by2, d.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            continue
        d_area = max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1)
        union = b_area + d_area - inter
        if union <= 0:
            continue
        iou = inter / union
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx
