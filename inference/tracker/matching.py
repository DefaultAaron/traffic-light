"""IoU cost + linear assignment for ByteTrack.

Vendored from ByteTrack (MIT, Copyright (c) 2021 Yifu Zhang).
Upstream: https://github.com/ifzhang/ByteTrack
Upstream commit: d1bf0191adff59bc8fcfeaa0b33d3d1642552a99

Local changes vs upstream:
- Replaced `cython_bbox.bbox_overlaps` with a numpy-only IoU (removes the
  cython_bbox build dependency; equivalent numerics).
- Replaced `lap.lapjv` with `scipy.optimize.linear_sum_assignment`
  (scipy is already a transitive dep via ultralytics; drops the `lap` dep).
- Dropped unused helpers: merge_matches, embedding_distance, v_iou_distance,
  gate_cost_matrix, fuse_motion, fuse_iou — not called by BYTETracker.update.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix, thresh):
    """Hungarian assignment with a cost ceiling.

    Returns (matches, unmatched_a, unmatched_b) where matches is an (M, 2)
    int array of (row, col) pairs whose cost is <= thresh; unmatched_* are
    int arrays of leftover indices on each side.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    # linear_sum_assignment minimizes total cost over a full assignment.
    # Entries above `thresh` are infeasible — set them to a large cost so
    # the solver avoids them, then filter post-hoc.
    masked = np.where(cost_matrix > thresh, 1e9, cost_matrix)
    row_ind, col_ind = linear_sum_assignment(masked)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
    matches = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)

    matched_rows = set(matches[:, 0].tolist()) if matches.size else set()
    matched_cols = set(matches[:, 1].tolist()) if matches.size else set()
    unmatched_a = np.array(
        [i for i in range(cost_matrix.shape[0]) if i not in matched_rows], dtype=int
    )
    unmatched_b = np.array(
        [i for i in range(cost_matrix.shape[1]) if i not in matched_cols], dtype=int
    )
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """Pairwise IoU over (N, 4) and (M, 4) tlbr arrays. Returns (N, M) matrix."""
    a = np.asarray(atlbrs, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(btlbrs, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def iou_distance(atracks, btracks):
    """IoU cost matrix (1 - IoU) over STrack lists or raw tlbr arrays."""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    return 1.0 - ious(atlbrs, btlbrs)


def fuse_score(cost_matrix, detections):
    """Multiply (1 - IoU-cost) by detection scores to bias assignment toward
    confident detections. Matches upstream semantics.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1.0 - fuse_sim
