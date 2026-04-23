"""Quantitative validation of TrackSmoother flicker reduction.

The real `demo/demo.mp4` is clean enough that raw detections rarely flip
class, so it doesn't exercise the reduction path. This script drives
TrackSmoother with a deterministic noisy detection stream and reports the
raw-vs-smoothed flip reduction against the ≥50% decision gate.

Scenario: 300 frames, one stable box at a fixed position. The "true" class
is red (0) for the first half and green (2) for the second half. On each
frame a noise class is substituted with probability `flip_rate` (default
0.3), uniformly drawn from the other classes. This mirrors small-light
regimes where YOLO's per-frame argmax is noisy between visually similar
classes.

Usage:
    uv run python scripts/validate_flicker_reduction.py
    uv run python scripts/validate_flicker_reduction.py --seed 1 --flip-rate 0.4
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.tracker import TrackSmoother  # noqa: E402
from inference.trt_pipeline import Detection  # noqa: E402


@dataclass
class Scenario:
    frames: int = 300
    num_classes: int = 7
    flip_rate: float = 0.3
    seed: int = 0
    transition_at: float = 0.5  # fraction of frames where true class changes


def _build_stream(s: Scenario) -> list[tuple[int, int]]:
    """Return list of (true_class, observed_class) per frame."""
    rng = random.Random(s.seed)
    transition = int(s.frames * s.transition_at)
    true_classes = [0] * transition + [2] * (s.frames - transition)
    stream: list[tuple[int, int]] = []
    for tc in true_classes:
        if rng.random() < s.flip_rate:
            choices = [c for c in range(s.num_classes) if c != tc]
            oc = rng.choice(choices)
        else:
            oc = tc
        stream.append((tc, oc))
    return stream


def _flip_count(seq: list[int]) -> int:
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def run(s: Scenario) -> dict:
    stream = _build_stream(s)
    ts = TrackSmoother(num_classes=s.num_classes, alpha=0.3, min_hits=3)

    raw_seq: list[int] = []
    smoothed_seq: list[int] = []
    for i, (_true, obs) in enumerate(stream):
        det = Detection(class_id=obs, confidence=0.9, x1=100, y1=100, x2=120, y2=150)
        out = ts.update([det], i)
        if out:
            smoothed_seq.append(out[0].class_id)
            raw_seq.append(out[0].raw_class_id)

    raw_flips = _flip_count(raw_seq)
    smoothed_flips = _flip_count(smoothed_seq)
    reduction = (1.0 - smoothed_flips / raw_flips) * 100 if raw_flips else 0.0

    return {
        "frames_in_stream": s.frames,
        "frames_emitted": len(smoothed_seq),
        "raw_flips": raw_flips,
        "smoothed_flips": smoothed_flips,
        "reduction_pct": reduction,
        "target_pct": 50.0,
        "passed": reduction >= 50.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Synthetic TrackSmoother flicker reduction validation.")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--flip-rate", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    result = run(Scenario(frames=args.frames, flip_rate=args.flip_rate, seed=args.seed))
    print(f"Scenario:          frames={args.frames} flip_rate={args.flip_rate} seed={args.seed}")
    print(f"Frames emitted:    {result['frames_emitted']}")
    print(f"Raw flips:         {result['raw_flips']}")
    print(f"Smoothed flips:    {result['smoothed_flips']}")
    print(f"Reduction:         {result['reduction_pct']:+.1f}%  (target ≥ {result['target_pct']:.0f}%)")
    print(f"Decision gate:     {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
