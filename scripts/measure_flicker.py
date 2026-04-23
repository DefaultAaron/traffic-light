"""Measure per-track class flicker in demo `--json` output.

Input is a stream of newline-delimited JSON frames as emitted by
`inference/demo.py --json`. Each record has a `detections[]` list; entries
from a `--track` run also carry `tracking_id`, `class_id` (EMA-smoothed),
and `raw_class_id` (per-frame argmax).

This script attributes per-frame class labels to tracks (using
`tracking_id`) and counts consecutive-frame class changes — separately for
raw (pre-smoothing) and smoothed outputs — so a single `--track` run
directly yields the flicker-reduction delta.

Usage:
    python -m inference.demo --source demo/demo.mp4 --model weights/best.onnx \\
        --track --no-show --json > /tmp/tracked.jsonl
    python scripts/measure_flicker.py /tmp/tracked.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _iter_records(path: str | Path | None) -> Iterable[dict]:
    if path in (None, "-"):
        stream = sys.stdin
        yield from _iter_stream(stream)
        return
    with open(path) as f:
        yield from _iter_stream(f)


def _iter_stream(stream) -> Iterable[dict]:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            print(f"skip malformed line: {e}", file=sys.stderr)


def analyze(records: Iterable[dict]) -> dict:
    raw_per_track: dict[int, list[int]] = defaultdict(list)
    smoothed_per_track: dict[int, list[int]] = defaultdict(list)
    frame_count = 0
    total_dets = 0
    untracked_dets = 0

    for rec in records:
        frame_count += 1
        dets = rec.get("detections", [])
        total_dets += len(dets)
        for d in dets:
            tid = d.get("tracking_id")
            if tid is None:
                untracked_dets += 1
                continue
            smoothed_per_track[tid].append(int(d["class_id"]))
            raw_per_track[tid].append(int(d.get("raw_class_id", d["class_id"])))

    def _flip_count(seq: list[int]) -> int:
        return sum(1 for a, b in zip(seq, seq[1:]) if a != b)

    raw_flips = {tid: _flip_count(seq) for tid, seq in raw_per_track.items()}
    smoothed_flips = {tid: _flip_count(seq) for tid, seq in smoothed_per_track.items()}
    lifespans = {tid: len(seq) for tid, seq in smoothed_per_track.items()}

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "frames": frame_count,
        "total_detections": total_dets,
        "untracked_detections": untracked_dets,
        "unique_tracks": len(smoothed_per_track),
        "mean_lifespan": _mean(list(lifespans.values())),
        "raw_flips_per_track": _mean(list(raw_flips.values())),
        "smoothed_flips_per_track": _mean(list(smoothed_flips.values())),
        "raw_flips_total": sum(raw_flips.values()),
        "smoothed_flips_total": sum(smoothed_flips.values()),
        "per_track_raw_flips": raw_flips,
        "per_track_smoothed_flips": smoothed_flips,
        "per_track_lifespan": lifespans,
    }


def report(metrics: dict) -> None:
    print(f"Frames:                  {metrics['frames']}")
    print(f"Total detections:        {metrics['total_detections']}")
    print(f"Untracked detections:    {metrics['untracked_detections']}")
    print(f"Unique tracks:           {metrics['unique_tracks']}")
    print(f"Mean track lifespan:     {metrics['mean_lifespan']:.1f} frames")
    print()
    raw_mean = metrics["raw_flips_per_track"]
    smooth_mean = metrics["smoothed_flips_per_track"]
    print(f"Raw flips / track:       {raw_mean:.2f}  (total {metrics['raw_flips_total']})")
    print(f"Smoothed flips / track:  {smooth_mean:.2f}  (total {metrics['smoothed_flips_total']})")
    if raw_mean > 0:
        reduction = (1.0 - smooth_mean / raw_mean) * 100
        print(f"Flicker reduction:       {reduction:+.1f}%  (target ≥ 50%)")
    else:
        print("Flicker reduction:       n/a (no raw flips observed)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure per-track class flicker in demo JSON output.")
    parser.add_argument("input", nargs="?", default="-",
                        help="Path to JSONL from demo --json (default: stdin)")
    parser.add_argument("--dump-per-track", action="store_true",
                        help="Print per-track flip counts after the summary")
    args = parser.parse_args()

    metrics = analyze(_iter_records(args.input))
    report(metrics)

    if args.dump_per_track:
        print()
        print("Per-track detail (tid: lifespan raw_flips smoothed_flips):")
        for tid in sorted(metrics["per_track_lifespan"]):
            life = metrics["per_track_lifespan"][tid]
            raw = metrics["per_track_raw_flips"][tid]
            sm = metrics["per_track_smoothed_flips"][tid]
            print(f"  {tid}: {life:4d} frames  raw={raw:3d}  smoothed={sm:3d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
