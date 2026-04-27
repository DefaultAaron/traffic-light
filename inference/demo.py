"""Demo script for TRT inference pipeline — test on video/camera with visualization.

Usage:
    python -m inference.demo --source video.mp4 --model best.engine
    python -m inference.demo --source 0 --model best.onnx  # webcam
    python -m inference.demo --source video.mp4 --model best.engine --track
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from inference.tracker import TrackedDetection, TrackSmoother
from inference.trt_pipeline import CLASS_NAMES, Detection, TRTDetector

# Colors for each class (BGR for OpenCV) — shared by raw detections and tracker output.
CLASS_COLORS = {
    0: (0, 0, 255),      # red
    1: (0, 255, 255),    # yellow
    2: (0, 255, 0),      # green
    3: (0, 0, 180),      # redLeft
    4: (0, 180, 0),      # greenLeft
    5: (128, 0, 255),    # redRight
    6: (128, 255, 0),    # greenRight
}


def _label(det) -> str:
    if isinstance(det, TrackedDetection):
        return f"#{det.tracking_id} {det.class_name} {det.confidence:.2f}"
    return f"{det.class_name} {det.confidence:.2f}"


def _color(det) -> tuple:
    return CLASS_COLORS.get(det.class_id, (255, 255, 255))


def draw_detections(frame: np.ndarray, detections: list, fps: float | None = None) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        color = _color(det)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        # Thicker box for tracker output so it's distinguishable from raw detector output
        # in side-by-side comparison videos. Class color stays semantic.
        thickness = 3 if isinstance(det, TrackedDetection) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = _label(det)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return frame


def run_video(
    source: str | int,
    detector: TRTDetector,
    show: bool = True,
    save: str | None = None,
    output_json: bool = False,
    tracker: TrackSmoother | None = None,
    track_json: str | None = None,
):
    """Run detection (and optional tracking) on a video source."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    track_json_fp = open(track_json, "w") if track_json else None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save, fourcc, input_fps, (w, h))

    frame_count = 0
    total_detect_time = 0.0
    total_track_time = 0.0
    fps_display = 0.0

    info = sys.stderr if output_json else sys.stdout
    print(f"Input: {source} ({w}x{h} @ {input_fps:.1f}fps)", file=info)
    print(f"Model: conf_thresh={detector.conf_thresh}, imgsz={detector.imgsz}", file=info)
    if tracker is not None:
        print(
            f"Tracker: alpha={tracker.alpha}, min_hits={tracker.min_hits}, "
            f"track_thresh={tracker.track_thresh}, high_thresh={tracker.high_thresh}",
            file=info,
        )
    print("Press 'q' to quit\n", file=info)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            raw_dets: list[Detection] = detector.detect(frame)
            t1 = time.perf_counter()
            dt_detect = t1 - t0

            # 0-indexed frame number passed to the tracker. Held separately
            # because frame_count below is post-incremented for the human-
            # readable "Frame N" log line — but --track-json must match the
            # C++ tl_demo schema which is 0-indexed for parity diffing.
            track_frame_idx = frame_count

            if tracker is not None:
                tracked = tracker.update(raw_dets, track_frame_idx)
                dt_track = time.perf_counter() - t1
                display_dets: list = tracked
            else:
                dt_track = 0.0
                display_dets = raw_dets

            total_detect_time += dt_detect
            total_track_time += dt_track
            frame_count += 1
            fps_display = frame_count / (total_detect_time + total_track_time)

            if output_json:
                record = {
                    "frame": frame_count,
                    "timestamp_ms": round((total_detect_time + total_track_time) * 1000, 1),
                    "inference_ms": round(dt_detect * 1000, 1),
                    "track_ms": round(dt_track * 1000, 2),
                    "detections": [d.to_dict() for d in display_dets],
                }
                print(json.dumps(record), flush=True)

            if track_json_fp is not None and tracker is not None:
                tracks_payload = [
                    {
                        "tracking_id": t.tracking_id,
                        "class_id": t.class_id,
                        "class_name": t.class_name,
                        "confidence": t.confidence,
                        "x1": t.x1, "y1": t.y1, "x2": t.x2, "y2": t.y2,
                        "age": t.age,
                        "hits": t.hits,
                        "raw_class_id": t.raw_class_id,
                        "raw_confidence": t.raw_confidence,
                        "class_probs": list(t.class_probs),
                    }
                    for t in display_dets
                ]
                track_json_fp.write(
                    json.dumps({"frame": track_frame_idx, "tracks": tracks_payload}) + "\n"
                )

            if frame_count % 30 == 0 and not output_json:
                suffix = f" track {dt_track*1000:.2f}ms" if tracker is not None else ""
                print(
                    f"Frame {frame_count}: {len(display_dets)} detections, "
                    f"{dt_detect*1000:.1f}ms{suffix} "
                    f"({fps_display:.1f} FPS avg)",
                    file=info,
                )

            vis = draw_detections(frame.copy(), display_dets, fps_display) if show or writer else frame

            if writer:
                writer.write(vis)

            if show:
                cv2.imshow("Traffic Light Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if track_json_fp is not None:
            track_json_fp.close()
        if show:
            cv2.destroyAllWindows()

    avg_ms = ((total_detect_time + total_track_time) / frame_count * 1000) if frame_count else 0
    print(
        f"\nSummary: {frame_count} frames, avg {avg_ms:.1f}ms/frame ({fps_display:.1f} FPS)",
        file=info,
    )


def main():
    parser = argparse.ArgumentParser(description="Traffic Light Detection Demo")
    parser.add_argument("--source", required=True, help="Video file path or camera index (0, 1, ...)")
    parser.add_argument("--model", required=True, help="Model path (.engine or .onnx)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size (default: 1280 for 8MP cameras)")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    parser.add_argument("--save", type=str, default=None, help="Save output video to path")
    parser.add_argument("--json", action="store_true", help="Output per-frame JSON to stdout")
    parser.add_argument("--track", action="store_true",
                        help="Enable ByteTrack + EMA class voting")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="EMA smoothing coefficient (default: 0.3; lower = smoother)")
    parser.add_argument("--min-hits", type=int, default=3,
                        help="Minimum observations before a track is emitted (default: 3)")
    parser.add_argument("--high-thresh", type=float, default=0.5,
                        help="High/low detection split for two-pass association (default: 0.5)")
    parser.add_argument("--track-buffer", type=int, default=30,
                        help="Frames to keep lost tracks alive (default: 30 = ~1s @ 30fps)")
    parser.add_argument("--track-json", type=str, default=None,
                        help="Write per-frame tracker output as JSONL (one {frame,tracks} object "
                             "per line, schema matches C++ tl_demo --track-json for parity diffing)")
    args = parser.parse_args()

    # Parse source: integer for camera, string for file
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    detector = TRTDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
    )

    tracker = None
    if args.track:
        tracker = TrackSmoother(
            num_classes=len(CLASS_NAMES),
            alpha=args.alpha,
            track_thresh=args.conf,
            high_thresh=args.high_thresh,
            min_hits=args.min_hits,
            track_buffer=args.track_buffer,
        )

    if args.track_json and tracker is None:
        parser.error("--track-json requires --track")

    run_video(
        source, detector,
        show=not args.no_show,
        save=args.save,
        output_json=args.json,
        tracker=tracker,
        track_json=args.track_json,
    )


if __name__ == "__main__":
    main()
