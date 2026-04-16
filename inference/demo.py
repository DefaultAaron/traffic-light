"""Demo script for TRT inference pipeline — test on video/camera with visualization.

Usage:
    python -m inference.demo --source video.mp4 --model best.engine
    python -m inference.demo --source 0 --model best.onnx  # webcam
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from inference.trt_pipeline import CLASS_NAMES, TRTDetector

# Colors for each class (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 255),      # red
    1: (0, 255, 255),     # yellow
    2: (0, 255, 0),       # green
    3: (0, 0, 180),       # redLeft
    4: (0, 180, 0),       # greenLeft
    5: (128, 0, 255),     # redRight
    6: (128, 255, 0),     # greenRight
}


def draw_detections(frame: np.ndarray, detections: list, fps: float | None = None) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        color = CLASS_COLORS.get(det.class_id, (255, 255, 255))
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return frame


def run_video(source: str | int, detector: TRTDetector, show: bool = True, save: str | None = None, output_json: bool = False):
    """Run detection on video source."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save, fourcc, input_fps, (w, h))

    frame_count = 0
    total_time = 0.0
    fps_display = 0.0

    print(f"Input: {source} ({w}x{h} @ {input_fps:.1f}fps)")
    print(f"Model: conf_thresh={detector.conf_thresh}, imgsz={detector.imgsz}")
    print("Press 'q' to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            dt = time.perf_counter() - t0

            total_time += dt
            frame_count += 1
            fps_display = frame_count / total_time

            if output_json:
                record = {
                    "frame": frame_count,
                    "timestamp_ms": round(total_time * 1000, 1),
                    "inference_ms": round(dt * 1000, 1),
                    "detections": [d.to_dict() for d in detections],
                }
                print(json.dumps(record), flush=True)
            elif frame_count % 30 == 0:
                print(
                    f"Frame {frame_count}: {len(detections)} detections, "
                    f"{dt*1000:.1f}ms ({fps_display:.1f} FPS avg)"
                )

            vis = draw_detections(frame.copy(), detections, fps_display) if show or writer else frame

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
        if show:
            cv2.destroyAllWindows()

    avg_ms = (total_time / frame_count * 1000) if frame_count else 0
    if not output_json:
        print(f"\nSummary: {frame_count} frames, avg {avg_ms:.1f}ms/frame ({fps_display:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Traffic Light Detection Demo")
    parser.add_argument("--source", required=True, help="Video file path or camera index (0, 1, ...)")
    parser.add_argument("--model", required=True, help="Model path (.engine or .onnx)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size (default: 1280 for 8MP cameras)")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    parser.add_argument("--save", type=str, default=None, help="Save output video to path")
    parser.add_argument("--json", action="store_true", help="Output per-frame JSON to stdout")
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

    run_video(source, detector, show=not args.no_show, save=args.save, output_json=args.json)


if __name__ == "__main__":
    main()
