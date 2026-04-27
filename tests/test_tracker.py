"""Fixture-driven tests for TrackSmoother.

Each JSON under `tests/fixtures/tracker/` describes an input frame sequence
plus expected post-conditions. The same fixtures will drive the C++ port's
tests so any semantic drift is caught immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference.tracker import TrackedDetection, TrackSmoother
from inference.tracker.basetrack import BaseTrack
from inference.trt_pipeline import Detection

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tracker"


def _load_fixtures():
    return sorted(FIXTURE_DIR.glob("*.json"))


@pytest.fixture(autouse=True)
def _reset_track_id_counter():
    """BaseTrack._count is a class-level global. Reset between tests so each
    case starts from id=1."""
    BaseTrack.reset_id()
    yield
    BaseTrack.reset_id()


def _build_detections(frame_entries: list[dict]) -> list[Detection]:
    return [Detection(**d) for d in frame_entries]


def _run_fixture(fixture_path: Path):
    data = json.loads(fixture_path.read_text())
    cfg = data["config"]
    ts = TrackSmoother(num_classes=data["num_classes"], **cfg)

    all_outputs: list[list[TrackedDetection]] = []
    for i, frame_dets in enumerate(data["frames"]):
        dets = _build_detections(frame_dets)
        out = ts.update(dets, i)
        all_outputs.append(out)
    return data, all_outputs


@pytest.mark.parametrize("fixture_path", _load_fixtures(), ids=lambda p: p.stem)
def test_fixture_frame_counts(fixture_path):
    """Per-frame output count matches the fixture's per_frame_track_count."""
    data, outputs = _run_fixture(fixture_path)
    expected_counts = data["expected"].get("per_frame_track_count")
    if expected_counts is None:
        pytest.skip("fixture has no per_frame_track_count expectation")
    actual_counts = [len(o) for o in outputs]
    assert actual_counts == expected_counts, (
        f"{fixture_path.name}: per-frame counts mismatch. "
        f"expected {expected_counts}, got {actual_counts}"
    )


@pytest.mark.parametrize("fixture_path", _load_fixtures(), ids=lambda p: p.stem)
def test_fixture_unique_tracks(fixture_path):
    data, outputs = _run_fixture(fixture_path)
    expected = data["expected"].get("unique_tracks")
    if expected is None:
        pytest.skip("fixture has no unique_tracks expectation")
    unique = {d.tracking_id for frame_out in outputs for d in frame_out}
    assert len(unique) == expected, (
        f"{fixture_path.name}: expected {expected} unique tracks, got {len(unique)}: {unique}"
    )


@pytest.mark.parametrize("fixture_path", _load_fixtures(), ids=lambda p: p.stem)
def test_fixture_final_smoothed_class(fixture_path):
    data, outputs = _run_fixture(fixture_path)
    expected = data["expected"].get("final_smoothed_class_id")
    if expected is None:
        pytest.skip("fixture has no final_smoothed_class_id expectation")
    # Use the last non-empty frame's first track as the reference.
    last_frame = next((o for o in reversed(outputs) if o), None)
    assert last_frame is not None, f"{fixture_path.name}: no output frames"
    actual = last_frame[0].class_id
    assert actual == expected, (
        f"{fixture_path.name}: final smoothed class_id {actual} != expected {expected}"
    )


@pytest.mark.parametrize("fixture_path", _load_fixtures(), ids=lambda p: p.stem)
def test_fixture_smoothed_never_in(fixture_path):
    data, outputs = _run_fixture(fixture_path)
    forbidden = data["expected"].get("smoothed_never_in")
    if forbidden is None:
        pytest.skip("fixture has no smoothed_never_in expectation")
    forbidden_set = set(forbidden)
    for i, frame_out in enumerate(outputs):
        for d in frame_out:
            assert d.class_id not in forbidden_set, (
                f"{fixture_path.name}: frame {i} tracking_id={d.tracking_id} "
                f"had smoothed class_id={d.class_id} in forbidden set {forbidden}"
            )


def test_gap_survival_same_tracking_id():
    """Gap-survival fixture: the id from before the gap must resume after it."""
    path = FIXTURE_DIR / "gap_survival.json"
    data, outputs = _run_fixture(path)
    before_gap = outputs[2][0].tracking_id
    after_gap = outputs[6][0].tracking_id
    assert before_gap == after_gap, (
        f"tracking_id changed across gap: before={before_gap} after={after_gap}"
    )


def test_overlapping_classes_no_pollution():
    """Two overlapping detections with different classes must not contaminate
    each other's EMA class state. Regression for inference_code_review_2026-04-27
    §Medium: with IoU-after-the-fact class recovery, both tracks could pick the
    same detection and pollute the other's class. After carrying the matched
    detection index out of ByteTrack, each track recovers its own detection."""
    path = FIXTURE_DIR / "overlapping_classes.json"
    data, outputs = _run_fixture(path)
    confirmed = [o for o in outputs if len(o) == 2]
    assert confirmed, "expected at least one confirmed two-track frame"
    expected_left = data["expected"]["left_track_class_id"]
    expected_right = data["expected"]["right_track_class_id"]
    for frame_out in confirmed:
        sorted_by_x = sorted(frame_out, key=lambda d: d.x1)
        left, right = sorted_by_x[0], sorted_by_x[1]
        assert left.class_id == expected_left, (
            f"left track class polluted: got {left.class_id} (expected {expected_left}); "
            f"raw_class={left.raw_class_id}, probs={left.class_probs}"
        )
        assert right.class_id == expected_right, (
            f"right track class polluted: got {right.class_id} (expected {expected_right}); "
            f"raw_class={right.raw_class_id}, probs={right.class_probs}"
        )


def test_two_box_ids_do_not_swap():
    """Two-box fixture: the static and moving boxes keep distinct, stable ids."""
    path = FIXTURE_DIR / "two_box_stability.json"
    data, outputs = _run_fixture(path)
    # First frame with two outputs after confirmation
    confirmed = [o for o in outputs if len(o) == 2]
    assert confirmed, "expected at least one frame with both tracks confirmed"
    # Static box is the one with x1≈100; moving box starts at x1≈400+
    static_ids = set()
    moving_ids = set()
    for frame_out in confirmed:
        for d in frame_out:
            if d.x1 < 200:
                static_ids.add(d.tracking_id)
            else:
                moving_ids.add(d.tracking_id)
    assert len(static_ids) == 1 and len(moving_ids) == 1, (
        f"IDs swapped or drifted: static={static_ids}, moving={moving_ids}"
    )
    assert static_ids.isdisjoint(moving_ids), (
        f"static and moving boxes share an id: {static_ids & moving_ids}"
    )


def test_reset_clears_state():
    ts = TrackSmoother(num_classes=7, alpha=0.3, min_hits=3)
    for i in range(5):
        ts.update([Detection(0, 0.9, 100, 100, 120, 150)], i)
    out_before = ts.update([Detection(0, 0.9, 100, 100, 120, 150)], 5)
    assert out_before, "expected confirmed track before reset"
    confirmed_id = out_before[0].tracking_id

    ts.reset()
    # After reset, id counter starts over at 1; track must re-confirm.
    out_reset_confirm = None
    for i in range(6, 10):
        out_reset_confirm = ts.update([Detection(0, 0.9, 100, 100, 120, 150)], i)
    assert out_reset_confirm, "expected re-confirmation after reset"
    assert out_reset_confirm[0].tracking_id == 1, (
        f"expected id to reset to 1; got {out_reset_confirm[0].tracking_id} "
        f"(pre-reset id was {confirmed_id})"
    )


def test_num_classes_validation():
    with pytest.raises(ValueError):
        TrackSmoother(num_classes=0)
    with pytest.raises(ValueError):
        TrackSmoother(num_classes=7, alpha=0)
    with pytest.raises(ValueError):
        TrackSmoother(num_classes=7, alpha=1.1)
    with pytest.raises(ValueError):
        TrackSmoother(num_classes=7, track_thresh=0.6, high_thresh=0.5)


def test_tracked_to_ros_msg_populates_tracking_id(monkeypatch):
    """Regression for inference_code_review_2026-04-27 §High: TrackedDetection
    must override to_ros_msg() so downstream ROS consumers see tracking_id."""
    import sys
    import types

    # Stub vision_msgs.msg with the minimum surface to_ros_msg() touches.
    class _Vec:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0

    class _Pose:
        def __init__(self):
            self.position = _Vec()

    class _Center:
        def __init__(self):
            self.position = _Vec()

    class BoundingBox2D:
        def __init__(self):
            self.center = _Center()
            self.size_x = 0.0
            self.size_y = 0.0

    class _Hyp:
        def __init__(self):
            self.class_id = ""
            self.score = 0.0

    class ObjectHypothesisWithPose:
        def __init__(self):
            self.hypothesis = _Hyp()

    class Detection2D:
        def __init__(self):
            self.bbox = None
            self.results = []
            self.tracking_id = ""

    fake_msg = types.ModuleType("vision_msgs.msg")
    fake_msg.BoundingBox2D = BoundingBox2D
    fake_msg.Detection2D = Detection2D
    fake_msg.ObjectHypothesisWithPose = ObjectHypothesisWithPose
    fake_pkg = types.ModuleType("vision_msgs")
    fake_pkg.msg = fake_msg
    monkeypatch.setitem(sys.modules, "vision_msgs", fake_pkg)
    monkeypatch.setitem(sys.modules, "vision_msgs.msg", fake_msg)

    tracked = TrackedDetection(
        class_id=0,
        confidence=0.9,
        x1=10.0, y1=20.0, x2=30.0, y2=40.0,
        tracking_id=42,
        age=5,
        hits=4,
        raw_class_id=0,
        raw_confidence=0.88,
        class_probs=[0.9, 0.1, 0, 0, 0, 0, 0],
    )
    msg = tracked.to_ros_msg()
    assert msg.tracking_id == "42"
    assert msg.results[0].hypothesis.class_id == "red"
    assert msg.results[0].hypothesis.score == pytest.approx(0.9)
