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
