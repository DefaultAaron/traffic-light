"""Tracker + EMA class-voting on top of per-frame TRTDetector.

See `docs/integration/tracker_voting_guide.md` for design.
"""

from inference.tracker.byte_tracker import BYTETracker, STrack
from inference.tracker.smoother import TrackedDetection, TrackSmoother

__all__ = ["BYTETracker", "STrack", "TrackedDetection", "TrackSmoother"]
