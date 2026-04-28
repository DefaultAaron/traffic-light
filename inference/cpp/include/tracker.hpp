#pragma once

// TrackSmoother — C++ port of inference/tracker/smoother.py.
//
// Wraps a ByteTrack-style association over detections coming out of the TRT
// pipeline and maintains a per-track EMA class probability vector. Public API
// is intentionally minimal and matches the Python side 1:1 (same fixtures
// drive both test suites — see docs/integration/tracker.md §7).
//
// Thread safety: TrackSmoother is NOT thread-safe. It owns the Kalman state
// of every live track. Construct and call update() on the same thread.
//
// ID scope note: track IDs come from a static counter shared across all
// TrackSmoother instances in the same process (matching the Python version).
// For multi-camera deployment keep each camera in its own process, or
// namespace IDs at the publisher layer.

#include <memory>
#include <vector>

#include "detection.hpp"

namespace tl {

struct TrackedDetection : Detection {
    int tracking_id = -1;
    int age = 0;                    // frames since first seen
    int hits = 0;                   // frames where a measurement was observed
    int raw_class_id = -1;          // per-frame argmax before smoothing
    float raw_confidence = 0.f;
    std::vector<float> class_probs; // size == num_classes when emitted
};

struct TrackerConfig {
    int   num_classes   = 7;
    float alpha         = 0.3f;     // EMA coefficient: lower = smoother
    float track_thresh  = 0.25f;    // low cutoff; detections below this are dropped
    float high_thresh   = 0.5f;     // split for two-pass association
    float match_thresh  = 0.8f;     // IoU cost ceiling for first-pass matching
    int   track_buffer  = 30;       // frames to keep lost tracks alive
    int   min_hits      = 3;        // measurements required before output
    int   frame_rate    = 30;       // used only to scale track_buffer to time
};

class TrackSmoother {
public:
    explicit TrackSmoother(const TrackerConfig& cfg = {});
    ~TrackSmoother();

    TrackSmoother(const TrackSmoother&) = delete;
    TrackSmoother& operator=(const TrackSmoother&) = delete;
    TrackSmoother(TrackSmoother&&) noexcept;
    TrackSmoother& operator=(TrackSmoother&&) noexcept;

    // Run one tracker step. `frame_idx` is the caller's frame counter and is
    // only used for `age` bookkeeping. Returns only tracks that have reached
    // min_hits — one-shot FPs are suppressed inside.
    std::vector<TrackedDetection> update(const std::vector<Detection>& detections,
                                         int frame_idx);

    // Clear all tracks and reset the global id counter. Call on source
    // switch or frame gap longer than track_buffer.
    void reset();

    const TrackerConfig& config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tl
