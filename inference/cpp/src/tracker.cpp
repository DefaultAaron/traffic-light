// TrackSmoother — C++ port of inference/tracker/*.py.
//
// Keeps everything internal in an anonymous namespace so the library exposes
// only the `tl::TrackSmoother` API declared in tracker.hpp. Depends only on
// OpenCV (cv::Mat for Kalman linear algebra) and the STL.
//
// Semantic parity with the Python version is enforced by the shared JSON
// fixtures under tests/fixtures/tracker/. See docs/integration/
// tracker_voting_guide.md §7 for the test strategy.

#include "tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace tl {

namespace {

// =====================================================================
// Global track id counter — matches Python's BaseTrack._count singleton.
// Parallels inference/tracker/basetrack.py intentionally so a single
// reset() clears both the tracker state and the id space.
// =====================================================================
int& globalTrackIdCounter() {
    static int counter = 0;
    return counter;
}
int nextTrackId() { return ++globalTrackIdCounter(); }
void resetTrackIdCounter() { globalTrackIdCounter() = 0; }

enum class TrackState { New, Tracked, Lost, Removed };

// =====================================================================
// KalmanFilter — 8-dim state (x, y, a, h, vx, vy, va, vh), constant
// velocity. Port of inference/tracker/kalman_filter.py.
// =====================================================================
class KalmanFilter {
public:
    KalmanFilter() {
        motion_mat_ = cv::Mat::eye(8, 8, CV_64F);
        for (int i = 0; i < 4; ++i) motion_mat_.at<double>(i, 4 + i) = 1.0;
        update_mat_ = cv::Mat::eye(4, 8, CV_64F);
    }

    // Initialize state from a 4-dim measurement (x, y, a, h).
    void initiate(const cv::Mat& measurement, cv::Mat& mean, cv::Mat& cov) const {
        mean = cv::Mat::zeros(8, 1, CV_64F);
        measurement.copyTo(mean.rowRange(0, 4));

        const double h = measurement.at<double>(3);
        const std::array<double, 8> std_ = {
            2 * kStdWeightPosition_ * h,
            2 * kStdWeightPosition_ * h,
            1e-2,
            2 * kStdWeightPosition_ * h,
            10 * kStdWeightVelocity_ * h,
            10 * kStdWeightVelocity_ * h,
            1e-5,
            10 * kStdWeightVelocity_ * h,
        };
        cov = cv::Mat::zeros(8, 8, CV_64F);
        for (int i = 0; i < 8; ++i) cov.at<double>(i, i) = std_[i] * std_[i];
    }

    void predict(cv::Mat& mean, cv::Mat& cov) const {
        const double h = mean.at<double>(3);
        const std::array<double, 8> std_ = {
            kStdWeightPosition_ * h,
            kStdWeightPosition_ * h,
            1e-2,
            kStdWeightPosition_ * h,
            kStdWeightVelocity_ * h,
            kStdWeightVelocity_ * h,
            1e-5,
            kStdWeightVelocity_ * h,
        };
        cv::Mat motion_cov = cv::Mat::zeros(8, 8, CV_64F);
        for (int i = 0; i < 8; ++i) motion_cov.at<double>(i, i) = std_[i] * std_[i];

        mean = motion_mat_ * mean;
        cov = motion_mat_ * cov * motion_mat_.t() + motion_cov;
    }

    // Project state into measurement space; adds innovation covariance.
    void project(const cv::Mat& mean, const cv::Mat& cov,
                 cv::Mat& proj_mean, cv::Mat& proj_cov) const {
        const double h = mean.at<double>(3);
        const std::array<double, 4> std_ = {
            kStdWeightPosition_ * h,
            kStdWeightPosition_ * h,
            1e-1,
            kStdWeightPosition_ * h,
        };
        cv::Mat innovation_cov = cv::Mat::zeros(4, 4, CV_64F);
        for (int i = 0; i < 4; ++i) innovation_cov.at<double>(i, i) = std_[i] * std_[i];

        proj_mean = update_mat_ * mean;
        proj_cov = update_mat_ * cov * update_mat_.t() + innovation_cov;
    }

    void update(cv::Mat& mean, cv::Mat& cov, const cv::Mat& measurement) const {
        cv::Mat proj_mean, proj_cov;
        project(mean, cov, proj_mean, proj_cov);

        // Kalman gain K = cov * H^T * (H * cov * H^T + R)^-1
        cv::Mat K;
        cv::solve(proj_cov.t(), (cov * update_mat_.t()).t(), K, cv::DECOMP_CHOLESKY);
        K = K.t();
        cv::Mat innovation = measurement - proj_mean;

        mean = mean + K * innovation;
        cov = cov - K * proj_cov * K.t();
    }

private:
    cv::Mat motion_mat_;  // 8x8
    cv::Mat update_mat_;  // 4x8
    static constexpr double kStdWeightPosition_ = 1.0 / 20.0;
    static constexpr double kStdWeightVelocity_ = 1.0 / 160.0;
};

// =====================================================================
// STrack — single tracklet. Mirrors inference/tracker/byte_tracker.py::STrack.
// =====================================================================
struct STrack {
    // ByteTrack per-tracklet state
    int track_id = 0;
    int start_frame = 0;
    int frame_id = 0;
    int tracklet_len = 0;
    float score = 0.f;
    TrackState state = TrackState::New;
    bool is_activated = false;

    // Kalman state: mean is 8x1, cov is 8x8 (CV_64F).
    cv::Mat mean;
    cv::Mat cov;

    // Initial box as tlwh (top-left + width/height).
    std::array<float, 4> init_tlwh{};

    // Index into the original `dets` array passed to ByteTracker::update();
    // propagated through update()/reActivate() so the smoother can recover
    // the exact matched detection (class, confidence) instead of doing an
    // IoU lookup after the fact, which can collide on overlapping boxes.
    int source_det_idx = -1;

    STrack(const std::array<float, 4>& tlwh, float s, int src_idx = -1)
        : score(s), init_tlwh(tlwh), source_det_idx(src_idx) {}

    static std::array<float, 4> tlbrToTlwh(float x1, float y1, float x2, float y2) {
        return {x1, y1, x2 - x1, y2 - y1};
    }

    // (x, y, aspect, height) = (cx, cy, w/h, h)
    static void tlwhToXyah(const std::array<float, 4>& tlwh, cv::Mat& out) {
        out = cv::Mat(4, 1, CV_64F);
        out.at<double>(0) = tlwh[0] + tlwh[2] / 2.0;
        out.at<double>(1) = tlwh[1] + tlwh[3] / 2.0;
        out.at<double>(2) = (tlwh[3] > 0.0) ? (double(tlwh[2]) / tlwh[3]) : 0.0;
        out.at<double>(3) = tlwh[3];
    }

    std::array<float, 4> tlwh() const {
        if (mean.empty()) return init_tlwh;
        // mean[:4] = (cx, cy, a, h) → tlwh = (cx - a*h/2, cy - h/2, a*h, h)
        double cx = mean.at<double>(0);
        double cy = mean.at<double>(1);
        double a = mean.at<double>(2);
        double h = mean.at<double>(3);
        double w = a * h;
        return {float(cx - w / 2.0), float(cy - h / 2.0), float(w), float(h)};
    }

    std::array<float, 4> tlbr() const {
        auto t = tlwh();
        return {t[0], t[1], t[0] + t[2], t[1] + t[3]};
    }

    void activate(const KalmanFilter& kf, int fid) {
        track_id = nextTrackId();
        cv::Mat xyah;
        tlwhToXyah(init_tlwh, xyah);
        kf.initiate(xyah, mean, cov);
        tracklet_len = 0;
        state = TrackState::Tracked;
        is_activated = (fid == 1);  // upstream quirk: only frame 1 auto-activates
        frame_id = fid;
        start_frame = fid;
    }

    void reActivate(const KalmanFilter& kf, const STrack& newTrack, int fid,
                    bool assignNewId) {
        cv::Mat xyah;
        tlwhToXyah(newTrack.init_tlwh, xyah);
        kf.update(mean, cov, xyah);
        tracklet_len = 0;
        state = TrackState::Tracked;
        is_activated = true;
        frame_id = fid;
        if (assignNewId) track_id = nextTrackId();
        score = newTrack.score;
        source_det_idx = newTrack.source_det_idx;
    }

    void update(const KalmanFilter& kf, const STrack& newTrack, int fid) {
        frame_id = fid;
        ++tracklet_len;
        cv::Mat xyah;
        tlwhToXyah(newTrack.init_tlwh, xyah);
        kf.update(mean, cov, xyah);
        state = TrackState::Tracked;
        is_activated = true;
        score = newTrack.score;
        source_det_idx = newTrack.source_det_idx;
    }

    void markLost()    { state = TrackState::Lost; }
    void markRemoved() { state = TrackState::Removed; }

    int endFrame() const { return frame_id; }
};

using STrackPtr = std::shared_ptr<STrack>;

// =====================================================================
// IoU and pairwise distance.
// =====================================================================
double iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    double ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
    double bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];
    double aw = std::max(0.0, ax2 - ax1);
    double ah = std::max(0.0, ay2 - ay1);
    double bw = std::max(0.0, bx2 - bx1);
    double bh = std::max(0.0, by2 - by1);
    double areaA = aw * ah;
    double areaB = bw * bh;

    double ix1 = std::max(ax1, bx1);
    double iy1 = std::max(ay1, by1);
    double ix2 = std::min(ax2, bx2);
    double iy2 = std::min(ay2, by2);
    double iw = std::max(0.0, ix2 - ix1);
    double ih = std::max(0.0, iy2 - iy1);
    double inter = iw * ih;
    double uni = areaA + areaB - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

std::vector<std::vector<double>> iouDistance(const std::vector<STrackPtr>& aTracks,
                                             const std::vector<STrackPtr>& bTracks) {
    const size_t n = aTracks.size();
    const size_t m = bTracks.size();
    std::vector<std::vector<double>> d(n, std::vector<double>(m, 1.0));
    for (size_t i = 0; i < n; ++i) {
        auto at = aTracks[i]->tlbr();
        for (size_t j = 0; j < m; ++j) {
            d[i][j] = 1.0 - iou(at, bTracks[j]->tlbr());
        }
    }
    return d;
}

void fuseScore(std::vector<std::vector<double>>& cost,
               const std::vector<STrackPtr>& detections) {
    if (cost.empty()) return;
    const size_t n = cost.size();
    const size_t m = cost[0].size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double iouSim = 1.0 - cost[i][j];
            double fused = iouSim * detections[j]->score;
            cost[i][j] = 1.0 - fused;
        }
    }
}

// =====================================================================
// Hungarian assignment for rectangular cost matrices.
//
// Implements the classic O(n^2 * m) Jonker-Volgenant-Castanon variant
// (aka e-maxx Hungarian). Entries above `thresh` are masked with a large
// pseudo-cost so the solver avoids them; any pair whose real cost still
// exceeds `thresh` is dropped post-hoc. Matches the Python path exactly.
// =====================================================================
struct Assignment {
    std::vector<std::pair<int, int>> matches;  // (row, col) pairs, cost <= thresh
    std::vector<int> unmatched_a;
    std::vector<int> unmatched_b;
};

Assignment linearAssignment(const std::vector<std::vector<double>>& cost,
                            double thresh, int n, int m) {
    // Dimensions are passed explicitly because `vector<vector<double>>` can't
    // represent "0 rows but M columns" — when the caller has zero tracks but
    // nonzero detections, `cost` is an empty outer vector and M would be
    // lost. scipy's linear_sum_assignment takes a 2-D numpy array that
    // carries shape=(0, M), so it preserves M via `cost_matrix.shape[1]`;
    // we must take the same information through the parameter list.
    Assignment out;
    if (n == 0 || m == 0) {
        out.unmatched_a.reserve(n);
        for (int i = 0; i < n; ++i) out.unmatched_a.push_back(i);
        out.unmatched_b.reserve(m);
        for (int j = 0; j < m; ++j) out.unmatched_b.push_back(j);
        return out;
    }

    const double kMasked = 1e9;

    // Pad to square with 0-cost dummy rows/cols; this lets us reuse the
    // square-only Hungarian template. Dummy pairs are filtered post-hoc.
    const int k = std::max(n, m);
    std::vector<std::vector<double>> a(k + 1, std::vector<double>(k + 1, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double c = cost[i][j];
            a[i + 1][j + 1] = (c > thresh) ? kMasked : c;
        }
    }

    std::vector<double> u(k + 1, 0.0), v(k + 1, 0.0);
    std::vector<int> p(k + 1, 0), way(k + 1, 0);
    const double kInf = std::numeric_limits<double>::infinity();

    for (int i = 1; i <= k; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(k + 1, kInf);
        std::vector<char> used(k + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = kInf;
            int j1 = 0;
            for (int j = 1; j <= k; ++j) {
                if (!used[j]) {
                    double cur = a[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= k; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // Invert: assignedCol[row] = col, where p[col] = row.
    std::vector<int> rowToCol(k, -1);
    for (int j = 1; j <= k; ++j) {
        if (p[j] > 0 && p[j] <= k) rowToCol[p[j] - 1] = j - 1;
    }

    std::unordered_set<int> matchedRows, matchedCols;
    for (int i = 0; i < n; ++i) {
        int j = rowToCol[i];
        if (j < 0 || j >= m) continue;
        if (cost[i][j] > thresh) continue;
        out.matches.emplace_back(i, j);
        matchedRows.insert(i);
        matchedCols.insert(j);
    }
    for (int i = 0; i < n; ++i) {
        if (!matchedRows.count(i)) out.unmatched_a.push_back(i);
    }
    for (int j = 0; j < m; ++j) {
        if (!matchedCols.count(j)) out.unmatched_b.push_back(j);
    }
    return out;
}

// =====================================================================
// Strack list utilities.
// =====================================================================
std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr>& a,
                                    const std::vector<STrackPtr>& b) {
    std::unordered_set<int> seen;
    std::vector<STrackPtr> res;
    res.reserve(a.size() + b.size());
    for (const auto& t : a) {
        res.push_back(t);
        seen.insert(t->track_id);
    }
    for (const auto& t : b) {
        if (!seen.count(t->track_id)) {
            res.push_back(t);
            seen.insert(t->track_id);
        }
    }
    return res;
}

std::vector<STrackPtr> subStracks(const std::vector<STrackPtr>& a,
                                  const std::vector<STrackPtr>& b) {
    std::unordered_set<int> drop;
    for (const auto& t : b) drop.insert(t->track_id);
    std::vector<STrackPtr> res;
    res.reserve(a.size());
    for (const auto& t : a) {
        if (!drop.count(t->track_id)) res.push_back(t);
    }
    return res;
}

void removeDuplicateStracks(std::vector<STrackPtr>& a, std::vector<STrackPtr>& b) {
    if (a.empty() || b.empty()) return;
    auto d = iouDistance(a, b);
    std::unordered_set<int> dupa, dupb;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            if (d[i][j] < 0.15) {
                int timeA = a[i]->frame_id - a[i]->start_frame;
                int timeB = b[j]->frame_id - b[j]->start_frame;
                if (timeA > timeB) dupb.insert(int(j));
                else               dupa.insert(int(i));
            }
        }
    }
    std::vector<STrackPtr> na, nb;
    for (size_t i = 0; i < a.size(); ++i) if (!dupa.count(int(i))) na.push_back(a[i]);
    for (size_t j = 0; j < b.size(); ++j) if (!dupb.count(int(j))) nb.push_back(b[j]);
    a = std::move(na);
    b = std::move(nb);
}

// =====================================================================
// BYTETracker — mirrors inference/tracker/byte_tracker.py::BYTETracker.
// =====================================================================
class ByteTracker {
public:
    ByteTracker(float trackThresh, int trackBuffer, float matchThresh,
                bool mot20, int frameRate)
        : trackThresh_(trackThresh),
          matchThresh_(matchThresh),
          mot20_(mot20),
          detThresh_(trackThresh + 0.1f),
          maxTimeLost_(int(double(frameRate) / 30.0 * trackBuffer)) {}

    void reset() {
        trackedStracks_.clear();
        lostStracks_.clear();
        removedStracks_.clear();
        frameId_ = 0;
        resetTrackIdCounter();
    }

    int frameId() const { return frameId_; }
    const std::vector<STrackPtr>& tracked() const { return trackedStracks_; }
    const std::vector<STrackPtr>& lost()    const { return lostStracks_; }

    // dets: each row is (x1, y1, x2, y2, score) in source-image pixels.
    std::vector<STrackPtr> update(const std::vector<std::array<float, 5>>& dets) {
        ++frameId_;
        std::vector<STrackPtr> activated, refind, nowLost, nowRemoved;

        // Split by score. Python uses strict inequalities on both sides
        // (`scores > track_thresh` for high, `scores > 0.1 AND scores <
        // track_thresh` for second), which drops `s == track_thresh`
        // exactly. We match that precisely rather than the more natural
        // `else if (s > 0.1)`, which would include the boundary in
        // the second bucket and diverge on quantized/rounded scores.
        std::vector<STrackPtr> detsHigh, detsSecond;
        // Carry original-array indices on each STrack so the smoother can
        // recover the exact matched detection without IoU-after-the-fact.
        for (size_t i = 0; i < dets.size(); ++i) {
            const auto& r = dets[i];
            float s = r[4];
            if (s > trackThresh_) {
                detsHigh.push_back(std::make_shared<STrack>(
                    STrack::tlbrToTlwh(r[0], r[1], r[2], r[3]), s, int(i)));
            } else if (s > 0.1f && s < trackThresh_) {
                detsSecond.push_back(std::make_shared<STrack>(
                    STrack::tlbrToTlwh(r[0], r[1], r[2], r[3]), s, int(i)));
            }
        }

        // Separate confirmed vs unconfirmed among currently tracked.
        std::vector<STrackPtr> unconfirmed, tracked;
        for (const auto& t : trackedStracks_) {
            if (!t->is_activated) unconfirmed.push_back(t);
            else                  tracked.push_back(t);
        }

        // ----- First association: high-score detections ---------------
        auto pool = jointStracks(tracked, lostStracks_);
        // Parity with STrack.multi_predict: for Lost tracks, zero the
        // height velocity (mean[7]) before predict so box size doesn't
        // drift during a multi-frame gap. Position velocities are kept —
        // objects can still move during gaps.
        for (const auto& t : pool) {
            if (t->state != TrackState::Tracked && !t->mean.empty()) {
                t->mean.at<double>(7) = 0.0;
            }
            kf_.predict(t->mean, t->cov);
        }

        auto dists = iouDistance(pool, detsHigh);
        if (!mot20_) fuseScore(dists, detsHigh);
        auto m1 = linearAssignment(dists, matchThresh_,
                                   int(pool.size()), int(detsHigh.size()));

        for (const auto& [iTrack, iDet] : m1.matches) {
            auto& t = pool[iTrack];
            auto& d = detsHigh[iDet];
            if (t->state == TrackState::Tracked) {
                t->update(kf_, *d, frameId_);
                activated.push_back(t);
            } else {
                t->reActivate(kf_, *d, frameId_, /*assignNewId=*/false);
                refind.push_back(t);
            }
        }

        // ----- Second association: low-score detections ---------------
        std::vector<STrackPtr> rTracked;
        for (int i : m1.unmatched_a) {
            if (pool[i]->state == TrackState::Tracked) rTracked.push_back(pool[i]);
        }
        auto dists2 = iouDistance(rTracked, detsSecond);
        auto m2 = linearAssignment(dists2, 0.5,
                                   int(rTracked.size()), int(detsSecond.size()));
        for (const auto& [iTrack, iDet] : m2.matches) {
            auto& t = rTracked[iTrack];
            auto& d = detsSecond[iDet];
            if (t->state == TrackState::Tracked) {
                t->update(kf_, *d, frameId_);
                activated.push_back(t);
            } else {
                t->reActivate(kf_, *d, frameId_, /*assignNewId=*/false);
                refind.push_back(t);
            }
        }
        for (int i : m2.unmatched_a) {
            auto& t = rTracked[i];
            if (t->state != TrackState::Lost) {
                t->markLost();
                nowLost.push_back(t);
            }
        }

        // ----- Unconfirmed pool: match leftover high-score dets --------
        std::vector<STrackPtr> remainingDetsHigh;
        for (int i : m1.unmatched_b) remainingDetsHigh.push_back(detsHigh[i]);

        auto dists3 = iouDistance(unconfirmed, remainingDetsHigh);
        if (!mot20_) fuseScore(dists3, remainingDetsHigh);
        auto m3 = linearAssignment(dists3, 0.7,
                                   int(unconfirmed.size()),
                                   int(remainingDetsHigh.size()));
        for (const auto& [iTrack, iDet] : m3.matches) {
            unconfirmed[iTrack]->update(kf_, *remainingDetsHigh[iDet], frameId_);
            activated.push_back(unconfirmed[iTrack]);
        }
        for (int i : m3.unmatched_a) {
            unconfirmed[i]->markRemoved();
            nowRemoved.push_back(unconfirmed[i]);
        }

        // ----- Init new tracks from unmatched high-score detections ---
        for (int i : m3.unmatched_b) {
            auto& d = remainingDetsHigh[i];
            if (d->score < detThresh_) continue;
            d->activate(kf_, frameId_);
            activated.push_back(d);
        }

        // ----- Age-out lost tracks ------------------------------------
        for (auto& t : lostStracks_) {
            if (frameId_ - t->endFrame() > maxTimeLost_) {
                t->markRemoved();
                nowRemoved.push_back(t);
            }
        }

        // ----- Merge and de-dup --------------------------------------
        std::vector<STrackPtr> newTracked;
        for (auto& t : trackedStracks_) {
            if (t->state == TrackState::Tracked) newTracked.push_back(t);
        }
        newTracked = jointStracks(newTracked, activated);
        newTracked = jointStracks(newTracked, refind);
        trackedStracks_ = std::move(newTracked);

        lostStracks_ = subStracks(lostStracks_, trackedStracks_);
        for (auto& t : nowLost) lostStracks_.push_back(t);
        lostStracks_ = subStracks(lostStracks_, removedStracks_);
        for (auto& t : nowRemoved) removedStracks_.push_back(t);

        removeDuplicateStracks(trackedStracks_, lostStracks_);

        std::vector<STrackPtr> out;
        for (auto& t : trackedStracks_) {
            if (t->is_activated) out.push_back(t);
        }
        return out;
    }

private:
    float trackThresh_;
    float matchThresh_;
    bool  mot20_;
    float detThresh_;
    int   maxTimeLost_;
    int   frameId_ = 0;

    KalmanFilter kf_;
    std::vector<STrackPtr> trackedStracks_;
    std::vector<STrackPtr> lostStracks_;
    std::vector<STrackPtr> removedStracks_;
};

// =====================================================================
// Per-track EMA bookkeeping shared between frames.
// =====================================================================
struct TrackState_ {
    int first_frame = 0;
    int last_seen_frame = 0;
    int hits = 0;
    std::vector<double> class_probs;  // length = num_classes
    int last_raw_class_id = 0;
};

std::vector<double> oneHot(int classId, int numClasses) {
    std::vector<double> v(numClasses, 0.0);
    if (classId >= 0 && classId < numClasses) v[classId] = 1.0;
    return v;
}

int argmax(const std::vector<double>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = int(i);
    }
    return best;
}

}  // namespace (internal)

// =====================================================================
// TrackSmoother Impl — pImpl holder mirroring smoother.py.
// =====================================================================
struct TrackSmoother::Impl {
    TrackerConfig cfg;
    std::unique_ptr<ByteTracker> tracker;
    std::unordered_map<int, TrackState_> state;

    explicit Impl(const TrackerConfig& c) : cfg(c) {
        if (cfg.num_classes <= 0) {
            throw std::invalid_argument("num_classes must be positive");
        }
        if (!(cfg.alpha > 0.0f && cfg.alpha <= 1.0f)) {
            throw std::invalid_argument("alpha must be in (0, 1]");
        }
        if (cfg.high_thresh < cfg.track_thresh) {
            throw std::invalid_argument(
                "high_thresh must be >= track_thresh");
        }
        tracker = std::make_unique<ByteTracker>(
            cfg.high_thresh, cfg.track_buffer, cfg.match_thresh,
            /*mot20=*/false, cfg.frame_rate);
    }

    std::vector<TrackedDetection> update(const std::vector<Detection>& detections,
                                         int frame_idx) {
        // Outer filter: drop detections below the low cutoff.
        std::vector<Detection> filtered;
        filtered.reserve(detections.size());
        for (const auto& d : detections) {
            if (d.confidence >= cfg.track_thresh) filtered.push_back(d);
        }
        std::vector<std::array<float, 5>> dets;
        dets.reserve(filtered.size());
        for (const auto& d : filtered) {
            dets.push_back({d.x1, d.y1, d.x2, d.y2, d.confidence});
        }

        auto active = tracker->update(dets);
        const int current = tracker->frameId();

        std::vector<TrackedDetection> output;
        output.reserve(active.size());

        for (const auto& strack : active) {
            const int tid = strack->track_id;
            auto it = state.find(tid);
            TrackState_* st = (it == state.end()) ? nullptr : &it->second;
            const bool got_measurement = (strack->frame_id == current);

            int raw_cls = 0;
            float raw_conf = 0.f;
            std::array<float, 4> box = strack->tlbr();

            if (got_measurement) {
                // Use the index ByteTrack actually associated to this track,
                // not an IoU lookup after the fact (which can collide on
                // overlapping boxes with different classes).
                int det_idx = strack->source_det_idx;
                if (det_idx < 0 || det_idx >= int(filtered.size())) det_idx = -1;
                if (det_idx >= 0) {
                    const auto& d = filtered[det_idx];
                    raw_cls = d.class_id;
                    raw_conf = d.confidence;
                    box = {d.x1, d.y1, d.x2, d.y2};
                    if (st == nullptr) {
                        TrackState_ fresh;
                        fresh.first_frame = frame_idx;
                        fresh.last_seen_frame = frame_idx;
                        fresh.hits = 1;
                        fresh.class_probs = oneHot(raw_cls, cfg.num_classes);
                        fresh.last_raw_class_id = raw_cls;
                        state.emplace(tid, std::move(fresh));
                        st = &state[tid];
                    } else {
                        ++st->hits;
                        st->last_seen_frame = frame_idx;
                        auto oh = oneHot(raw_cls, cfg.num_classes);
                        for (int i = 0; i < cfg.num_classes; ++i) {
                            st->class_probs[i] =
                                cfg.alpha * oh[i] + (1.0 - cfg.alpha) * st->class_probs[i];
                        }
                        st->last_raw_class_id = raw_cls;
                    }
                } else {
                    // Tracker matched a detection we can't recover via IoU.
                    // Treat as measurement-less: no hits increment, no
                    // class_probs update (avoids echo-smoothing toward
                    // the stale class). Matches the Python fallback.
                    if (st == nullptr) continue;
                    raw_cls = st->last_raw_class_id;
                    raw_conf = 0.f;
                }
            } else {
                if (st == nullptr) continue;
                raw_cls = st->last_raw_class_id;
                raw_conf = 0.f;
            }

            if (st->hits < cfg.min_hits) continue;

            const int smoothed_cls = argmax(st->class_probs);
            const float smoothed_conf = float(st->class_probs[smoothed_cls]);

            TrackedDetection out;
            out.class_id = smoothed_cls;
            out.confidence = smoothed_conf;
            out.x1 = box[0];
            out.y1 = box[1];
            out.x2 = box[2];
            out.y2 = box[3];
            out.tracking_id = tid;
            out.age = frame_idx - st->first_frame + 1;
            out.hits = st->hits;
            out.raw_class_id = raw_cls;
            out.raw_confidence = raw_conf;
            out.class_probs.assign(st->class_probs.begin(), st->class_probs.end());
            output.push_back(std::move(out));
        }

        // GC bookkeeping for tracks that have dropped out of the tracker.
        std::unordered_set<int> live;
        for (const auto& t : tracker->tracked()) live.insert(t->track_id);
        for (const auto& t : tracker->lost())    live.insert(t->track_id);
        for (auto it = state.begin(); it != state.end(); ) {
            if (!live.count(it->first)) it = state.erase(it);
            else                        ++it;
        }

        return output;
    }

    void reset() {
        tracker->reset();
        state.clear();
    }
};

TrackSmoother::TrackSmoother(const TrackerConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

TrackSmoother::~TrackSmoother() = default;
TrackSmoother::TrackSmoother(TrackSmoother&&) noexcept = default;
TrackSmoother& TrackSmoother::operator=(TrackSmoother&&) noexcept = default;

std::vector<TrackedDetection>
TrackSmoother::update(const std::vector<Detection>& detections, int frame_idx) {
    return impl_->update(detections, frame_idx);
}

void TrackSmoother::reset() { impl_->reset(); }

const TrackerConfig& TrackSmoother::config() const noexcept { return impl_->cfg; }

}  // namespace tl
