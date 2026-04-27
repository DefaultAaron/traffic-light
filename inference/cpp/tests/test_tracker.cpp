// Fixture-driven C++ tracker parity test.
//
// Reads JSON fixtures at tests/fixtures/tracker/*.json (shared with the
// Python suite) and asserts TrackSmoother produces the same per-frame
// counts, unique track counts, final smoothed class, and "never in"
// constraints as the Python reference. Any divergence is a parity break
// and should fail CI.
//
// We deliberately avoid GoogleTest / Catch2 to keep build-time deps low —
// nlohmann/json is fetched via FetchContent (header-only), and assertions
// are plain C++ that print context and exit nonzero.
//
// Usage (set by CMake via compile-time definition):
//   tl_tracker_tests --fixtures <dir>
//   tl_tracker_tests               # default TL_FIXTURE_DIR

#include "tracker.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int g_failures = 0;

#define TL_CHECK(cond, ctx)                                                       \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "FAIL [" << (ctx) << "] " << #cond                       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";           \
            ++g_failures;                                                         \
        }                                                                         \
    } while (0)

tl::TrackerConfig configFromJson(const json& j) {
    tl::TrackerConfig cfg;
    if (j.contains("alpha"))        cfg.alpha        = j["alpha"].get<float>();
    if (j.contains("track_thresh")) cfg.track_thresh = j["track_thresh"].get<float>();
    if (j.contains("high_thresh"))  cfg.high_thresh  = j["high_thresh"].get<float>();
    if (j.contains("match_thresh")) cfg.match_thresh = j["match_thresh"].get<float>();
    if (j.contains("track_buffer")) cfg.track_buffer = j["track_buffer"].get<int>();
    if (j.contains("min_hits"))     cfg.min_hits     = j["min_hits"].get<int>();
    if (j.contains("frame_rate"))   cfg.frame_rate   = j["frame_rate"].get<int>();
    return cfg;
}

std::vector<tl::Detection> detectionsFromJson(const json& arr) {
    std::vector<tl::Detection> out;
    out.reserve(arr.size());
    for (const auto& d : arr) {
        out.push_back(tl::Detection{
            d.at("class_id").get<int>(),
            d.at("confidence").get<float>(),
            d.at("x1").get<float>(),
            d.at("y1").get<float>(),
            d.at("x2").get<float>(),
            d.at("y2").get<float>(),
        });
    }
    return out;
}

struct FixtureRun {
    std::string name;
    json expected;
    std::vector<std::vector<tl::TrackedDetection>> outputs;
};

FixtureRun runFixture(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open fixture: " << path << "\n";
        std::exit(2);
    }
    json data = json::parse(in);

    tl::TrackerConfig cfg = configFromJson(data.value("config", json::object()));
    cfg.num_classes = data.value("num_classes", 7);

    FixtureRun run;
    run.name = data.value("name", path.stem().string());
    run.expected = data.value("expected", json::object());

    tl::TrackSmoother ts(cfg);

    const auto& frames = data.at("frames");
    run.outputs.reserve(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        auto dets = detectionsFromJson(frames[i]);
        run.outputs.push_back(ts.update(dets, static_cast<int>(i)));
    }
    return run;
}

void checkPerFrameCount(const FixtureRun& run) {
    if (!run.expected.contains("per_frame_track_count")) return;
    const auto& expected = run.expected["per_frame_track_count"];
    TL_CHECK(expected.size() == run.outputs.size(), run.name + ":count_len");
    const size_t n = std::min<size_t>(expected.size(), run.outputs.size());
    for (size_t i = 0; i < n; ++i) {
        int exp = expected[i].get<int>();
        int got = static_cast<int>(run.outputs[i].size());
        if (exp != got) {
            std::cerr << "  " << run.name << " frame " << i
                      << ": expected " << exp << " tracks, got " << got << "\n";
            ++g_failures;
        }
    }
}

void checkUniqueTracks(const FixtureRun& run) {
    if (!run.expected.contains("unique_tracks")) return;
    std::set<int> ids;
    for (const auto& frame : run.outputs) {
        for (const auto& t : frame) ids.insert(t.tracking_id);
    }
    int expected = run.expected["unique_tracks"].get<int>();
    int got = static_cast<int>(ids.size());
    if (expected != got) {
        std::cerr << "  " << run.name << " unique_tracks: expected " << expected
                  << ", got " << got << " (ids:";
        for (int id : ids) std::cerr << " " << id;
        std::cerr << ")\n";
        ++g_failures;
    }
}

void checkFinalSmoothedClass(const FixtureRun& run) {
    if (!run.expected.contains("final_smoothed_class_id")) return;
    int expected = run.expected["final_smoothed_class_id"].get<int>();
    const std::vector<tl::TrackedDetection>* last = nullptr;
    for (auto it = run.outputs.rbegin(); it != run.outputs.rend(); ++it) {
        if (!it->empty()) { last = &(*it); break; }
    }
    if (last == nullptr) {
        std::cerr << "  " << run.name << ": no output frames — cannot check final class\n";
        ++g_failures;
        return;
    }
    int got = (*last)[0].class_id;
    if (got != expected) {
        std::cerr << "  " << run.name << " final_smoothed_class_id: expected "
                  << expected << ", got " << got << "\n";
        ++g_failures;
    }
}

void checkSmoothedNeverIn(const FixtureRun& run) {
    if (!run.expected.contains("smoothed_never_in")) return;
    std::set<int> forbidden;
    for (const auto& v : run.expected["smoothed_never_in"]) {
        forbidden.insert(v.get<int>());
    }
    for (size_t i = 0; i < run.outputs.size(); ++i) {
        for (const auto& t : run.outputs[i]) {
            if (forbidden.count(t.class_id)) {
                std::cerr << "  " << run.name << " frame " << i
                          << ": track #" << t.tracking_id
                          << " smoothed class_id=" << t.class_id
                          << " is in forbidden set\n";
                ++g_failures;
            }
        }
    }
}

void checkGapSurvival(const FixtureRun& run) {
    if (run.name != "gap_survival") return;
    if (run.outputs.size() < 7) return;
    if (run.outputs[2].empty() || run.outputs[6].empty()) {
        std::cerr << "  gap_survival: missing outputs at frames 2/6\n";
        ++g_failures;
        return;
    }
    int before = run.outputs[2][0].tracking_id;
    int after = run.outputs[6][0].tracking_id;
    if (before != after) {
        std::cerr << "  gap_survival: tracking_id changed across gap "
                  << "(before=" << before << " after=" << after << ")\n";
        ++g_failures;
    }
}

void checkOverlappingClasses(const FixtureRun& run) {
    if (run.name != "overlapping_classes") return;
    if (!run.expected.contains("left_track_class_id") ||
        !run.expected.contains("right_track_class_id")) return;
    int expected_left = run.expected["left_track_class_id"].get<int>();
    int expected_right = run.expected["right_track_class_id"].get<int>();
    bool any_confirmed = false;
    for (const auto& frame : run.outputs) {
        if (frame.size() != 2) continue;
        any_confirmed = true;
        // Sort by x1 ascending → left, right.
        std::vector<tl::TrackedDetection> sorted = frame;
        std::sort(sorted.begin(), sorted.end(),
                  [](const tl::TrackedDetection& a, const tl::TrackedDetection& b) {
                      return a.x1 < b.x1;
                  });
        if (sorted[0].class_id != expected_left) {
            std::cerr << "  overlapping_classes: left track class polluted "
                      << "(got " << sorted[0].class_id
                      << " expected " << expected_left
                      << " raw=" << sorted[0].raw_class_id << ")\n";
            ++g_failures;
        }
        if (sorted[1].class_id != expected_right) {
            std::cerr << "  overlapping_classes: right track class polluted "
                      << "(got " << sorted[1].class_id
                      << " expected " << expected_right
                      << " raw=" << sorted[1].raw_class_id << ")\n";
            ++g_failures;
        }
    }
    if (!any_confirmed) {
        std::cerr << "  overlapping_classes: no confirmed two-track frames\n";
        ++g_failures;
    }
}

void checkTwoBoxStability(const FixtureRun& run) {
    if (run.name != "two_box_stability") return;
    std::set<int> static_ids, moving_ids;
    for (const auto& frame : run.outputs) {
        if (frame.size() != 2) continue;
        for (const auto& t : frame) {
            if (t.x1 < 200) static_ids.insert(t.tracking_id);
            else            moving_ids.insert(t.tracking_id);
        }
    }
    if (static_ids.size() != 1 || moving_ids.size() != 1) {
        std::cerr << "  two_box_stability: expected exactly one id per box"
                  << " (static=" << static_ids.size()
                  << " moving=" << moving_ids.size() << ")\n";
        ++g_failures;
    }
    for (int s : static_ids) {
        if (moving_ids.count(s)) {
            std::cerr << "  two_box_stability: static/moving share id " << s << "\n";
            ++g_failures;
        }
    }
}

void runResetSmokeTest() {
    tl::TrackerConfig cfg;
    tl::TrackSmoother ts(cfg);
    for (int i = 0; i < 5; ++i) {
        tl::Detection d{0, 0.9f, 100, 100, 120, 150};
        ts.update({d}, i);
    }
    auto before = ts.update({tl::Detection{0, 0.9f, 100, 100, 120, 150}}, 5);
    TL_CHECK(!before.empty(), "reset:pre");

    ts.reset();
    std::vector<tl::TrackedDetection> after;
    for (int i = 6; i < 10; ++i) {
        after = ts.update({tl::Detection{0, 0.9f, 100, 100, 120, 150}}, i);
    }
    TL_CHECK(!after.empty(), "reset:post-confirm");
    TL_CHECK(!after.empty() && after[0].tracking_id == 1, "reset:id-restarts-at-1");
}

void runConfigValidation() {
    bool threw = false;
    try {
        tl::TrackerConfig cfg; cfg.num_classes = 0;
        tl::TrackSmoother ts(cfg);
    } catch (const std::exception&) { threw = true; }
    TL_CHECK(threw, "cfg:num_classes=0");

    threw = false;
    try {
        tl::TrackerConfig cfg; cfg.alpha = 0.0f;
        tl::TrackSmoother ts(cfg);
    } catch (const std::exception&) { threw = true; }
    TL_CHECK(threw, "cfg:alpha=0");

    threw = false;
    try {
        tl::TrackerConfig cfg; cfg.alpha = 1.1f;
        tl::TrackSmoother ts(cfg);
    } catch (const std::exception&) { threw = true; }
    TL_CHECK(threw, "cfg:alpha=1.1");

    threw = false;
    try {
        tl::TrackerConfig cfg; cfg.track_thresh = 0.6f; cfg.high_thresh = 0.5f;
        tl::TrackSmoother ts(cfg);
    } catch (const std::exception&) { threw = true; }
    TL_CHECK(threw, "cfg:high<track");
}

}  // namespace

int main(int argc, char** argv) {
    fs::path fixture_dir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--fixtures" && i + 1 < argc) fixture_dir = argv[++i];
    }
#ifdef TL_FIXTURE_DIR
    if (fixture_dir.empty()) fixture_dir = TL_FIXTURE_DIR;
#endif
    if (fixture_dir.empty() || !fs::is_directory(fixture_dir)) {
        std::cerr << "fixture dir not found: '" << fixture_dir.string()
                  << "' (pass --fixtures <dir>)\n";
        return 2;
    }

    std::vector<fs::path> fixtures;
    for (const auto& entry : fs::directory_iterator(fixture_dir)) {
        if (entry.path().extension() == ".json") fixtures.push_back(entry.path());
    }
    std::sort(fixtures.begin(), fixtures.end());

    std::cout << "Running " << fixtures.size() << " fixture(s) from "
              << fixture_dir.string() << "\n";

    for (const auto& path : fixtures) {
        std::cout << "  • " << path.stem().string() << "\n";
        FixtureRun run = runFixture(path);
        checkPerFrameCount(run);
        checkUniqueTracks(run);
        checkFinalSmoothedClass(run);
        checkSmoothedNeverIn(run);
        checkGapSurvival(run);
        checkTwoBoxStability(run);
        checkOverlappingClasses(run);
    }

    std::cout << "Smoke: reset\n";
    runResetSmokeTest();
    std::cout << "Smoke: config validation\n";
    runConfigValidation();

    if (g_failures) {
        std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
        return 1;
    }
    std::cout << "\nAll fixture and smoke checks passed.\n";
    return 0;
}
