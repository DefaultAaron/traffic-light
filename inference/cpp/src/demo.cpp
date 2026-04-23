// Demo for the C++ TensorRT traffic-light detector + tracker.
//
// Usage:
//   tl_demo --source video.mp4 --model best.engine [--conf 0.25] [--imgsz 1280]
//   tl_demo --source 0 --model best.engine --no-show --save out.mp4
//
// Tracker mode (mirrors inference/demo.py --track):
//   tl_demo --source video.mp4 --model best.engine --track \
//           [--alpha 0.3] [--min-hits 3] [--high-thresh 0.5] \
//           [--track-buffer 30] [--track-json out.jsonl]

#include "tracker.hpp"
#include "trt_pipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

// BGR colors keyed to class_id (used when tracker is disabled).
const cv::Scalar kClassColors[] = {
    {0, 0, 255},    {0, 255, 255}, {0, 255, 0},   {0, 0, 180},
    {0, 180, 0},    {128, 0, 255}, {128, 255, 0},
};

// Deterministic distinct palette keyed by tracking_id (used when --track).
// Matches inference/demo.py's per-track palette style so side-by-side videos
// read the same.
const cv::Scalar kTrackColors[] = {
    {255,  64,  64}, { 64, 255,  64}, { 64,  64, 255}, {255, 255,  64},
    { 64, 255, 255}, {255,  64, 255}, {255, 160,  64}, { 64, 160, 255},
    {160,  64, 255}, {255, 255, 255}, {128, 255, 128}, {255, 128, 128},
};

cv::Scalar colorForClass(int cls) {
    constexpr int n = static_cast<int>(sizeof(kClassColors) / sizeof(kClassColors[0]));
    if (cls < 0 || cls >= n) return {255, 255, 255};
    return kClassColors[cls];
}

cv::Scalar colorForTrack(int track_id) {
    constexpr int n = static_cast<int>(sizeof(kTrackColors) / sizeof(kTrackColors[0]));
    return kTrackColors[((track_id % n) + n) % n];
}

void drawBox(cv::Mat& frame, const cv::Scalar& color, float x1, float y1, float x2, float y2,
             const std::string& label) {
    cv::Point p1(static_cast<int>(x1), static_cast<int>(y1));
    cv::Point p2(static_cast<int>(x2), static_cast<int>(y2));
    cv::rectangle(frame, p1, p2, color, 2);
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(frame, {p1.x, p1.y - ts.height - 4}, {p1.x + ts.width, p1.y}, color, cv::FILLED);
    cv::putText(frame, label, {p1.x, p1.y - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                {255, 255, 255}, 1);
}

void drawDetections(cv::Mat& frame, const std::vector<tl::Detection>& dets) {
    for (const auto& d : dets) {
        std::ostringstream label;
        label.precision(2);
        label << d.class_name() << " " << std::fixed << d.confidence;
        drawBox(frame, colorForClass(d.class_id), d.x1, d.y1, d.x2, d.y2, label.str());
    }
}

void drawTracks(cv::Mat& frame, const std::vector<tl::TrackedDetection>& tracks) {
    for (const auto& t : tracks) {
        std::ostringstream label;
        label.precision(2);
        label << "#" << t.tracking_id << " " << t.class_name() << " "
              << std::fixed << t.confidence;
        drawBox(frame, colorForTrack(t.tracking_id), t.x1, t.y1, t.x2, t.y2, label.str());
    }
}

void drawFps(cv::Mat& frame, float fps) {
    if (fps <= 0.f) return;
    std::ostringstream s;
    s.precision(1);
    s << "FPS: " << std::fixed << fps;
    cv::putText(frame, s.str(), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);
}

// Minimal JSONL writer — one object per frame, matching inference/demo.py
// --track-json output so offline diffing tools stay shared.
void writeTrackJson(std::ostream& os, int frame_idx,
                    const std::vector<tl::TrackedDetection>& tracks) {
    os << "{\"frame\":" << frame_idx << ",\"tracks\":[";
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        if (i) os << ',';
        os << "{\"tracking_id\":" << t.tracking_id
           << ",\"class_id\":" << t.class_id
           << ",\"class_name\":\"" << t.class_name() << "\""
           << ",\"confidence\":" << std::fixed << std::setprecision(6) << t.confidence
           << ",\"x1\":" << t.x1 << ",\"y1\":" << t.y1
           << ",\"x2\":" << t.x2 << ",\"y2\":" << t.y2
           << ",\"age\":" << t.age << ",\"hits\":" << t.hits
           << ",\"raw_class_id\":" << t.raw_class_id
           << ",\"raw_confidence\":" << t.raw_confidence
           << ",\"class_probs\":[";
        for (size_t k = 0; k < t.class_probs.size(); ++k) {
            if (k) os << ',';
            os << t.class_probs[k];
        }
        os << "]}";
    }
    os << "]}\n";
}

struct Args {
    std::string source;
    std::string model;
    float conf = 0.25f;
    int imgsz = 1280;
    bool show = true;
    std::string save;

    // Tracker flags — default off, turned on by --track.
    bool track = false;
    float alpha = 0.3f;
    int min_hits = 3;
    float high_thresh = 0.5f;
    float match_thresh = 0.8f;
    int track_buffer = 30;
    std::string track_json;
};

void printUsage() {
    std::cerr << "Usage: tl_demo --source <video|camera_idx> --model <path.engine> "
                 "[--conf 0.25] [--imgsz 1280] [--no-show] [--save out.mp4]\n"
                 "       [--track] [--alpha 0.3] [--min-hits 3] [--high-thresh 0.5]\n"
                 "       [--match-thresh 0.8] [--track-buffer 30] [--track-json out.jsonl]\n";
}

float parseFloat(const char* s, const char* flag) {
    try {
        size_t pos = 0;
        float v = std::stof(s, &pos);
        if (pos != std::strlen(s)) throw std::invalid_argument("trailing chars");
        return v;
    } catch (const std::exception& e) {
        std::cerr << flag << ": cannot parse '" << s << "' as float (" << e.what() << ")\n";
        std::exit(2);
    }
}

int parseInt(const char* s, const char* flag) {
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != std::strlen(s)) throw std::invalid_argument("trailing chars");
        return v;
    } catch (const std::exception& e) {
        std::cerr << flag << ": cannot parse '" << s << "' as int (" << e.what() << ")\n";
        std::exit(2);
    }
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--source")              a.source       = next("--source");
        else if (k == "--model")          a.model        = next("--model");
        else if (k == "--conf")           a.conf         = parseFloat(next("--conf"), "--conf");
        else if (k == "--imgsz")          a.imgsz        = parseInt(next("--imgsz"), "--imgsz");
        else if (k == "--save")           a.save         = next("--save");
        else if (k == "--no-show")        a.show         = false;
        else if (k == "--track")          a.track        = true;
        else if (k == "--alpha")          a.alpha        = parseFloat(next("--alpha"), "--alpha");
        else if (k == "--min-hits")       a.min_hits     = parseInt(next("--min-hits"), "--min-hits");
        else if (k == "--high-thresh")    a.high_thresh  = parseFloat(next("--high-thresh"), "--high-thresh");
        else if (k == "--match-thresh")   a.match_thresh = parseFloat(next("--match-thresh"), "--match-thresh");
        else if (k == "--track-buffer")   a.track_buffer = parseInt(next("--track-buffer"), "--track-buffer");
        else if (k == "--track-json")     a.track_json   = next("--track-json");
        else if (k == "-h" || k == "--help") { printUsage(); std::exit(0); }
        else { std::cerr << "unknown arg: " << k << "\n"; return false; }
    }
    return !a.source.empty() && !a.model.empty();
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage();
        return 2;
    }

    cv::VideoCapture cap;
    try {
        int cam_idx = std::stoi(args.source);
        cap.open(cam_idx);
    } catch (const std::exception&) {
        cap.open(args.source);
    }
    if (!cap.isOpened()) {
        std::cerr << "Cannot open source: " << args.source << "\n";
        return 1;
    }

    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double input_fps = cap.get(cv::CAP_PROP_FPS);
    if (input_fps <= 0) input_fps = 30.0;

    cv::VideoWriter writer;
    if (!args.save.empty()) {
        writer.open(args.save, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), input_fps, {w, h});
        if (!writer.isOpened()) std::cerr << "warning: failed to open writer for " << args.save << "\n";
    }

    std::unique_ptr<tl::TRTDetector> detector;
    try {
        detector.reset(new tl::TRTDetector(args.model, args.conf, args.imgsz));
    } catch (const std::exception& e) {
        std::cerr << "failed to init detector: " << e.what() << "\n";
        return 1;
    }

    std::unique_ptr<tl::TrackSmoother> tracker;
    std::ofstream track_json_out;
    if (args.track) {
        tl::TrackerConfig tcfg;
        tcfg.alpha         = args.alpha;
        tcfg.track_thresh  = args.conf;          // reuse detector conf as low cutoff
        tcfg.high_thresh   = args.high_thresh;
        tcfg.match_thresh  = args.match_thresh;
        tcfg.track_buffer  = args.track_buffer;
        tcfg.min_hits      = args.min_hits;
        tcfg.frame_rate    = static_cast<int>(input_fps);
        try {
            tracker = std::make_unique<tl::TrackSmoother>(tcfg);
        } catch (const std::exception& e) {
            std::cerr << "failed to init tracker: " << e.what() << "\n";
            return 1;
        }
        if (!args.track_json.empty()) {
            track_json_out.open(args.track_json);
            if (!track_json_out.is_open()) {
                std::cerr << "failed to open --track-json path: " << args.track_json << "\n";
                return 1;
            }
        }
    }

    std::cout << "Input: " << args.source << " (" << w << "x" << h << " @ "
              << input_fps << " fps)\n"
              << "Model: conf=" << args.conf << " imgsz=" << detector->imgsz() << "\n";
    if (args.track) {
        std::cout << "Tracker: alpha=" << args.alpha << " min_hits=" << args.min_hits
                  << " high_thresh=" << args.high_thresh << " buffer=" << args.track_buffer
                  << "\n";
    }
    std::cout << "Press 'q' to quit\n";

    cv::Mat frame;
    int frame_count = 0;
    double total_s = 0.0;
    float fps = 0.f;

    while (cap.read(frame)) {
        auto t0 = std::chrono::steady_clock::now();
        auto dets = detector->detect(frame);

        std::vector<tl::TrackedDetection> tracks;
        if (tracker) tracks = tracker->update(dets, frame_count);

        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        total_s += dt;
        ++frame_count;
        fps = static_cast<float>(frame_count / total_s);

        if (frame_count % 30 == 0) {
            std::cout << "Frame " << frame_count << ": " << dets.size() << " det";
            if (tracker) std::cout << ", " << tracks.size() << " tracks";
            std::cout << ", " << (dt * 1000.0) << " ms (" << fps << " FPS avg)\n";
        }

        if (track_json_out.is_open()) {
            writeTrackJson(track_json_out, frame_count - 1, tracks);
        }

        if (args.show || writer.isOpened()) {
            cv::Mat vis = frame.clone();
            if (tracker) drawTracks(vis, tracks);
            else         drawDetections(vis, dets);
            drawFps(vis, fps);
            if (writer.isOpened()) writer.write(vis);
            if (args.show) {
                cv::imshow("Traffic Light Detection", vis);
                int key = cv::waitKey(1) & 0xFF;
                if (key == 'q') break;
            }
        }
    }

    if (frame_count > 0) {
        double avg_ms = total_s / frame_count * 1000.0;
        std::cout << "\nSummary: " << frame_count << " frames, avg " << avg_ms
                  << " ms/frame (" << fps << " FPS)\n";
    }
    return 0;
}
